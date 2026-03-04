"""
ACT (Action Chunking Transformer) Model Implementation
基于 LeRobot 的 PyTorch 实现 - 完整版
包含 CVAE、Spatial Softmax 和 Temporal Ensembling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Tuple, Dict
import math
from .configuration_act import ACTConfig
from .ACTDataset import ACTDataset


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings - 与 LeRobot 一致"""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # 归一化到 [0, 2π]
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax - 保留图像特征图中的空间关键点信息
    这是 ACT 视觉编码器的关键组件
    """

    def __init__(
        self,
        height: int = 7,
        width: int = 7,
        temperature: float = 1.0,
        learned_temperature: bool = False,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        self.learned_temperature = learned_temperature

        # 创建坐标网格
        # 生成 -1 到 1 之间的归一化坐标
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing='xy'
        )
        # 堆叠成 [2, height, width]
        pos = torch.stack([pos_x, pos_y], dim=0)
        self.register_buffer('pos', pos)

        # 可学习的温度参数
        if learned_temperature:
            self.log_temperature = nn.Parameter(torch.zeros(1))
        else:
            self.log_temperature = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, channels, height, width]

        Returns:
            softmax_features: [batch_size, channels * 2] (x, y 坐标加权)
        """
        batch_size, channels, height, width = features.shape

        # 确保特征图尺寸匹配
        if height != self.height or width != self.width:
            features = F.interpolate(
                features,
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=True
            )

        # 展平空间维度
        features = features.view(batch_size, channels, -1)  # [B, C, H*W]
        features = F.softmax(features, dim=-1)

        # 获取温度
        if self.learned_temperature:
            temperature = self.log_temperature.exp()
        else:
            temperature = self.temperature

        # 计算加权的 x, y 坐标
        pos = self.pos  # [2, H, W]
        pos = pos.view(2, -1)  # [2, H*W]

        # 期望位置 = sum(features * position)
        expected_x = (features * pos[0:1]).sum(dim=-1)  # [B, C]
        expected_y = (features * pos[1:2]).sum(dim=-1)  # [B, C]

        # 拼接 x, y 坐标
        spatial_features = torch.cat([expected_x, expected_y], dim=-1)  # [B, C*2]

        return spatial_features / temperature


class RGBEncoder(nn.Module):
    """
    视觉编码器 - 支持 Spatial Softmax
    使用预训练的 ResNet18 作为骨干网络
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        hidden_dim: int = 512,
        pretrained: bool = True,
        use_spatial_softmax: bool = True,
        spatial_softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.use_spatial_softmax = use_spatial_softmax

        # 使用 ResNet18 作为视觉骨干
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        resnet = resnet18(weights=weights)

        # 移除最后的分类层，获取特征图
        # ResNet18 的特征图尺寸是 7x7 (对于 224x224 输入)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        self.feature_size = 7  # ResNet18 最后一层特征图大小

        # Spatial Softmax
        if use_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(
                height=self.feature_size,
                width=self.feature_size,
                temperature=spatial_softmax_temperature,
                learned_temperature=True,
            )
            # Spatial softmax 输出 2 * 512 = 1024 维
            self.spatial_proj = nn.Linear(self.feature_dim * 2, hidden_dim)
        else:
            self.spatial_softmax = None
            # 简单的平均池化
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(self.feature_dim, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, num_cameras, channels, height, width]
                   或 [batch_size, channels, height, width]

        Returns:
            features: [batch_size, 1, hidden_dim] 或 [batch_size, num_cameras, hidden_dim]
        """
        batch_size = images.shape[0]

        # 处理多相机输入
        if images.ndim == 5:
            # [batch_size, num_cameras, channels, height, width]
            num_cameras = images.shape[1]
            images = images.view(-1, *images.shape[2:])  # [batch_size * num_cameras, C, H, W]
        else:
            num_cameras = 1

        # 提取特征
        features = self.backbone(images)  # [B, 512, 7, 7]

        if self.use_spatial_softmax:
            # Spatial Softmax: [B, 512, 7, 7] -> [B, 1024]
            features = self.spatial_softmax(features)  # [B, 1024]
            # 投影到 hidden_dim
            features = self.spatial_proj(features)  # [B, hidden_dim]
        else:
            # 简单平均池化
            features = self.avgpool(features)  # [B, 512, 1, 1]
            features = features.squeeze(-1).squeeze(-1)  # [B, 512]
            features = self.projection(features)  # [B, hidden_dim]

        # 恢复批量维度
        if images.ndim == 5:  # 多相机输入
            features = features.view(batch_size, num_cameras, -1)  # [B, num_cameras, hidden_dim]

            # 如果有多个相机，对特征取平均
            if num_cameras > 1:
                features = features.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
            else:
                features = features.unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            features = features.unsqueeze(1)  # [B, 1, hidden_dim]

        return features


class CVAEEncoder(nn.Module):
    """
    CVAE 编码器 - 从未来动作序列推断隐变量 z 的分布
    这是 ACT 的核心组件，用于建模人类演示中的多模态分布
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 观察上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 隐变量分布参数预测
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        action_sequence: torch.Tensor,
        observation_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action_sequence: [batch_size, action_chunk_size, action_dim]
            observation_context: [batch_size, hidden_dim] 可选的观察上下文

        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        batch_size = action_sequence.shape[0]

        # 编码动作序列
        action_features = self.action_encoder(action_sequence)  # [B, chunk, hidden]

        # 对动作序列取平均
        action_features = action_features.mean(dim=1)  # [B, hidden_dim]

        # 如果提供了观察上下文，将其融合
        if observation_context is not None:
            context = self.context_encoder(observation_context)  # [B, hidden_dim]
            combined = action_features + context  # 残差连接
        else:
            combined = action_features

        # 预测分布参数
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 - 从分布中采样

        Args:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]

        Returns:
            z: [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class CVAEDecoder(nn.Module):
    """
    CVAE 解码器 - 从隐变量 z 和观察生成动作
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 将隐变量投影到 hidden_dim
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # 融合隐变量和观察特征
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        latent_z: torch.Tensor,
        observation_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent_z: [batch_size, latent_dim]
            observation_features: [batch_size, seq_len, hidden_dim]

        Returns:
            action: [batch_size, action_dim]
        """
        batch_size = latent_z.shape[0]

        # 投影隐变量
        z_features = self.latent_proj(latent_z)  # [B, hidden_dim]

        # 获取观察特征的平均
        if observation_features.ndim == 3:
            obs_features = observation_features.mean(dim=1)  # [B, hidden_dim]
        else:
            obs_features = observation_features  # [B, hidden_dim]

        # 融合
        combined = torch.cat([z_features, obs_features], dim=-1)  # [B, hidden_dim * 2]
        output = self.fusion(combined)  # [B, hidden_dim]

        return output


class StateEncoder(nn.Module):
    """
    状态编码器 - 处理非图像状态（关节位置、速度等）
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # 多层感知机
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim] 或 [batch_size, seq_len, state_dim]

        Returns:
            features: [batch_size, hidden_dim] 或 [batch_size, seq_len, hidden_dim]
        """
        original_shape = state.shape
        is_3d = state.ndim == 3

        if is_3d:
            batch_size, seq_len, state_dim = state.shape
            state = state.view(-1, state_dim)

        features = self.encoder(state)

        if is_3d:
            features = features.view(batch_size, seq_len, -1)

        return features


class ACTModel(nn.Module):
    """
    ACT (Action Chunking Transformer) 模型 - 完整版

    架构:
    1. 视觉编码器 (RGBEncoder): 处理图像输入，支持 Spatial Softmax
    2. 状态编码器 (StateEncoder): 处理关节状态
    3. CVAE 编码器 (CVAEEncoder): 从未来动作推断隐变量 z (训练时)
    4. CVAE 解码器 (CVAEDecoder): 从隐变量 z 和观察生成动作
    5. Transformer 编码器: 进一步处理融合特征
    6. Transformer 解码器: 输出动作序列
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # 视觉编码器
        self.vision_encoder = RGBEncoder(
            image_size=config.image_size,
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            pretrained=True,
            use_spatial_softmax=config.use_spatial_softmax,
            spatial_softmax_temperature=config.spatial_softmax_temperature,
        )

        # 状态编码器
        self.state_encoder = StateEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
        )

        # CVAE 组件
        if config.use_cvae:
            self.cvae_encoder = CVAEEncoder(
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
            )
            self.cvae_decoder = CVAEDecoder(
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
            )

        # 融合投影层
        self.fusion_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # 隐变量投影（用于添加到查询中）
        if config.use_cvae:
            self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.dim_feedforward,  # 使用配置中的值 (3200)
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.dim_feedforward,  # 使用配置中的值 (3200)
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )

        # 动作预测头
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

        # 2D 图像位置编码 - 与 LeRobot 一致
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.hidden_dim // 2)

        # 图像特征投影层
        self.encoder_img_feat_input_proj = nn.Conv2d(512, config.hidden_dim, kernel_size=1)

        # 状态 1D 位置编码
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, config.hidden_dim)  # latent + state

        # Temporal Ensembling 状态
        self.register_buffer('prev_action', None)
        self.temporal_ensembling_alpha = config.temporal_ensembling_weight

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        action_target: Optional[torch.Tensor] = None,
        infer_cvae: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 修改为 LeRobot 架构 (使用 Cross-Attention)

        Args:
            images: [batch_size, num_cameras, channels, height, width]
            state: [batch_size, state_dim]
            action_target: [batch_size, action_chunk_size, action_dim] (训练时提供)
            infer_cvae: 是否使用 CVAE

        Returns:
            output_dict: 包含 'action' 和其他中间结果
        """
        batch_size = images.shape[0]

        # 确保 state 是 3D
        if state.ndim == 2:
            state = state.unsqueeze(1)

        # ========== 1. 视觉编码 (使用 Spatial Softmax 或 特征图 + 2D Pos Embed) ==========
        vision_features = self.vision_encoder(images)  # [B, 1, hidden_dim]

        # ========== 2. 状态编码 ==========
        state_features = self.state_encoder(state)  # [B, 1, hidden_dim]

        # ========== 3. 融合并过 Transformer Encoder ==========
        fused = torch.cat([vision_features, state_features], dim=-1)
        fused = self.fusion_proj(fused)  # [B, 2, hidden_dim]

        encoder_out = self.transformer_encoder(fused)  # [B, 2, hidden_dim]

        # ========== 4. CVAE 处理 (不使用) ==========
        latent_z = None
        kl_loss = None
        observation_context = encoder_out.mean(dim=1)  # [B, hidden_dim]

        # ========== 5. Transformer Decoder (Cross-Attention 到 encoder 输出) ==========
        action_chunk_size = self.config.action_chunk_size
        hidden_dim = self.config.hidden_dim

        # 直接创建位置编码，确保在同一设备上
        decoder_pos_embed = torch.randn(1, action_chunk_size, hidden_dim, dtype=encoder_out.dtype, device=encoder_out.device) * 0.02
        decoder_pos_embed = decoder_pos_embed.expand(batch_size, -1, -1)  # [B, chunk, hidden]

        # 解码器输入 (零初始化 + 位置编码)
        decoder_in = torch.zeros(
            (batch_size, action_chunk_size, hidden_dim),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )  # [B, chunk, hidden]

        # Cross-attention: decoder queries attend to encoder output (memory)
        decoder_out = self.transformer_decoder(
            decoder_in + decoder_pos_embed,  # [B, chunk, hidden]
            memory=encoder_out,  # encoder output as memory
        )  # [B, chunk, hidden]

        # ========== 6. 预测动作 ==========
        action_pred = self.action_head(decoder_out)  # [B, chunk, action_dim]

        result = {
            "action": action_pred,
            "vision_features": vision_features,
            "state_features": state_features,
            "memory": encoder_out,
        }

        if latent_z is not None:
            result["latent_z"] = latent_z
        if kl_loss is not None:
            result["kl_loss"] = kl_loss

        return result

        if latent_z is not None:
            result["latent_z"] = latent_z
        if kl_loss is not None:
            result["kl_loss"] = kl_loss

        return result

    def get_action(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        use_temporal_ensembling: bool = True,
        noise: float = 0.0,
    ) -> torch.Tensor:
        """
        推理时获取动作 - 支持 Temporal Ensembling

        Args:
            images: [batch_size, num_cameras, channels, height, width]
            state: [batch_size, state_dim]
            use_temporal_ensembling: 是否使用时间集成
            noise: 添加到动作的噪声（用于探索）

        Returns:
            actions: [batch_size, action_chunk_size, action_dim]
        """
        self.eval()
        with torch.no_grad():
            # 推理时不使用 CVAE（因为没有未来动作）
            output = self.forward(images, state, action_target=None, infer_cvae=False)
            action = output["action"]

            # Temporal Ensembling: 与上一帧预测加权平均
            if use_temporal_ensembling and self.config.use_temporal_ensembling:
                if self.prev_action is not None:
                    # EMA 融合: alpha * prev + (1-alpha) * current
                    action = (
                        self.temporal_ensembling_alpha * self.prev_action
                        + (1 - self.temporal_ensembling_alpha) * action
                    )

                # 保存当前预测用于下一帧
                self.prev_action = action.clone()

            if noise > 0:
                action = action + torch.randn_like(action) * noise

        return action

    def reset_temporal_ensembling(self):
        """重置 Temporal Ensembling 状态"""
        self.prev_action = None


class ACTLoss(nn.Module):
    """
    ACT 损失函数 - 支持 CVAE 的 ELBO 损失
    """

    def __init__(
        self,
        action_chunk_size: int = 16,
        kl_weight: float = 0.1,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        kl_loss: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            pred_action: [batch_size, action_chunk_size, action_dim]
            target_action: [batch_size, action_chunk_size, action_dim]
            kl_loss: KL 散度损失 (CVAE)
            weight: 可选的权重

        Returns:
            loss_dict: 包含总损失和各组成部分
        """
        # MSE 损失
        mse = self.mse_loss(pred_action, target_action)  # [B, chunk, dim]
        mse = mse.mean(dim=-1)  # [B, chunk]

        if weight is not None:
            mse = mse * weight

        reconstruction_loss = mse.mean()
        first_step_loss = mse[:, 0].mean()
        last_step_loss = mse[:, -1].mean()

        # 总损失
        total_loss = reconstruction_loss
        if kl_loss is not None:
            total_loss = total_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss if kl_loss is not None else torch.tensor(0.0),
            "first_step_loss": first_step_loss,
            "last_step_loss": last_step_loss,
            "per_step_loss": mse,
        }


class ACTTrainer:
    """
    ACT 模型训练器
    """

    def __init__(
        self,
        model: ACTModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[ACTLoss] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or ACTLoss(model.config.action_chunk_size)
        self.device = device

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        执行一个训练步骤
        """
        self.model.train()

        images = batch["observation"]["image"].to(self.device)
        state = batch["observation"]["state"].to(self.device)
        action = batch["action"].to(self.device)

        # 前向传播
        output = self.model(images, state, action, infer_cvae=True)

        # 计算损失
        loss_dict = self.loss_fn(
            output["action"],
            action,
            kl_loss=output.get("kl_loss"),
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        metrics = {
            "loss": loss_dict["loss"].item(),
            "reconstruction_loss": loss_dict["reconstruction_loss"].item(),
            "kl_loss": loss_dict["kl_loss"].item() if isinstance(loss_dict["kl_loss"], torch.Tensor) else loss_dict["kl_loss"],
            "first_step_loss": loss_dict["first_step_loss"].item(),
            "last_step_loss": loss_dict["last_step_loss"].item(),
        }

        return metrics

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        评估模型
        """
        self.model.eval()
        total_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["observation"]["image"].to(self.device)
                state = batch["observation"]["state"].to(self.device)
                action = batch["action"].to(self.device)

                output = self.model(images, state, action, infer_cvae=False)
                loss_dict = self.loss_fn(output["action"], action)

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v

                num_batches += 1

        for k in total_metrics:
            total_metrics[k] /= num_batches

        return total_metrics


def create_act_model(
    state_dim: int = 7,
    action_dim: int = 7,
    action_chunk_size: int = 16,
    hidden_dim: int = 512,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    num_attention_heads: int = 8,
    use_cvae: bool = True,
    use_temporal_ensembling: bool = True,
    use_spatial_softmax: bool = True,
    **kwargs,
) -> ACTModel:
    """
    创建 ACT 模型的便捷函数
    """
    config = ACTConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_attention_heads=num_attention_heads,
        use_cvae=use_cvae,
        use_temporal_ensembling=use_temporal_ensembling,
        use_spatial_softmax=use_spatial_softmax,
        **kwargs,
    )
    return ACTModel(config)


if __name__ == "__main__":
    # 测试完整版 ACT
    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_attention_heads=4,
        use_cvae=True,
        use_temporal_ensembling=True,
        use_spatial_softmax=True,
        latent_dim=32,
    )

    model = ACTModel(config)
    print(f"ACT 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 2
    images = torch.randn(batch_size, 1, 3, 224, 224)
    state = torch.randn(batch_size, 7)
    action_target = torch.randn(batch_size, 16, 7)

    output = model(images, state, action_target)
    print(f"输出动作形状: {output['action'].shape}")
    print(f"隐变量形状: {output.get('latent_z', 'N/A')}")
    print(f"KL 损失: {output.get('kl_loss', 'N/A')}")

    # 测试推理
    model.eval()
    action1 = model.get_action(images, state, use_temporal_ensembling=True)
    action2 = model.get_action(images, state, use_temporal_ensembling=True)
    print(f"推理动作形状: {action1.shape}")
    print("测试完成!")
