"""
ACT (Action Chunking Transformer) Model - 完全对齐 LeRobot
支持 CVAE、多相机、完整的 Transformer 架构
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from .configuration_act import ACTConfig

logger = logging.getLogger(__name__)


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> torch.Tensor:
    """1D sinusoidal positional embeddings"""
    def get_position_angle_vec(position):
        return [position / math.pow(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = torch.tensor([
        get_position_angle_vec(pos_i) for pos_i in range(num_positions)
    ], dtype=torch.float32)
    sinusoid_table[:, 0::2] = sinusoid_table[:, 0::2].sin()
    sinusoid_table[:, 1::2] = sinusoid_table[:, 1::2].cos()
    return sinusoid_table


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings"""
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

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


class RGBEncoder(nn.Module):
    """视觉编码器 - LeRobot 风格"""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = 512

        from torchvision.models import resnet18, ResNet18_Weights
        try:
            weights = ResNet18_Weights.DEFAULT
            resnet = resnet18(weights=weights)
        except Exception as exc:
            logger.warning("Failed to load pretrained ResNet18 weights, falling back to random init: %s", exc)
            resnet = resnet18(weights=None)

        # 保留卷积特征图，避免全局池化后丢失空间信息。
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_img_feat_input_proj = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(hidden_dim // 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, num_cameras, channels, height, width] 或 [B, C, H, W]

        Returns:
            features: [batch_size, num_tokens, hidden_dim]
        """
        batch_size = images.shape[0]

        if images.ndim == 5:
            num_cameras = images.shape[1]
            images = images.view(-1, *images.shape[2:])
        else:
            num_cameras = 1

        features = self.backbone(images)  # [B*num_cameras, 512, H', W']

        cam_pos_embed = self.encoder_cam_feat_pos_embed(features)
        features = self.encoder_img_feat_input_proj(features) + cam_pos_embed
        features = features.flatten(2).transpose(1, 2)  # [B*num, H*W, hidden_dim]
        features = features.view(batch_size, num_cameras, -1, self.hidden_dim)
        features = features.reshape(batch_size, -1, self.hidden_dim)

        return features


class StateEncoder(nn.Module):
    """状态编码器"""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            state = state.unsqueeze(1)
        return self.encoder(state)


class ACTEncoderLayer(nn.Module):
    """Transformer Encoder Layer - 与 LeRobot 一致"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = F.gelu
        self.pre_norm = True  # LeRobot 使用 pre-norm

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = x
        x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x, _ = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = skip + self.dropout1(x)

        skip = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        return x


class ACTEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.num_encoder_layers  # VAE encoder 和主 encoder 用相同层数
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """Transformer Decoder Layer - 与 LeRobot 一致"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )

        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = F.gelu
        self.pre_norm = True

    def maybe_add_pos_embed(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor]) -> torch.Tensor:
        return x if pos_embed is None else x + pos_embed

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                decoder_pos_embed: Optional[torch.Tensor] = None,
                encoder_pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = x
        x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x, _ = self.self_attn(q, k, value=x)
        x = skip + self.dropout1(x)

        skip = x
        x = self.norm2(x)
        x, _ = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )
        x = skip + self.dropout2(x)

        skip = x
        x = self.norm3(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        return x


class ACTDecoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                decoder_pos_embed: Optional[torch.Tensor] = None,
                encoder_pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTModel(nn.Module):
    """
    ACT 模型 - 完全对齐 LeRobot
    支持 CVAE、改进的位置编码、更稳定的推理
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # 视觉编码器
        self.vision_encoder = RGBEncoder(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
        )

        # 状态编码器
        self.state_encoder = StateEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
        )

        # CVAE 编码器 (仅当 use_cvae=True 时)
        if config.use_cvae:
            self.action_encoder = nn.Linear(
                config.action_chunk_size * config.action_dim,
                config.hidden_dim
            )
            self.vae_output_proj = nn.Linear(config.hidden_dim, config.latent_dim * 2)
            self.latent_query = nn.Embedding(1, config.hidden_dim)

        # Transformer Encoder
        self.encoder = ACTEncoder(config)

        # Transformer Decoder
        self.decoder = ACTDecoder(config)

        # 动作预测头
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

        # 图像 2D 位置编码
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.hidden_dim // 2)

        # Encoder 1D 位置编码 (使用可学习的 embedding)
        self.encoder_pos_embed = nn.Embedding(128, config.hidden_dim)  # 最大 128 个位置

        # Decoder 位置编码
        self.decoder_pos_embed = nn.Embedding(config.action_chunk_size, config.hidden_dim)

        # Latent 投影
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # CVAE 推理时使用的 latent 统计（训练后设置）
        self.register_buffer('_inference_latent_mu', torch.zeros(1, config.latent_dim))
        self.register_buffer('_inference_latent_log_sigma', torch.zeros(1, config.latent_dim))
        self._has_inference_latent = False

        self._reset_parameters()

    def _reset_parameters(self):
        """只初始化新增层，保留视觉 backbone 的预训练权重。"""
        modules = [
            self.state_encoder,
            self.encoder,
            self.decoder,
            self.action_head,
            self.encoder_pos_embed,
            self.decoder_pos_embed,
            self.latent_proj,
            self.vision_encoder.encoder_img_feat_input_proj,
        ]
        if self.config.use_cvae:
            modules.extend([
                self.action_encoder,
                self.vae_output_proj,
                self.latent_query,
            ])

        for module in modules:
            for name, p in module.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
                elif "weight" in name:
                    nn.init.ones_(p)

    def set_inference_latent(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        设置 CVAE 推理时使用的 latent 分布参数
        应在加载模型后、推理前调用

        Args:
            mu: latent 均值 [latent_dim] 或 [batch, latent_dim]
            log_sigma: latent log 方差 [latent_dim] 或 [batch, latent_dim]
        """
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
        if log_sigma.ndim == 1:
            log_sigma = log_sigma.unsqueeze(0)
        self.register_buffer('_inference_latent_mu', mu)
        self.register_buffer('_inference_latent_log_sigma', log_sigma)
        self._has_inference_latent = True
        print(f"已设置推理 latent: mu={mu.mean().item():.4f}, log_sigma={log_sigma.mean().item():.4f}")

    def clear_inference_latent(self):
        """清除推理 latent，恢复到零向量"""
        self._has_inference_latent = False
        self.register_buffer('_inference_latent_mu', torch.zeros(1, self.config.latent_dim))
        self.register_buffer('_inference_latent_log_sigma', torch.zeros(1, self.config.latent_dim))

    def _encode_action(self, action_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """VAE 编码器"""
        batch_size = action_target.shape[0]
        action_flat = action_target.reshape(batch_size, -1)
        h = F.gelu(self.action_encoder(action_flat))
        latent_params = self.vae_output_proj(h)
        mu = latent_params[:, :self.config.latent_dim]
        log_sigma_x2 = latent_params[:, self.config.latent_dim:]
        return mu, log_sigma_x2

    def _sample_latent(self, mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
        """重参数化采样"""
        sigma = (log_sigma_x2 / 2).exp()
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z

    def _compute_kl_loss(self, mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
        """KL 散度损失"""
        kl = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
        return kl.sum(-1).mean()

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        action_target: Optional[torch.Tensor] = None,
        infer_cvae: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 与 LeRobot 一致

        Args:
            images: 图像张量 [batch, channels, height, width] 或 [batch, num_cameras, channels, height, width]
            state: 状态张量 [batch, state_dim]
            action_target: 目标动作（训练时使用）[batch, chunk_size, action_dim]
            infer_cvae: 是否使用 CVAE 推理
        """
        batch_size = images.shape[0]

        # 1. 处理 latent (CVAE)
        mu = None
        log_sigma_x2 = None

        if self.config.use_cvae and action_target is not None and self.training:
            # 训练模式：从 action target 编码获取 latent
            mu, log_sigma_x2 = self._encode_action(action_target)
            latent = self._sample_latent(mu, log_sigma_x2)
        elif self.config.use_cvae and infer_cvae and self._has_inference_latent:
            # 推理模式（已设置 latent 分布）：从存储的分布采样
            inf_mu = self._inference_latent_mu.to(images.device)
            inf_log_sig = self._inference_latent_log_sigma.to(images.device)
            # 如果 batch > 1，扩展维度
            if batch_size > 1:
                inf_mu = inf_mu.expand(batch_size, -1)
                inf_log_sig = inf_log_sig.expand(batch_size, -1)
            latent = self._sample_latent(inf_mu, inf_log_sig)
        elif self.config.use_cvae and infer_cvae:
            # 推理模式（未设置 latent 分布）：使用零向量并添加噪声
            latent = torch.randn(batch_size, self.config.latent_dim, device=images.device) * 0.1
        else:
            latent = torch.zeros(batch_size, self.config.latent_dim, device=images.device)

        # 2. 视觉编码 - 返回特征
        vision_features = self.vision_encoder(images)  # [B, H*W, hidden_dim]

        # 3. 状态编码
        state_features = self.state_encoder(state)  # [B, 1, hidden_dim]

        # 4. Latent 投影
        latent_features = self.latent_proj(latent).unsqueeze(1)  # [B, 1, hidden_dim]

        # 5. 构建 Encoder 输入 - [latent, state, image_features]
        encoder_in = torch.cat([latent_features, state_features, vision_features], dim=1)  # [B, 2+H*W, hidden_dim]

        # 6. 使用可学习的位置编码
        seq_len = encoder_in.shape[1]
        if seq_len <= self.encoder_pos_embed.num_embeddings:
            pos_embed = self.encoder_pos_embed.weight[:seq_len].unsqueeze(0)
        else:
            # 如果序列长度超过 embedding 大小，使用循环
            pos_embed = self.encoder_pos_embed.weight.repeat(1, (seq_len // self.encoder_pos_embed.num_embeddings) + 1, 1)[:, :seq_len]

        # 7. Transformer Encoder
        encoder_out = self.encoder(encoder_in, pos_embed=pos_embed)

        # 8. Transformer Decoder
        decoder_pos_embed = self.decoder_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        decoder_in = torch.zeros(
            batch_size, self.config.action_chunk_size, self.config.hidden_dim,
            device=images.device
        ) + decoder_pos_embed

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=decoder_pos_embed,
            encoder_pos_embed=pos_embed,
        )

        # 9. 预测动作
        action_pred = self.action_head(decoder_out)

        # 计算 KL 损失
        kl_loss = None
        if self.config.use_cvae and mu is not None and log_sigma_x2 is not None:
            kl_loss = self._compute_kl_loss(mu, log_sigma_x2)

        return {
            "action": action_pred,
            "mu": mu,
            "log_sigma_x2": log_sigma_x2,
            "kl_loss": kl_loss,
        }

    def get_action(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        use_temporal_ensembling: bool = False,
        noise: float = 0.0,
    ) -> torch.Tensor:
        """
        推理时获取动作

        Args:
            images: 图像张量
            state: 状态张量
            use_temporal_ensembling: 是否使用时序集成（多次推理取平均）
            noise: 添加到动作的噪声水平

        Returns:
            预测的动作序列 [action_chunk_size, action_dim]
        """
        self.eval()

        # 收集多次推理结果用于时序集成
        all_actions = []

        with torch.no_grad():
            num_ensembles = 3 if use_temporal_ensembling else 1

            for _ in range(num_ensembles):
                output = self.forward(
                    images,
                    state,
                    action_target=None,
                    infer_cvae=True
                )
                action = output["action"]

                if noise > 0:
                    action = action + torch.randn_like(action) * noise

                all_actions.append(action)

        # 时序集成：取平均
        if use_temporal_ensembling and len(all_actions) > 1:
            final_action = torch.stack(all_actions).mean(dim=0)
        else:
            final_action = all_actions[0]

        return final_action
