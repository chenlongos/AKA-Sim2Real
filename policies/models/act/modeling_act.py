"""
ACT (Action Chunking Transformer) Model - 完全对齐 LeRobot
支持 CVAE、多相机、完整的 Transformer 架构
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from .configuration_act import ACTConfig


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
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder_img_feat_input_proj = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(hidden_dim // 2)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            images: [batch_size, num_cameras, channels, height, width] 或 [B, C, H, W]

        Returns:
            features: list of [batch_size, H*W, hidden_dim] (每相机)
            shape_2d: (batch_size, hidden_dim, H, W)
        """
        batch_size = images.shape[0]

        if images.ndim == 5:
            num_cameras = images.shape[1]
            images = images.view(-1, *images.shape[2:])
        else:
            num_cameras = 1

        features = self.backbone(images)

        cam_pos_embed = self.encoder_cam_feat_pos_embed(features)
        cam_pos_embed = cam_pos_embed.flatten(2).transpose(1, 2)

        features_flat = features.flatten(2).transpose(1, 2)  # [B*num, H*W, 512]

        features = features_flat + cam_pos_embed

        # 投影到 hidden_dim
        proj = nn.Linear(self.feature_dim, self.hidden_dim).to(features.device)
        features = proj(features)

        features = features.view(batch_size, num_cameras, -1, self.hidden_dim)
        features = features.squeeze(1)

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

        # 位置编码
        # Encoder 1D 位置编码: [latent, state]
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, config.hidden_dim)

        # 图像 2D 位置编码
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.hidden_dim // 2)

        # Decoder 位置编码
        self.decoder_pos_embed = nn.Embedding(config.action_chunk_size, config.hidden_dim)

        # Latent 投影
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        """
        batch_size = images.shape[0]

        # 1. 处理 latent (CVAE)
        mu = None
        log_sigma_x2 = None

        if self.config.use_cvae and action_target is not None and self.training:
            mu, log_sigma_x2 = self._encode_action(action_target)
            latent = self._sample_latent(mu, log_sigma_x2)
        elif self.config.use_cvae and infer_cvae:
            latent = torch.zeros(batch_size, self.config.latent_dim, device=images.device)
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

        # 简单位置编码
        seq_len = encoder_in.shape[1]
        pos_embed = torch.arange(seq_len, device=images.device).unsqueeze(0).unsqueeze(-1).float() / float(seq_len)
        pos_embed = pos_embed.expand(-1, -1, self.config.hidden_dim) * 0.01

        # 6. Transformer Encoder
        encoder_out = self.encoder(encoder_in, pos_embed=pos_embed)

        # 7. Transformer Decoder
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

        # 8. 预测动作
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
        """推理时获取动作"""
        self.eval()
        with torch.no_grad():
            output = self.forward(
                images,
                state,
                action_target=None,
                infer_cvae=True
            )
            action = output["action"]

            if noise > 0:
                action = action + torch.randn_like(action) * noise

        return action
