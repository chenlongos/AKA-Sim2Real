from typing import Tuple


class ACTConfig:
    """
    ACT 模型配置
    """

    def __init__(
        self,
        # 观察空间
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        state_dim: int = 7,  # 关节状态维度
        # 动作空间
        action_dim: int = 7,  # 动作维度
        # 模型参数
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        dim_feedforward: int = 3200,  # 根据 LeRobot: 3200
        # 动作分块
        action_chunk_size: int = 16,
        # 相机数量
        num_cameras: int = 1,
        # CVAE 参数
        latent_dim: int = 32,  # 隐变量 z 的维度
        use_cvae: bool = True,  # 是否使用 CVAE
        kl_weight: float = 0.1,  # KL 散度损失权重
        # Temporal Ensembling 参数
        use_temporal_ensembling: bool = True,  # 是否使用时间集成
        temporal_ensembling_weight: float = 0.5,  # 时间集成权重 (EMA系数)
        # Spatial Softmax 参数
        use_spatial_softmax: bool = True,  # 是否使用 Spatial Softmax
        spatial_softmax_temperature: float = 1.0,  # Spatial Softmax 温度参数
    ):
        self.image_size = image_size
        self.in_channels = in_channels
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward  # 前馈网络维度
        self.action_chunk_size = action_chunk_size
        self.num_cameras = num_cameras
        # CVAE
        self.latent_dim = latent_dim
        self.use_cvae = use_cvae
        self.kl_weight = kl_weight
        # Temporal Ensembling
        self.use_temporal_ensembling = use_temporal_ensembling
        self.temporal_ensembling_weight = temporal_ensembling_weight
        # Spatial Softmax
        self.use_spatial_softmax = use_spatial_softmax
        self.spatial_softmax_temperature = spatial_softmax_temperature
