# 第 4 讲：模型实现（下）— Transformer 与 CVAE

> 对应源文件：`policies/models/act/modeling_act.py` 第 189–591 行

## 学习目标

- 逐行理解 Transformer Encoder/Decoder 的 Pre-Norm 实现
- 掌握 CVAE 的编码、重参数化、KL 散度计算的完整推导
- 理解 ACTModel.forward() 中 9 个步骤的数据流动

---

## 4.1 ACTEncoderLayer — 编码器层

### 4.1.1 `__init__` — 构建 Self-Attention + FFN

```python
class ACTEncoderLayer(nn.Module):
    """Transformer Encoder Layer - 与 LeRobot 一致"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
```

**逐行解释**：

- `nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)`：PyTorch 内置的多头注意力。
  - `embed_dim = config.hidden_dim = 512`：输入/输出特征维度
  - `num_heads = config.num_attention_heads = 8`：每个头的维度 = 512/8 = 64
  - `dropout = 0.1`：在注意力权重矩阵上应用 dropout（正则化）
  - `batch_first = True`：输入/输出格式为 `[batch, seq_len, hidden_dim]`（而非 PyTorch 旧默认的 `[seq_len, batch, hidden_dim]`）

```python
        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)    # 512 → 3200
        self.dropout = nn.Dropout(config.dropout)                               # p=0.1
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)    # 3200 → 512
```

**逐行解释**：

- FFN（Feed-Forward Network）的结构：`LinearUp → Activation → Dropout → LinearDown`。这是 Transformer 标准的两层全连接子层。

- `linear1`：将隐藏维度从 512 扩展到 3200（约 6.25 倍），提供更大的表示容量。
- `dropout`：在激活函数之后、线性投影之前应用丢弃正则化。
- `linear2`：将 FFN 中间维度 3200 压缩回 512，恢复为 Transformer 的残差连接兼容的维度。

```python
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
```

**逐行解释**：

- `LayerNorm`：层归一化，对每个样本的每个 token 独立归一化（对所有特征维度）。与 BatchNorm 在整个 batch 上归一化不同，LayerNorm 不依赖 batch 大小，在 batch 大小变化时更稳定。
- `norm1`：Attention 子层之前的归一化
- `norm2`：FFN 子层之前的归一化
- `dropout1`：Attention 输出之后的 dropout
- `dropout2`：FFN 输出之后的 dropout

```python
        self.activation = F.gelu
        self.pre_norm = True  # LeRobot 使用 pre-norm
```

**逐行解释**：

- `F.gelu`：GELU 激活函数。与 `nn.GELU()` 不同，`F.gelu` 是函数式的——不包含可学习参数，不需要存储在 state_dict 中。这是一个微小的优化：当模块只使用 `F.xxx` 而非 `nn.XXX()`，模型的参数量统计更精确。

- `self.pre_norm = True`：标志位，指示使用的是 **Pre-Norm**（归一化在子层之前）而非 Post-Norm（归一化在子层之后）。
  - Pre-Norm：$x + \text{SubLayer}(\text{LayerNorm}(x))$
  - Post-Norm：$\text{LayerNorm}(x + \text{SubLayer}(x))$

  Pre-Norm 在现代 Transformer 中更流行，因为它在训练初期梯度更稳定，不需要 warmup 学习率调度。

### 4.1.2 `forward()` — 编码器层的前向传播

```python
def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    skip = x
    x = self.norm1(x)
    q = k = x if pos_embed is None else x + pos_embed
    x, _ = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
    x = skip + self.dropout1(x)
```

**逐行解释**（Self-Attention 子层 + 残差连接）：

1. `skip = x`：保存输入作为残差连接的 skip connection。这是 ResNet 的核心创新——直接传递梯度回到更早的层。

2. `x = self.norm1(x)`：Pre-Norm。在 Attention 前先归一化。LayerNorm 的计算：
   $$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   其中 $\mu, \sigma^2$ 在最后一维（hidden_dim=512）上计算。

3. `q = k = x if pos_embed is None else x + pos_embed`：构建 Query 和 Key，如果提供了位置编码则加到 Query/Key 上。注意 Value 不使用位置编码（`value=x`，不带 pos_embed）。这是因为：
   - Query 和 Key 用于计算注意力权重（"哪里需要注意"），位置信息帮助模型知道"token A 和 token B 的空间关系"
   - Value 是实际被聚合的语义信息，不需要位置编码叠加

4. `x, _ = self.self_attn(q, k, value=x, ...)`：多头自注意力。返回两个值：注意力输出和注意力权重矩阵（后者用 `_` 丢弃）。Self-Attention 的公式：
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
   其中 $d_k = 512/8 = 64$（每个头的维度）。

5. `x = skip + self.dropout1(x)`：残差连接 + Dropout 正则化。$x_{out} = x_{in} + \text{Dropout}(\text{Attention}(\text{Norm}(x_{in})))$

```python
    skip = x
    x = self.norm2(x)
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    x = skip + self.dropout2(x)
    return x
```

**逐行解释**（FFN 子层 + 残差连接）：

1. `skip = x`：再次保存残差
2. `x = self.norm2(x)`：Pre-Norm（第二次）
3. `self.linear1(x)`：Linear(512→3200)，扩展维度
4. `self.activation(...)`：GELU 激活，引入非线性。GELU 公式：
   $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
5. `self.dropout(...)`：在激活后丢弃（防止过拟合）
6. `self.linear2(...)`：Linear(3200→512)，恢复到原始维度
7. `x = skip + self.dropout2(x)`：残差连接 + Dropout

---

## 4.2 ACTEncoder — 编码器堆栈

```python
class ACTEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.num_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)
```

**逐行解释**：

- `is_vae_encoder: bool = False`：标志位（尽管当前代码未使用，但保留了扩展性）。未来可能会用不同的层数或配置构建 VAE 专用的 encoder。

- `nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])`：创建 `num_layers` 个相同的 encoder 层。注意是 `ModuleList` 而非 Python `list`——`ModuleList` 会正确注册子模块，使它们的参数出现在 `model.parameters()` 中。

- `self.norm = nn.LayerNorm(config.hidden_dim)`：编码器输出后的最终 LayerNorm。在 Pre-Norm 架构中，最后一层只有残差没有后归一化，需要显式添加这个最终 norm。

```python
def forward(self, x, pos_embed=None, key_padding_mask=None):
    for layer in self.layers:
        x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
    x = self.norm(x)
    return x
```

**逐行解释**：

- `for layer in self.layers`：顺序通过每一层。Transformer 的 encoder 是自回归无关的（所有 token 同时处理），所以不需要像 RNN 那样的循环状态传递。

- `x = self.norm(x)`：最后的归一化。在 Pre-Norm 架构中，子层内部做了 norm，但最终输出可能没有 norm。这里显式添加确保输出分布稳定。

---

## 4.3 ACTDecoderLayer — 解码器层

### 4.3.1 `__init__` — 构建 Self-Attention + Cross-Attention + FFN

```python
class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
```

**逐行解释**：

- `self_attn`：解码器的**自注意力**（Self-Attention）。用于处理 decoder 内部的 action query tokens。注意这里没有使用 causal mask（`attn_mask`），因为 action queries 之间是全连接的——所有 8 个 action query 可以同时看到彼此。

- `multihead_attn`：**交叉注意力**（Cross-Attention）。将 decoder 的 action queries 与 encoder 的视觉/状态特征进行匹配。Query 来自 decoder，Key/Value 来自 encoder。

  这是 Transformer 的核心机制——Encoder 产生"理解"，Decoder 根据这个理解"生成"。

```python
        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
```

**逐行解释**：

- 相比 EncoderLayer（2 个 norm + 2 个 dropout），DecoderLayer 有 **3 个** norm/dropout 对：
  - `norm1/dropout1`：Self-Attention 子层
  - `norm2/dropout2`：Cross-Attention 子层
  - `norm3/dropout3`：FFN 子层

### 4.3.2 `maybe_add_pos_embed` — 位置编码辅助方法

```python
def maybe_add_pos_embed(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor]) -> torch.Tensor:
    return x if pos_embed is None else x + pos_embed
```

**逐行解释**：

- 这是一个工具方法，提取了"如果位置编码存在就加上"的逻辑。避免在 `forward` 中重复写条件判断。

- `x + pos_embed`：利用 PyTorch 的广播机制——即使 `pos_embed` 的 batch/dim 维度不同，只要兼容就能相加。

### 4.3.3 `forward()` — 解码器层的前向传播

```python
def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
    # 1. Self-Attention
    skip = x
    x = self.norm1(x)
    q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
    x, _ = self.self_attn(q, k, value=x)
    x = skip + self.dropout1(x)
```

**逐行解释**（Self-Attention 子层）：

- `q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)`：Query 和 Key 加上 decoder 的位置编码。Decoder 的 action queries 需要知道"我是第几个动作"——例如第一个 query 对应 chunk 中第 1 步动作。

```python
    # 2. Cross-Attention
    skip = x
    x = self.norm2(x)
    x, _ = self.multihead_attn(
        query=self.maybe_add_pos_embed(x, decoder_pos_embed),
        key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
        value=encoder_out,
    )
    x = skip + self.dropout2(x)
```

**逐行解释**（Cross-Attention 子层）：

- `query`：来自 decoder（action queries），带上 decoder 位置编码。
- `key`：来自 encoder 输出（视觉+状态特征），带上 encoder 位置编码。
- `value`：来自 encoder 输出（**不带**位置编码的原始内容信息）。

  Cross-Attention 的直觉：
  - "Query（decoder）：'我应该预测什么样的动作？'"
  - "Key（encoder）：'我看到了什么视觉特征和状态信息？'"
  - "Attention 权重表示：'这个动作位置应该关注图像的哪个区域/状态的哪个维度'"
  - "Value：实际的图像和状态内容"

```python
    # 3. FFN
    skip = x
    x = self.norm3(x)
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    x = skip + self.dropout3(x)
    return x
```

**解释**：与 EncoderLayer 相同的 FFN 子层，不再赘述。

---

## 4.4 ACTDecoder — 解码器堆栈

```python
class ACTDecoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x
```

**逐行解释**：

- 与 ACTEncoder 结构相似，但 forward 需要额外接收 `encoder_out` 参数（用于 Cross-Attention）。

- `if self.norm is not None`：防御性检查。虽然 `__init__` 总是创建了 `self.norm`，但这个检查使得子类可以安全地重写 `__init__` 不创建 norm。

---

## 4.5 ACTModel — 完整模型

这是整个 ACT 系统的核心，将所有组件组装在一起。

### 4.5.1 `__init__` — 组装所有子模块

```python
class ACTModel(nn.Module):
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
```

**解释**：初始化视觉和状态编码器。视觉使用 ResNet18，状态使用 MLP。

```python
        # CVAE 编码器 (仅当 use_cvae=True 时)
        if config.use_cvae:
            self.action_encoder = nn.Linear(
                config.action_chunk_size * config.action_dim,
                config.hidden_dim
            )
            self.vae_output_proj = nn.Linear(config.hidden_dim, config.latent_dim * 2)
            self.latent_query = nn.Embedding(1, config.hidden_dim)
```

**逐行解释**（CVAE 的三个核心组件）：

- `action_encoder = Linear(chunk_size * action_dim, hidden_dim)`：将完整动作 chunk 展平后投影到隐藏维度。例如 8×2=16 → 512。这个编码器只在训练时使用——它将"正确答案"编码为潜变量的条件信息。

- `vae_output_proj = Linear(hidden_dim, latent_dim * 2)`：将隐藏表示投影为潜变量分布的参数——均值 $\mu$（前 `latent_dim` 维）和 log 方差（后 `latent_dim` 维）。`* 2` 是因为要同时输出 $\mu$ 和 $\log\sigma^2$。
  $$h = \text{GELU}(\text{action\_encoder}(a_{flattened}))$$
  $$\mu = h[:, 0:d_z], \quad \log\sigma^2 = h[:, d_z:2d_z]$$

- `latent_query = Embedding(1, hidden_dim)`：一个可学习的 latent token。概念上等同于 ViT 的 `[CLS]` token 或 BERT 的 `[SEP]` token——它是一组可学习参数，作为 Transformer encoder 的第一输入，通过 SA 聚合整个序列的全局信息。

```python
        # Transformer Encoder
        self.encoder = ACTEncoder(config)

        # Transformer Decoder
        self.decoder = ACTDecoder(config)

        # 动作预测头
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
```

**逐行解释**：

- `self.action_head`：将 decoder 输出的 hidden_dim 维向量投影到 action_dim 维。对于每个 action query，输出一个动作向量。
  - 输入：`[B, chunk_size, 512]`
  - 输出：`[B, chunk_size, action_dim]`

```python
        # 图像 2D 位置编码
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.hidden_dim // 2)

        # Encoder 1D 位置编码 (使用可学习的 embedding)
        self.encoder_pos_embed = nn.Embedding(128, config.hidden_dim)  # 最大 128 个位置

        # Decoder 位置编码
        self.decoder_pos_embed = nn.Embedding(config.action_chunk_size, config.hidden_dim)
```

**逐行解释**（位置编码的三种形式）：

- `encoder_cam_feat_pos_embed`：2D 正弦位置编码（用于图像特征图内部）。固定不可学习。

- `encoder_pos_embed = Embedding(128, 512)`：可学习的 1D 位置编码。128 是预设的最大序列长度（latent(1) + state(1) + visual(H'×W')，对于 7×7 特征图 = 51 tokens，远小于 128）。

- `decoder_pos_embed = Embedding(chunk_size, hidden_dim)`：可学习的解码器位置编码，每个 action query 一个。

```python
        # Latent 投影
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
```

**逐行解释**：

- `latent_proj = Linear(32, 512)`：将 CVAE 潜变量 $z$（维度 32）投影到 Transformer 的隐藏维度（512）。这使得 latent token 可以和其他 token（状态、视觉）在同一个向量空间内交互。

```python
        # CVAE 推理时使用的 latent 统计（训练后设置）
        self.register_buffer('_inference_latent_mu', torch.zeros(1, config.latent_dim))
        self.register_buffer('_inference_latent_log_sigma', torch.zeros(1, config.latent_dim))
        self._has_inference_latent = False
```

**逐行解释**：

- `register_buffer('_inference_latent_mu', ...)`：注册为 PyTorch buffer。buffer 是模型状态的一部分（会被 `state_dict()` 保存），但不是可训练参数（不会被 optimizer 更新）。

- `torch.zeros(1, latent_dim)`：初始化为零向量。在训练完成后，`set_inference_latent()` 会用训练数据中收集的统计量更新这些值。

- `self._has_inference_latent = False`：标志位——模型加载后需要调用 `set_inference_latent()` 才会设置为 True。未经设置的模型在推理时会使用噪声先验（`randn * 0.1`）。

### 4.5.2 `_reset_parameters()` — 参数初始化

```python
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
```

**逐行解释**：

- **关键设计决策**：`_reset_parameters()` 不重置 `vision_encoder.backbone`（ResNet18）。这是因为 ResNet18 已经用 ImageNet 预训练权重初始化了，重新随机初始化会丢失迁移学习的效果。

- 被初始化的模块列表包含了所有新增的可学习层。

```python
    for module in modules:
        for name, p in module.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name:
                nn.init.ones_(p)
```

**逐行解释**（三种初始化策略）：

1. `if p.dim() > 1`：矩阵参数（如 Linear.weight, Embedding.weight）——使用 **Xavier 均匀初始化**：
   $$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$
   Xavier 初始化保持了前向和反向传播中的方差，使得深层网络训练更稳定。

2. `elif "bias" in name`：偏置参数——初始化为 0。这是标准做法，偏差在训练初期对输出没有贡献。

3. `elif "weight" in name`：1D 权重（如 LayerNorm 的 weight）——初始化为 1。LayerNorm 初始化为恒等映射。

### 4.5.3 CVAE 推理潜变量管理

```python
def set_inference_latent(self, mu: torch.Tensor, log_sigma: torch.Tensor):
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    if log_sigma.ndim == 1:
        log_sigma = log_sigma.unsqueeze(0)
    self.register_buffer('_inference_latent_mu', mu)
    self.register_buffer('_inference_latent_log_sigma', log_sigma)
    self._has_inference_latent = True
```

**逐行解释**：

- `if mu.ndim == 1: mu = mu.unsqueeze(0)`：将 `[latent_dim]` 提升为 `[1, latent_dim]`。统一为 batch 格式。

- `register_buffer`：保存推理统计量。这些值在训练后通过统计训练集中的 $\mu$ 和 $\log\sigma^2$ 获得（见 `train_act.py` 中的 `latent_collection_epochs` 逻辑）。

```python
def clear_inference_latent(self):
    self._has_inference_latent = False
    self.register_buffer('_inference_latent_mu', torch.zeros(1, self.config.latent_dim))
    self.register_buffer('_inference_latent_log_sigma', torch.zeros(1, self.config.latent_dim))
```

**解释**：清除推理统计量，恢复为零向量。在加载新模型或想要使用纯噪声先验时使用。

### 4.5.4 CVAE 编码/采样/KL 方法

```python
def _encode_action(self, action_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = action_target.shape[0]
    action_flat = action_target.reshape(batch_size, -1)
    h = F.gelu(self.action_encoder(action_flat))
    latent_params = self.vae_output_proj(h)
    mu = latent_params[:, :self.config.latent_dim]
    log_sigma_x2 = latent_params[:, self.config.latent_dim:]
    return mu, log_sigma_x2
```

**逐行解释**：

- `action_flat = action_target.reshape(batch_size, -1)`：将 `[B, k, d_a]` 展平为 `[B, k*d_a]`。例如 `[8, 8, 2]` → `[8, 16]`。

- `F.gelu(self.action_encoder(action_flat))`：`[B, k*d_a] → [B, hidden_dim]`。编码动作序列的特征表示。

- `self.vae_output_proj(h)`：`[B, hidden_dim] → [B, latent_dim*2]`。投影为分布参数。

- `mu = latent_params[:, :latent_dim]`：前 `latent_dim` 维是均值 $\mu$。

- `log_sigma_x2 = latent_params[:, latent_dim:]`：后 `latent_dim` 维是 log 方差 $\log\sigma^2$。注意变量名中的 `x2` 表示"$\sigma^2$ 的 log"。
  - 为什么用 $\log\sigma^2$ 而非 $\sigma$？因为 $\log\sigma^2 \in (-\infty, +\infty)$ 是无约束的，适合神经网络直接输出。而 $\sigma$ 必须是正数，需要通过 softplus 等操作保证（增加了复杂度）。

```python
def _sample_latent(self, mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
    sigma = (log_sigma_x2 / 2).exp()
    eps = torch.randn_like(mu)
    z = mu + sigma * eps
    return z
```

**逐行解释**（**重参数化技巧 Reparameterization Trick**）：

- `sigma = (log_sigma_x2 / 2).exp()`：从 $\log\sigma^2$ 计算 $\sigma$：
  $$\sigma = \exp\left(\frac{\log\sigma^2}{2}\right) = \sqrt{\exp(\log\sigma^2)} = \sqrt{\sigma^2}$$

- `eps = torch.randn_like(mu)`：从 $\mathcal{N}(0, I)$ 采样噪声。

- `z = mu + sigma * eps`：重参数化采样。$z = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

  重参数化的意义：将随机性（$\epsilon$）与可学习参数（$\mu, \sigma$）分离。梯度可以通过 $\mu$ 和 $\sigma$ 回传，但 $\epsilon$ 不接收梯度（因为它来自固定的随机过程）。这使得 VAE 可以使用标准反向传播训练。

```python
def _compute_kl_loss(self, mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
    return kl.sum(-1).mean()
```

**逐行解释**（**KL 散度解析解**）：

两个高斯分布之间的 KL 散度有解析形式。$KL(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$：

$$KL = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

其中 $\log\sigma^2$ 就是我们的 `log_sigma_x2`。

- `mu.pow(2)`：$\mu^2$
- `log_sigma_x2.exp()`：$\exp(\log\sigma^2) = \sigma^2$
- 对于每个潜变量维度 $j$：$KL_j = -\frac{1}{2}(1 + \log\sigma^2_j - \mu^2_j - \sigma^2_j)$
- `.sum(-1)`：对所有潜变量维度求和：$KL = \sum_{j=1}^{d_z} KL_j$
- `.mean()`：在 batch 维度上取平均

最终返回值是一个标量（scalar），表示这个 batch 的平均 KL 散度。

### 4.5.5 `forward()` — 完整前向传播（9 步）

这是整个模型最核心的方法。我们逐步拆解。

```python
def forward(self, images, state, action_target=None, infer_cvae=True):
    batch_size = images.shape[0]
```

**解释**：`infer_cvae=True` 表示推理时会采样潜变量 z（随机性推理）。训练时传入 `infer_cvae=False`，因为 z 的采样在 `_encode_action` 中已经处理。

#### 步骤 1：处理 latent (CVAE)

```python
    mu = None
    log_sigma_x2 = None

    if self.config.use_cvae and action_target is not None and self.training:
        # 训练模式：从 action target 编码获取 latent
        mu, log_sigma_x2 = self._encode_action(action_target)
        latent = self._sample_latent(mu, log_sigma_x2)
```

**逐行解释**：

- 条件 `use_cvae and action_target is not None and self.training`：三个条件同时满足时才进入 CVAE 编码分支：
  1. CVAE 开关开启
  2. 提供了目标动作（训练时）
  3. 模型处于训练模式

- 训练时：编码器 $q(z|o, a)$ 将观测和动作映射为潜变量分布，然后重参数化采样。

```python
    elif self.config.use_cvae and infer_cvae and self._has_inference_latent:
        # 推理模式（已设置 latent 分布）：从存储的分布采样
        inf_mu = self._inference_latent_mu.to(images.device)
        inf_log_sig = self._inference_latent_log_sigma.to(images.device)
        if batch_size > 1:
            inf_mu = inf_mu.expand(batch_size, -1)
            inf_log_sig = inf_log_sig.expand(batch_size, -1)
        latent = self._sample_latent(inf_mu, inf_log_sig)
```

**逐行解释**：

- `.to(images.device)`：确保分布参数在正确的设备上。

- `if batch_size > 1: .expand(batch_size, -1)`：当 batch 推理时（如并行评估多个场景），将 `[1, latent_dim]` 扩展到 `[batch_size, latent_dim]`。`.expand` 不复制内存（只是修改 stride），比 `.repeat` 更高效。

```python
    elif self.config.use_cvae and infer_cvae:
        # 推理模式（未设置 latent 分布）：使用零向量并添加噪声
        latent = torch.randn(batch_size, self.config.latent_dim, device=images.device) * 0.1
    else:
        latent = torch.zeros(batch_size, self.config.latent_dim, device=images.device)
```

**逐行解释**：

- `torch.randn(...) * 0.1`：当没有训练好的 latent 统计量时，使用缩小到 0.1 倍的标准正态噪声。乘以 0.1 是为了限制噪声范围——未经训练的 latent 可能产生极端的动作输出。

- `torch.zeros(...)`：当 CVAE 禁用时，latent 为全 0 向量（退化为确定性模型）。

#### 步骤 2：视觉编码

```python
    vision_features = self.vision_encoder(images)  # [B, H*W, hidden_dim]
```

#### 步骤 3：状态编码

```python
    state_features = self.state_encoder(state)  # [B, 1, hidden_dim]
```

#### 步骤 4：Latent 投影

```python
    latent_features = self.latent_proj(latent).unsqueeze(1)  # [B, 1, hidden_dim]
```

**解释**：`Linear(32→512)` 投影后加 `.unsqueeze(1)` 使其成为 `[B, 1, 512]` 格式，与状态 token 维度统一。

#### 步骤 5：构建 Encoder 输入序列

```python
    encoder_in = torch.cat([latent_features, state_features, vision_features], dim=1)
    # [B, 1 + 1 + H'*W', hidden_dim] = [B, 2+49, 512] = [B, 51, 512]
```

**逐行解释**：

- 拼接顺序：`[latent(1), state(1), vision(49)]`。这个顺序不是任意的——它反映了信息流的层次：
  - latent token 在最前面：作为"全局上下文查询"，通过 Self-Attention 聚合后续所有 token 的信息
  - state token 紧随其后：状态与动作直接相关
  - vision tokens 在最后：空间信息是动作预测的基础

  实际上，由于 Self-Attention 是全连接的，token 的顺序（在没有位置编码的情况下）并不影响信息传播。但加上位置编码后，嵌入的位置信息会赋予不同位置的 token 不同的"角色"。

#### 步骤 6：构建位置编码

```python
    seq_len = encoder_in.shape[1]
    if seq_len <= self.encoder_pos_embed.num_embeddings:
        pos_embed = self.encoder_pos_embed.weight[:seq_len].unsqueeze(0)
    else:
        pos_embed = self.encoder_pos_embed.weight.repeat(
            (seq_len // self.encoder_pos_embed.num_embeddings) + 1, 1
        )[:, :seq_len]
```

**逐行解释**：

- `self.encoder_pos_embed.weight[:seq_len]`：从 Embedding 权重矩阵中取前 `seq_len` 行。Embedding 的权重形状是 `[128, 512]`，取 `[:51]` 表示只用前 51 个位置编码。

- `.unsqueeze(0)`：添加 batch 维度：`[51, 512] → [1, 51, 512]`。这利用广播使得后续可以直接与 `[B, 51, 512]` 相加。

- `else` 分支：如果序列长度超过 128（不太可能），使用循环重复位置编码。

#### 步骤 7：Transformer Encoder

```python
    encoder_out = self.encoder(encoder_in, pos_embed=pos_embed)
```

**解释**：编码器处理整个序列，输出 `[B, 51, 512]`。每个 token 现在都包含了全局上下文信息（通过 Self-Attention）。

#### 步骤 8：Transformer Decoder

```python
    decoder_pos_embed = self.decoder_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

    decoder_in = torch.zeros(
        batch_size, self.config.action_chunk_size, self.config.hidden_dim,
        device=images.device
    ) + decoder_pos_embed

    decoder_out = self.decoder(
        decoder_in, encoder_out,
        decoder_pos_embed=decoder_pos_embed,
        encoder_pos_embed=pos_embed,
    )
```

**逐行解释**：

- `decoder_pos_embed = Embedding.weight.unsqueeze(0).expand(B, -1, -1)`：`[k, 512] → [1, k, 512] → [B, k, 512]`

- `decoder_in = torch.zeros(...) + decoder_pos_embed`：Decoder 的初始输入是**纯位置编码**（全零内容 + 位置编码）。这与 NLP Transformer 不同——NLP 中 decoder 输入是"已生成的部分序列"。在 ACT 中，所有 k 个动作是**一次并行输出**的（非自回归），所以 8 个 action queries 同时以全零的内容开始，仅靠位置编码区分彼此。

- `self.decoder(...)`：Cross-Attention 使得 action queries 从 encoder 输出中提取相关信息，经过 FFN 加工后输出 `[B, k, 512]`。

#### 步骤 9：预测动作与计算 KL 损失

```python
    action_pred = self.action_head(decoder_out)  # [B, k, 512] → [B, k, action_dim]

    kl_loss = None
    if self.config.use_cvae and mu is not None and log_sigma_x2 is not None:
        kl_loss = self._compute_kl_loss(mu, log_sigma_x2)

    return {
        "action": action_pred,
        "mu": mu,
        "log_sigma_x2": log_sigma_x2,
        "kl_loss": kl_loss,
    }
```

**逐行解释**：

- `action_head = Linear(512, action_dim)`：每个 action query 独立投影为动作向量。
- 返回字典包含所有可能需要的中间输出。训练时使用 `action` 和 `kl_loss`；分析时可能用到 `mu` 和 `log_sigma_x2`。

### 4.5.6 `get_action()` — 推理接口

```python
def get_action(self, images, state, use_temporal_ensembling=False,
               temporal_ensembler=None, noise=0.0):
    self.eval()

    with torch.no_grad():
        output = self.forward(images, state, action_target=None, infer_cvae=True)
        actions = output["action"]  # [batch, chunk_size, action_dim]

        if noise > 0:
            actions = actions + torch.randn_like(actions) * noise

        if use_temporal_ensembling and temporal_ensembler is not None:
            action = temporal_ensembler.update(actions)
        else:
            action = actions[:, 0]

    return action
```

**逐行解释**：

- `self.eval()`：将模型设为评估模式。这会：
  1. 关闭 Dropout（所有神经元都参与计算）
  2. 确保 BatchNorm 使用运行统计量而非 batch 统计量
  3. 不影响 CVAE，因为 `forward` 中的路径由 `self.training` 和 `infer_cvae` 参数控制

- `torch.no_grad()`：禁用梯度计算，节省内存和计算量。

- `noise > 0`：可选的动作噪声。在探索性场景（如在线学习）中注入噪声帮助探索。

- `action = actions[:, 0]`：无 Temporal Ensembling 时，只取 chunk 的第一个动作。

- `return action`：返回形状 `[B, action_dim]` 的单步动作。

### 4.5.7 `reset_temporal_ensembler()` — 重置集成器

```python
def reset_temporal_ensembler(self, temporal_ensembler):
    temporal_ensembler.reset()
```

**解释**：在每次 episode 开始时调用，清除历史累积。否则上一个 episode 的动作预测会污染新 episode。

---

## 课后思考

1. 为什么 decoder 使用全零的 content + 位置编码作为初始输入，而不是像 NLP Transformer 那样使用已生成的部分序列？
2. 如果去掉 `_reset_parameters()` 中"跳过 ResNet backbone"的逻辑，模型性能会如何变化？
3. CVAE 的 KL 散度公式中，为什么 $\log\sigma^2$ 放在前面是 `+` 号，$\sigma^2$ 放在后面是 `-` 号？推导一下这个公式。
