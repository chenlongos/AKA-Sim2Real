# 第 3 讲：模型实现（上）— 位置编码与编码器

> 对应源文件：`policies/models/act/modeling_act.py` 第 1–187 行

## 学习目标

- 理解 2D sinusoidal 位置编码的数学推导
- 掌握 ResNet18 作为视觉 backbone 的改造方式
- 理解为什么视觉特征需要位置编码

---

## 3.0 导入与日志

```python
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from .configuration_act import ACTConfig

logger = logging.getLogger(__name__)
```

**逐行解释**：

- `math`：用于位置编码中的三角函数和指数计算（`sin`, `cos`, `pow`）
- `logging`：Python 标准日志库。`logging.getLogger(__name__)` 创建一个以当前模块名为名的 logger，便于日志分级和过滤
- `torch.nn as nn`：PyTorch 的神经网络模块。所有自定义层都继承 `nn.Module`
- `torch.nn.functional as F`：PyTorch 的函数式接口。提供 `F.gelu`, `F.l1_loss` 等无状态的激活/损失函数
- `Optional, Dict, Tuple, List`：类型注解
- `from .configuration_act import ACTConfig`：相对导入配置类

---

## 3.1 ACTTemporalEnsembler — 时间集成器

这是 ACT 论文 Algorithm 2 的精确实现。

### 3.1.1 类文档

```python
class ACTTemporalEnsembler:
    """Temporal Ensembling - LeRobot ACT 官方实现

    根据 Algorithm 2 of https://huggingface.co/papers/2304.13705

    权重计算: w_i = exp(-temporal_ensemble_coeff * i)，其中 w0 是最旧的动作
    权重归一化: 除以 Σw_i

    系数工作原理:
    - 设为 0: 所有动作均匀加权
    - 设为正数: 更重视旧动作
    - 设为负数: 更重视新动作

    默认值 0.01 (LeRobot ACT 原版) 会更重视旧动作。
    """
```

**逐行解释**：

关键公式：`w_i = exp(-c * i)`

其中 $c$ = `temporal_ensemble_coeff`，$i$ 表示该预测是第几个时间步之前产生的（$i=0$ 是最早的，$i=k-1$ 是最新的）。

- 当 $c > 0$（如 $c=0.01$）：$e^{-0.01 \cdot 0} = 1.0$（旧动作权重大），$e^{-0.01 \cdot 7} \approx 0.93$（新动作权重略小）。**更重视旧动作** → 输出更平滑稳定
- 当 $c = 0$：所有权重为 1 → **均匀加权**
- 当 $c < 0$（如 $c=-0.01$）：$e^{-(-0.01) \cdot 0} = 1.0$（旧动作权重小），$e^{-(-0.01) \cdot 7} \approx 1.07$（新动作权重大）。**更重视新动作** → 响应更快但抖动更大

### 3.1.2 `__init__` — 初始化权重

```python
def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
    self.chunk_size = chunk_size
    self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
    self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
    self.reset()
```

**逐行解释**：

- `torch.arange(chunk_size)`：创建 `[0, 1, 2, ..., chunk_size-1]` 的索引序列

- `-temporal_ensemble_coeff * torch.arange(chunk_size)`：计算每个位置上的指数参数。假设 coeff=0.01, chunk_size=8：
  $$[-0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]$$

- `torch.exp(...)`：逐元素指数运算，得到权重向量：
  $$w = [1.000, 0.990, 0.980, 0.970, 0.961, 0.951, 0.942, 0.932]$$

- `torch.cumsum(self.ensemble_weights, dim=0)`：计算累积和，用于在线更新时的归一化。第 $i$ 个元素 = $\sum_{j=0}^{i} w_j$。这是在线运行指数加权平均时的归一化分母。

- `self.reset()`：初始化在线计算状态（详见 3.1.3）。

### 3.1.3 `reset()` — 重置状态

```python
def reset(self):
    """重置在线计算变量"""
    self.ensembled_actions = None
    self.ensembled_actions_count = None
```

**逐行解释**：

- `self.ensembled_actions = None`：累积的动作张量。形状为 `[B, remaining_chunk, action_dim]`。设为 `None` 表示"尚未开始累积"。

- `self.ensembled_actions_count = None`：每个位置已经被累积了多少次。形状为 `[remaining_chunk, 1]`。当 count 达到 chunk_size 时，该位置不再更新（因为它已经累积了一轮完整的 chunk）。

### 3.1.4 `update()` — 在线更新（核心算法）

```python
def update(self, actions: torch.Tensor) -> torch.Tensor:
    """
    输入: (batch, chunk_size, action_dim) 的动作序列
    输出: (batch, action_dim) - 序列中的下一个动作

    更新所有时间步的 temporal ensemble，并返回下一个动作。
    """
    self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
    self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
```

**逐行解释**：

- `to(device=actions.device)`：将预计算的权重移动到与输入相同的设备（CPU/GPU）。这是必要的，因为模型可能在不同的设备上运行，权重需要跟随。

#### 第一次调用：初始化

```python
    if self.ensembled_actions is None:
        # 第一次调用：用第一个时间步的动作序列初始化
        self.ensembled_actions = actions.clone()
        self.ensembled_actions_count = torch.ones(
            (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
        )
```

**逐行解释**：

- `actions.clone()`：深拷贝动作张量。`clone()` 创建新的内存，断开与原始计算图的连接。在推理时（`torch.no_grad()` 下），`clone()` 和普通的赋值没有梯度问题，但保持一致性。

- `torch.ones((self.chunk_size, 1), dtype=torch.long)`：创建一个 `[chunk_size, 1]` 的全 1 计数张量。每个位置都被认为是"累积了 1 次"。

#### 后续调用：在线更新

```python
    else:
        # 在线更新: 对 (batch_size, chunk_size - 1, action_dim) 部分进行更新
        self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
        self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
        self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
        self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
```

**逐行解释**（这是整个 temporal ensembling 最核心的数学）：

这一段实现了在线指数加权移动平均（EWMA）。设我们有：
- $A_{n-1}$：之前累积的动作（`ensembled_actions`）
- $A_n$：新预测的动作（`actions[:, :-1]`——去掉最后一个，因为它没有历史对应）
- $c_{n-1}$：之前的累积次数（`ensembled_actions_count`）
- $w_i = e^{-m \cdot i}$：权重

在线更新公式为：
$$ A_n = \frac{A_{n-1} \cdot C_{c-1} + a_n \cdot w_{c}}{C_c} $$

其中 $C_c = \sum_{j=0}^{c} w_j$（累积权重和）。

**第一步**：
```python
self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
```
将旧累积值乘以旧累积权重和——相当于恢复**未归一化**的加权和。

**第二步**：
```python
self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
```
加上新预测乘以新权重——现在 `ensembled_actions` 是未归一化的加权和。

**第三步**：
```python
self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
```
除以新的累积权重和——完成归一化。

**第四步**：
```python
self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
```
计数 + 1，但不超过 `chunk_size`（因为累积 chunk_size 次后，第一个位置的信息已经被完全替代，不再需要增加计数）。

#### 拼接最后一个动作

```python
        # 最后一个动作（没有先前的在线平均）需要拼接到末尾
        self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
        self.ensembled_actions_count = torch.cat(
            [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
        )
```

**逐行解释**：

- `actions[:, -1:]`：取最后一个动作（索引 `-1`），保留维度（`:` 切片不降维）。形状 `[B, 1, action_dim]`。

- 为什么最后一个动作特殊对待？因为在线更新只更新了前 `chunk_size-1` 个位置（它们有历史对应），最新产生的第 `chunk_size` 个位置是全新的、没有历史的。所以它直接拼接到累积序列末尾，计数从 1 开始。

- `torch.ones_like(self.ensembled_actions_count[-1:])`：为新加入的位置创建计数 = 1。

#### 消费第一个动作

```python
    # "消费"第一个动作
    action, self.ensembled_actions, self.ensembled_actions_count = (
        self.ensembled_actions[:, 0],
        self.ensembled_actions[:, 1:],
        self.ensembled_actions_count[1:],
    )
    return action
```

**逐行解释**：

- `self.ensembled_actions[:, 0]`：取出第一个（最"成熟"、累积最久的）动作，形状 `[B, action_dim]`。这就是本次推理实际输出的动作。

- `self.ensembled_actions[:, 1:]`：剩下的动作（索引 1 到末尾）保留在状态中，等下次推理时继续累积。

- `self.ensembled_actions_count[1:]`：对应的计数也同步移出第一个。

- 这个"消费"模式是 temporal ensembling 的精髓——每次推理产出一个完整的 chunk，但只输出第一个位置（已经过多次累积），其余位置保留用于未来融合。

---

## 3.2 位置编码（Positional Encoding）

### 3.2.1 `create_sinusoidal_pos_embedding()` — 1D 正弦位置编码

```python
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
```

**逐行解释**（这是 Transformer 原论文 "Attention Is All You Need" 中位置编码的实现）：

- `get_position_angle_vec(position)`：对给定的位置，计算 `dimension` 个角频率。公式：
  $$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
  $$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

  其中 $pos$ 是位置索引，$i$ 是维度对索引（`hid_j // 2`），$d$ 是总维度。

- `math.pow(10000, 2 * (hid_j // 2) / dimension)`：计算频率的基数。注意 `hid_j // 2`（整除）使得偶数/奇数维度对共享同一个频率。

- `[:, 0::2]`（偶数索引）应用 `sin()`，`[:, 1::2]`（奇数索引）应用 `cos()`。这是标准做法——sin/cos 交替使得位置编码可以通过线性变换表达相对位置关系。

### 3.2.2 `ACTSinusoidalPositionEmbedding2d` — 2D 正弦位置编码

用于图像特征图的空间位置编码——这是视觉 Transformer 的标准做法。

#### `__init__`

```python
class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings"""
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000
```

**逐行解释**：

- `self._two_pi = 2 * math.pi`：$2\pi$，用于将空间坐标映射到完整的周期范围
- `self._eps = 1e-6`：一个小常数，防止除零
- `self._temperature = 10000`：与 1D 位置编码相同的温度基数

#### `forward()` — 2D 位置编码的前向传播

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    not_mask = torch.ones_like(x[0, :1])
    y_range = not_mask.cumsum(1, dtype=torch.float32)
    x_range = not_mask.cumsum(2, dtype=torch.float32)
```

**逐行解释**：

- `torch.ones_like(x[0, :1])`：创建一个与输入第一个通道同形状的全 1 张量。`x[0, :1]` 取第一个 batch 的第一个通道，形状 `[1, H', W']`。

- `not_mask.cumsum(1, dtype=torch.float32)`：沿高度维度（dim=1）计算累积和。由于输入是全 1，结果就是 `[1, 2, 3, ..., H']` 的 y 坐标网格。`dtype=torch.float32` 确保后续的浮点运算精度。

- `not_mask.cumsum(2, dtype=torch.float32)`：沿宽度维度（dim=2）计算累积和。结果就是 `[1, 2, 3, ..., W']` 的 x 坐标网格。

**思考**：这里通过 `cumsum` 而非直接 `torch.arange` 来生成坐标，是一种巧妙的方式——它利用了张量运算，避免额外的 reshape 操作，并且代码更简洁。

```python
    y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
    x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
```

**逐行解释**：

- `y_range[:, -1:, :]`：取每列的最大 y 值（`H'`）。用 `-1:` 而非 `-1` 保留维度，确保广播正确。

- `y_range / (y_range[:, -1:, :] + self._eps)`：将 y 坐标归一化到 $[0, 1]$ 范围。加上 `_eps` 防止除零。

- `* self._two_pi`：映射到 $[0, 2\pi]$ 范围。这使得 sin/cos 能够在完整的周期上采样，产生丰富的频率成分。

```python
    inverse_frequency = self._temperature ** (
        2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
    )
```

**逐行解释**：

- `torch.arange(self.dimension)`：创建 `[0, 1, ..., dimension-1]` 的维度索引。

- `// 2`（整除）：将维度索引分组为对——`[0,0,1,1,2,2,...]`，使得偶数/奇数对共享同一频率。

- `self._temperature ** (2 * i / dimension)`：与 1D 位置编码相同的频率计算：$10000^{2i/d}$

```python
    x_range = x_range.unsqueeze(-1) / inverse_frequency
    y_range = y_range.unsqueeze(-1) / inverse_frequency
```

**解释**：将空间坐标的每个位置除以对应的频率——这是"位置 × 频率"的标准傅里叶特征构造。`.unsqueeze(-1)` 添加一个维度用于广播，结果形状为 `[1, H', W', dimension]`。

```python
    pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
    pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
    pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
```

**逐行解释**：

- `x_range[..., 0::2].sin()`：偶数索引维度应用 sin
- `x_range[..., 1::2].cos()`：奇数索引维度应用 cos
- `torch.stack(..., dim=-1)`：将 sin 和 cos 结果堆叠在最后一维，使得每个频率对在同一个维度上
- `.flatten(3)`：将最后一维（dim=3）的 sin/cos 对展平
- `torch.cat((pos_embed_y, pos_embed_x), dim=3)`：拼接 y 和 x 方向的位置编码
- `.permute(0, 3, 1, 2)`：将维度顺序从 `[B, H', W', D]` 转为标准的 `[B, D, H', W']` 特征图格式

```python
    return pos_embed
```

**解释**：返回形状 `[B, dimension, H', W']` 的 2D 位置编码，可以直接与特征图相加。

---

## 3.3 RGBEncoder — 视觉编码器

### 3.3.1 `__init__` — 加载 ResNet18

```python
class RGBEncoder(nn.Module):
    """视觉编码器 - LeRobot 风格"""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = 512
```

**逐行解释**：

- `self.feature_dim = 512`：ResNet18 最后一个卷积层输出的通道数。ResNet 架构中，各阶段输出通道为 64→128→256→512（最后一个阶段）。这里保留这个中间特征维度。

```python
        from torchvision.models import resnet18, ResNet18_Weights
        try:
            weights = ResNet18_Weights.DEFAULT
            resnet = resnet18(weights=weights)
        except Exception as exc:
            logger.warning("Failed to load pretrained ResNet18 weights, falling back to random init: %s", exc)
            resnet = resnet18(weights=None)
```

**逐行解释**：

- `from torchvision.models import resnet18, ResNet18_Weights`：懒导入（lazy import）——只在实例化 RGBEncoder 时才导入 torchvision。这避免了在不需要视觉编码器的场景下（如纯测试）导入重量级的 torchvision。

- `ResNet18_Weights.DEFAULT`：torchvision 0.14+ 的新权重 API。它会自动下载 ImageNet 预训练权重。与旧 API (`pretrained=True`) 相比，新 API 提供了更好的权重版本管理。

- Try/Except 回退策略：如果下载权重失败（如网络问题），使用随机初始化（`weights=None`）。这保证了代码在任何环境下都能运行，但会显著影响训练效果。使用 logger 发出警告让用户知道发生了什么。

```python
        # 保留卷积特征图，避免全局池化后丢失空间信息。
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
```

**逐行解释**（这是改造 ResNet 为特征提取器的关键步骤）：

- `list(resnet.children())`：将 ResNet 的各个子模块转换为列表。ResNet18 的结构：
  ```
  [0] Conv2d(3→64, 7×7, stride=2)
  [1] BatchNorm2d(64)
  [2] ReLU
  [3] MaxPool2d(3×3, stride=2)
  [4] BasicBlock(64→64)  ×2  -- layer1
  [5] BasicBlock(64→128) ×2  -- layer2
  [6] BasicBlock(128→256) ×2 -- layer3
  [7] BasicBlock(256→512) ×2 -- layer4
  [8] AdaptiveAvgPool2d    -- 去掉
  [9] Linear(512→1000)     -- 去掉
  ```

- `[:-2]`：移除最后两个子模块——`AdaptiveAvgPool2d` 和 `Linear`。这两个模块会将特征图压缩为 1D 分类向量，而我们需要的恰恰是空间特征图。

- `nn.Sequential(*...)`：将修改后的子模块重新包装为 `Sequential`，便于前向传播。

- 最终 `backbone` 的输出形状：`[B, 512, H/32, W/32]`（224×224 输入 → 7×7 特征图）。

```python
        self.encoder_img_feat_input_proj = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(hidden_dim // 2)
```

**逐行解释**：

- `nn.Conv2d(512, 512, kernel_size=1)`：1×1 卷积，也称为"逐点卷积"（pointwise convolution）。它的作用是：
  1. 将 ResNet 特征维度（512）投影到 Transformer 的隐藏维度（也是 512）
  2. 即使用了相同的维度数，1×1 卷积仍然提供了可学习的跨通道混合
  3. 不改变空间分辨率（kernel_size=1，无 padding 损失）

- `ACTSinusoidalPositionEmbedding2d(hidden_dim // 2)`：2D 位置编码，输出维度 = 256（= 512/2）。为什么是 half？因为 pos_embed_y 和 pos_embed_x 各占一半，拼接后总维度 = 512。

### 3.3.2 `forward()` — 视觉编码

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    batch_size = images.shape[0]
```

**解释**：记录原始 batch size，用于后续的 `view`/`reshape` 操作中恢复批量维度。

```python
    if images.ndim == 5:
        num_cameras = images.shape[1]
        images = images.view(-1, *images.shape[2:])
    else:
        num_cameras = 1
```

**逐行解释**：

- `images.ndim == 5`：多相机输入，形状 `[B, num_cameras, C, H, W]`
- `images.ndim == 4`：单相机输入，形状 `[B, C, H, W]`
- `images.view(-1, *images.shape[2:])`：将 `[B, num_cameras, C, H, W]` 展开为 `[B*num_cameras, C, H, W]`。`-1` 表示自动推断该维度大小，`*images.shape[2:]` 保持 `[C, H, W]` 形状不变。

```python
    features = self.backbone(images)  # [B*num_cameras, 512, H', W']
```

**解释**：通过改造后的 ResNet18 提取特征。输出形状 `[B*num_cameras, 512, H', W']`，其中 H' = H/32, W' = W/32。

```python
    cam_pos_embed = self.encoder_cam_feat_pos_embed(features)
    features = self.encoder_img_feat_input_proj(features) + cam_pos_embed
```

**逐行解释**：

- `self.encoder_cam_feat_pos_embed(features)`：计算 2D 正弦位置编码，形状 `[B*num_cam, 256, H', W']`。但投影卷积期望 512 通道——这里有个细节：在 `ACTSinusoidalPositionEmbedding2d` 中，y 和 x 方向各 256 维，拼接后是 512 维。但我重新检查代码，`ACTSinusoidalPositionEmbedding2d(dimension)` 的输出维度 = `2 * dimension`（因为拼接了 y 和 x），所以 `dimension = hidden_dim // 2 = 256`，输出 = 512 维。正确！

- `self.encoder_img_feat_input_proj(features)`：1×1 卷积投影，输出也是 512 通道。

- `+ cam_pos_embed`：**残差式的位置编码加法**。这不同于 NLP 中的"加到 token embedding 上"，这里是直接加到特征图的每个空间位置上，让模型知道"这个特征在图像的哪个位置"。

```python
    features = features.flatten(2).transpose(1, 2)  # [B*num, H*W, hidden_dim]
```

**逐行解释**：

- `flatten(2)`：将 `[B*num_cam, 512, H', W']` 展平为 `[B*num_cam, 512, H'*W']`
- `transpose(1, 2)`：转置为 `[B*num_cam, H'*W', 512]`——这是 Transformer 期望的序列格式：`[batch, seq_len, hidden_dim]`

```python
    features = features.view(batch_size, num_cameras, -1, self.hidden_dim)
    features = features.reshape(batch_size, -1, self.hidden_dim)
```

**逐行解释**：

- `view(batch_size, num_cameras, -1, self.hidden_dim)`：恢复批量维度，中间形状为 `[B, num_cam, H'*W', 512]`
- `reshape(batch_size, -1, self.hidden_dim)`：将多相机的 tokens 拼接为一个长序列 `[B, num_cam * H'*W', 512]`

```python
    return features
```

**解释**：返回视觉特征 tokens，形状 `[B, num_cam * H'*W', 512]`。对于单相机 224×224 输入，H'=W'=7，所以视觉 tokens 数 = 49。

---

## 3.4 StateEncoder — 状态编码器

```python
class StateEncoder(nn.Module):
    """状态编码器"""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
```

**逐行解释**：

- 这是一个两层 MLP（多层感知机）。
- `Linear(state_dim, hidden_dim)`：将低维状态向量投影到高维隐藏空间。例如 2 维 → 512 维。
- `GELU()`：Gaussian Error Linear Unit 激活函数。与 ReLU 相比，GELU 在 0 附近是平滑的，提供更好的梯度流。这是 Transformer 标准使用的激活函数。
- `Linear(hidden_dim, hidden_dim)`：保持维度的变换层，提供额外的非线性表达能力。

```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 2:
        state = state.unsqueeze(1)
    return self.encoder(state)
```

**逐行解释**：

- `if state.ndim == 2`：状态输入可能是 `[B, state_dim]`。`.unsqueeze(1)` 在 dim=1 位置插入一个维度，变为 `[B, 1, state_dim]`。
- `self.encoder(state)`：线性层作用于最后一维（`state_dim`），所以 `[B, 1, state_dim] → [B, 1, hidden_dim]` 是正确的。
- 最终输出形状：`[B, 1, hidden_dim]`——单一的状态 token，将与其他 tokens（latent、视觉）拼接后送入 Transformer。

**设计意图**：为什么状态编码这么简单（2 层 MLP），而视觉编码那么复杂（ResNet18 + 位置编码）？因为状态的维度很低（2 维，左右轮速度），信息量小，不需要深层网络。而图像包含丰富的空间信息，需要预训练 backbone 来提取有意义的特征。

---

## 课后思考

1. 为什么 `ACTSinusoidalPositionEmbedding2d` 中 y 和 x 各占一半的维度（`hidden_dim // 2`），而不是各占完整维度？
2. 为什么去掉 ResNet 的最后两层（AvgPool + Linear）而不是只去掉最后一层？
3. Temporal Ensembling 中 `torch.clamp(count + 1, max=chunk_size)` 为什么需要 `max=chunk_size`？如果去掉会发生什么？
