# 第 2 讲：数据集与归一化

> 对应源文件：`policies/models/act/ACTDataset.py`

## 学习目标

- 理解 ACT 数据集的滑窗构造逻辑
- 掌握 QUANTILES 百分位数归一化的数学原理
- 理解两种数据格式（单步 vs chunked）的兼容设计

---

## 2.1 导入

```python
from typing import Dict, Tuple, Optional
import torch
```

**解释**：`Dict`, `Tuple`, `Optional` 用于类型注解，提高代码可读性。`torch` 是 PyTorch 的核心库。

---

## 2.2 ACTDataset 类

### 2.2.1 类声明

```python
class ACTDataset(torch.utils.data.Dataset):
```

**解释**：继承 `torch.utils.data.Dataset`——这是 PyTorch 数据加载的标准基类。继承它后，只需要实现 `__len__` 和 `__getitem__` 两个方法，就可以使用 PyTorch 的 `DataLoader` 进行批量加载、shuffle、多进程加载等。

### 2.2.2 `__init__` 参数

```python
def __init__(
    self,
    data: Dict[str, torch.Tensor],
    action_chunk_size: int = 16,
    normalize_images: bool = True,
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    state_q01: Optional[torch.Tensor] = None,
    state_q99: Optional[torch.Tensor] = None,
    action_q01: Optional[torch.Tensor] = None,
    action_q99: Optional[torch.Tensor] = None,
):
```

**逐行解释**：

- `data: Dict[str, torch.Tensor]`：原始数据字典。有两种可能的键名格式：
  1. 扁平格式：`'observation.image'`、`'observation.state'`、`'action'`（来自 `load_dataset()`）
  2. 嵌套格式：`'observation'` 是字典，包含 `'image'` 和 `'state'`（来自自定义加载器）

- `action_chunk_size = 16`：动作分块大小。注意这里的默认值是 16（匹配 LeRobot/ALOHA），而 AKA-Sim2Real 默认配置中是 8。

- `normalize_images = True`：是否对图像做 ImageNet 标准化。不开启时图像像素值保持 [0, 1] 范围。

- `image_mean = (0.485, 0.456, 0.406)`：ImageNet 训练集的 RGB 三通道均值。这些值是 ImageNet 数据集上统计得到的，用于**均值减法**，使每个通道的均值为 0。

- `image_std = (0.229, 0.224, 0.225)`：ImageNet 训练集的 RGB 三通道标准差。用于**标准差除法**，使每个通道的标准差为 1。

  为什么要用 ImageNet 的统计量？因为 ResNet18 的预训练权重是在 ImageNet 上用这些归一化参数训练的。如果不用相同的归一化，ResNet18 提取的特征分布会偏离预训练时的分布，破坏迁移学习的效果。

- `state_q01/q99`：状态的 1% 和 99% 百分位数（详见 2.3 节）。

- `action_q01/q99`：动作的 1% 和 99% 百分位数。

### 2.2.3 图像归一化参数的张量化

```python
    self.data = data
    self.action_chunk_size = action_chunk_size

    # 图像归一化参数
    self.normalize_images = normalize_images
    self.image_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
    self.image_std = torch.tensor(image_std).view(1, 3, 1, 1)
```

**逐行解释**：

- `torch.tensor(image_mean)`：将 Python tuple 转换为 PyTorch 张量。不指定 dtype 时默认为 `float32`。

- `.view(1, 3, 1, 1)`：重塑为 `[1, 3, 1, 1]` 形状。这是为了和图像张量 `[C, H, W]` 做广播（broadcasting）运算：
  - 图像张量形状：`[3, H, W]`
  - `image_mean` 形状：`[1, 3, 1, 1]` —— 广播后等价于 `[3, H, W]`，每个通道的 H×W 像素共享同一个均值

  为什么不用 `.view(3, 1, 1)`？因为图像可能是 4 维（`[B, C, H, W]`）或 3 维（`[C, H, W]`），在第一个维度加一个 1 可以兼容两种格式，PyTorch 的广播机制会自动处理。实际上，在 `__getitem__` 中使用 `self.image_mean.to(images.device)` 时，广播机制确保了无论批量维度是否存在都能正确相减。

### 2.2.4 百分位数归一化参数的存储

```python
    # QUANTILES 归一化参数
    self.state_q01 = state_q01
    self.state_q99 = state_q99
    self.action_q01 = action_q01
    self.action_q99 = action_q99
```

**解释**：直接保存传入的百分位数张量。如果传入 `None`（表示不使用归一化），在 `__getitem__` 中会通过条件判断跳过归一化步骤。

### 2.2.5 动作数据格式的自适应

```python
    action_tensor = data["action"]
    if action_tensor.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported action tensor shape {tuple(action_tensor.shape)}; "
            "expected [T, action_dim] or [N, chunk_size, action_dim]."
        )
```

**逐行解释**：

- `.ndim`：张量的维度数（number of dimensions）。
  - `ndim=2`：形状 `[T, action_dim]`，单步动作格式（来自 LeRobot 数据集）
  - `ndim=3`：形状 `[N, chunk_size, action_dim]`，已分块格式（来自自定义数据）
  - 其他维度数：不支持，抛出 `ValueError`

- `tuple(action_tensor.shape)`：将 `torch.Size` 转换为 Python tuple，方便在错误信息中打印。

### 2.2.6 已分块格式的处理

```python
    self.actions_are_chunked = action_tensor.ndim == 3
    if self.actions_are_chunked:
        self.num_samples = action_tensor.shape[0]
        if action_tensor.shape[1] != action_chunk_size:
            raise ValueError(
                f"Configured action_chunk_size={action_chunk_size}, "
                f"but action data has chunk size {action_tensor.shape[1]}."
            )
```

**逐行解释**：

- `self.actions_are_chunked`：布尔标志位，在 `__getitem__` 中用于选择不同的索引逻辑。

- `self.num_samples = action_tensor.shape[0]`：已分块格式下，样本数 = 数据的第一维大小。

- `if action_tensor.shape[1] != action_chunk_size`：校验 chunk 大小一致性。如果你用训练脚本 `train_act.py --action_chunk_size 8` 但数据集是 chunk_size=16 的格式，这里会报错。

### 2.2.7 单步格式的处理（滑窗采样）

```python
    else:
        self.num_samples = action_tensor.shape[0] - action_chunk_size + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough timesteps ({action_tensor.shape[0]}) for "
                f"action_chunk_size={action_chunk_size}."
            )
```

**逐行解释**：

- `action_tensor.shape[0] - action_chunk_size + 1`：**滑窗采样的关键公式**。
  假设有 $T$ 个时间步的动作数据，chunk_size = 8：
  - 第 0 个样本：时间步 `[0, 7]`
  - 第 1 个样本：时间步 `[1, 8]`
  - ...
  - 最后一个样本：时间步 `[T-8, T-1]`
  - 总共 `T - 8 + 1` 个样本

- 如果 `T <= 8`（数据量不足一个 chunk），样本数为 0 或负数，直接报错。

### 2.2.8 `__len__` — 返回数据集大小

```python
def __len__(self) -> int:
    return self.num_samples
```

**解释**：PyTorch DataLoader 需要的标准接口。返回可采样的样本总数。

### 2.2.9 `__getitem__` — 取一个样本

这是数据集中最重要的方法。DataLoader 通过调用它来获取每个 batch 中的样本。

#### 观测数据的获取

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    # 获取当前时间步的观察（不是未来时刻！）
    current_idx = idx

    # 支持两种数据格式
    if "observation.image" in self.data:
        images = self.data["observation.image"][current_idx]
        state = self.data["observation.state"][current_idx]
    else:
        images = self.data["observation"]["image"][current_idx]
        state = self.data["observation"]["state"][current_idx]
```

**逐行解释**：

- `current_idx = idx`：**关键概念**——ACT 根据"当前观测"预测"未来动作"。这意味着：
  - 观测取的是时刻 `t`（`idx`）
  - 动作取的是时刻 `[t, t + chunk_size)`（下面会讲）

  有一类常见的错误实现是：用时刻 `t + chunk_size` 的观测来预测时刻 `t` 到 `t + chunk_size` 的动作——这导致模型"看到了未来"，是一个信息泄漏的 bug。本实现正确地对齐了观测和动作的时间。

- `if "observation.image" in self.data`：扁平格式 vs 嵌套格式的兼容分支。使用 `in` 操作符而不是 `try/except`，因为这是已知的两种格式，不需要异常处理的开销。

#### 动作数据的获取

```python
    if self.actions_are_chunked:
        action = self.data["action"][idx]
    else:
        # 动作是从当前时刻开始的未来 chunk_size 步
        action = self.data["action"][idx:idx + self.action_chunk_size]
```

**逐行解释**：

- 已分块格式：每个样本独立存储，`idx` 直接索引到对应的 chunk。
- 单步格式：使用切片 `[idx : idx + chunk_size]` 取连续的 chunk_size 步动作。这就是滑窗——相邻的样本有 `chunk_size - 1` 步的重叠。

#### 图像归一化

```python
    # 归一化图像
    if self.normalize_images:
        images = (images - self.image_mean.to(images.device)) / self.image_std.to(images.device)
```

**逐行解释**：

- `self.image_mean.to(images.device)`：将归一化参数移动到图像所在的设备（CPU 或 CUDA）。`to()` 方法对于 CPU 上的张量是空操作（no-op），所以没有性能损失。

- `(images - mean) / std`：标准归一化公式。对于每个通道 $(r, g, b)$：
  $$ r' = \frac{r - 0.485}{0.229}, \quad g' = \frac{g - 0.456}{0.224}, \quad b' = \frac{b - 0.406}{0.225} $$

  广播机制：`images` 形状为 `[H, W, 3]` 或 `[3, H, W]`，`mean/std` 形状为 `[1, 3, 1, 1]`，PyTorch 自动将后者广播到前者的形状。

#### QUANTILES 状态归一化

```python
    # 使用 QUANTILES 归一化状态: 2 * (x - q01) / (q99 - q01) - 1
    if self.state_q01 is not None and self.state_q99 is not None:
        q01 = self.state_q01.to(state.device)
        q99 = self.state_q99.to(state.device)
        denom = q99 - q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=state.device), denom)
        state = 2 * (state - q01) / denom - 1
```

**逐行解释**（这是本讲最核心的数学部分）：

- **公式**：$x' = 2 \cdot \frac{x - q_{01}}{q_{99} - q_{01}} - 1$

- **步骤分解**：
  1. $x - q_{01}$：减去 1% 百分位数，使得 1% 分位数处映射到 0
  2. $\frac{x - q_{01}}{q_{99} - q_{01}}$：除以极差（99% 分位 - 1% 分位），使得 99% 分位数处映射到 1
  3. $2 \cdot (\dots) - 1$：将 [0, 1] 区间线性映射到 [-1, 1]

- **为什么用百分位数而不是 min/max？**
  - Min/Max 对异常值极度敏感——一个离谱的数据点就能拉偏整个归一化
  - 百分位数（如 1%/99%）剔除了两端的极端异常值，归一化更鲁棒
  - 这是 LeRobot 的标准做法

- `torch.where(denom == 0, torch.tensor(1e-8, device=state.device), denom)`：**防止除零**。
  - 如果 `denom == 0`（q01 == q99，即数据没有变化），分母替换为 1e-8
  - 使用 `torch.where` 而非 `if` 语句，因为 `denom` 是张量，需要支持向量化条件

#### QUANTILES 动作归一化

```python
    # 使用 QUANTILES 归一化动作: 2 * (x - q01) / (q99 - q01) - 1
    if self.action_q01 is not None and self.action_q99 is not None:
        q01 = self.action_q01.to(action.device)
        q99 = self.action_q99.to(action.device)
        denom = q99 - q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=action.device), denom)
        action = 2 * (action - q01) / denom - 1
```

**解释**：与状态归一化完全相同的逻辑。唯一需要注意的是：动作经过归一化后输出在 [-1, 1] 范围内，这意味着模型的预测输出也是 [-1, 1] 范围，推理时需要**反归一化**才能用于实际控制。

#### 返回样本

```python
    return {
        "observation": {
            "image": images,
            "state": state,
        },
        "action": action,
    }
```

**解释**：返回一个嵌套字典。这个结构使得训练代码中可以通过 `batch["observation"]["image"]` 优雅地访问数据。

---

## 2.3 QUANTILES 归一化详解

### 什么是 QUANTILES 归一化？

标准归一化使用全局最小值/最大值：
$$ x' = \frac{x - x_{min}}{x_{max} - x_{min}} $$

QUANTILES 归一化使用百分位数：
$$ x' = 2 \cdot \frac{x - q_{01}}{q_{99} - q_{01}} - 1 $$

### 为什么是 [-1, 1]？

1. **对称性**：正值和负值对称，梯度更新更均衡
2. **与 tanh 激活兼容**：[-1, 1] 是 tanh 的自然输出范围
3. **神经网络友好**：输入分布在 0 附近时训练更稳定

### 百分位数的计算

在数据采集端（`backend/services/episode/`），统计量是这样计算的：

```python
# 伪代码
q01 = numpy.percentile(all_states, 1, axis=0)
q99 = numpy.percentile(all_states, 99, axis=0)
```

注意 99% 的分位值 < 实际的最大值——这意味着有少量数据点在归一化后会超出 [-1, 1] 范围。这是有意为之——允许模型"见过"超出常规范围的值，但又不让极端值主导归一化。

---

## 课后思考

1. 如果 `action_chunk_size=8` 且数据有 100 个时间步，`num_samples` 是多少？如果 `action_chunk_size=16` 呢？
2. 为什么需要在归一化时做"防止除零"检查？在什么场景下 `q01 == q99` 会发生？
3. 为什么不对图像使用 QUANTILES 归一化，而是用 ImageNet 的均值和标准差？
