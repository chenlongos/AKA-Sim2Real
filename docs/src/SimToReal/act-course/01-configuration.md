# 第 1 讲：配置系统

> 对应源文件：`policies/models/act/configuration_act.py` + `policies/models/act/defaults.py`

## 学习目标

- 理解 ACT 模型所有超参数的含义和数学对应关系
- 掌握参数校验逻辑的设计原因
- 理解默认配置的"为什么"

---

## 1.1 ACTConfig — 模型配置数据类

`configuration_act.py` 定义了 `ACTConfig` 类，它是整个模型唯一的配置入口。

### 1.1.1 类型导入

```python
from typing import Tuple
```

**解释**：导入 `Tuple` 用于 `image_size` 参数的类型注解 `Tuple[int, int]`，明确表示图像尺寸是 "(高度, 宽度)"的二元组。

### 1.1.2 类声明与文档

```python
class ACTConfig:
    """
    ACT 模型配置 - 对齐 LeRobot
    """
```

**解释**：`@dataclass` 是 Python 的标准数据类装饰器，它会自动为类生成 `__init__`、`__repr__`、`__eq__` 等方法。使用它而不是 `dataclasses.dataclass` 装饰器的原因是：我们需要在 `__init__` 中添加自定义的参数校验逻辑（调用 `_validate()`），如果使用 `@dataclass` 装饰器，自定义 `__init__` 会比较复杂。

### 1.1.3 `__init__` 参数：观察空间

```python
def __init__(
    self,
    # 观察空间
    image_size: Tuple[int, int] = (224, 224),
    in_channels: int = 3,
    state_dim: int = 7,  # 关节状态维度
```

**逐行解释**：

- `image_size = (224, 224)`：输入图像的像素尺寸。ResNet18 的标准输入是 224×224（ImageNet 预训练权重对应此尺寸）。这里"(高, 宽)"的约定与 PyTorch 的 `[C, H, W]` 格式一致。
- `in_channels = 3`：输入图像的通道数。RGB 图像 = 3 个通道。ResNet18 的第一层卷积期望 3 通道。
- `state_dim = 7`：机器人关节状态的维度。在 AKA-Sim2Real 中实际只用 2 维（左右轮速度），但默认值 7 对齐了 LeRobot（ALOHA 机械臂有 7 个自由度）。

### 1.1.4 `__init__` 参数：动作空间

```python
    # 动作空间
    action_dim: int = 7,  # 动作维度
```

**解释**：`action_dim = 7` 是动作向量的维度。ALOHA 机械臂中每个手臂控制 7 个关节。在 AKA-Sim2Real 中实际只用 2 维（左右轮速度指令），通过 `defaults.py` 的覆盖实现。

### 1.1.5 `__init__` 参数：模型架构

```python
    # 模型参数
    hidden_dim: int = 512,
    num_attention_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dropout: float = 0.1,
    dim_feedforward: int = 3200,  # 根据 LeRobot: 3200
```

**逐行解释**（也是 Transformer 标准参数）：

- `hidden_dim = 512`：Transformer 中所有 token 的向量维度。注意这个数字必须能被 `num_attention_heads` 整除，因为多头注意力需要将 hidden_dim 均分给每个头：$d_{head} = d_h / n_{heads} = 512 / 8 = 64$。

- `num_attention_heads = 8`：多头自注意力的头数。每个头独立计算 Attention，然后拼接。8 个头是 Transformer 原论文的默认值。

- `num_encoder_layers = 6`：Encoder 堆叠层数。每层包含一层 Self-Attention + 一层 FFN。6 层是 Transformer 原论文的设置。

- `num_decoder_layers = 6`：Decoder 堆叠层数。每层包含一层 Masked Self-Attention + 一层 Cross-Attention + 一层 FFN。

- `dropout = 0.1`：Dropout 丢弃概率。在每个子层（Attention 和 FFN）之后应用，用于正则化。

- `dim_feedforward = 3200`：FFN 中间层的维度。注意这里不是 Transformer 原论文的 2048，而是 LeRobot 特有的 3200。FFN 的结构是：`Linear(512→3200) → GELU → Dropout → Linear(3200→512)`。

### 1.1.6 `__init__` 参数：动作分块

```python
    # 动作分块
    action_chunk_size: int = 16,
    n_action_steps: int = 16,  # 每次实际执行多少步（≤ chunk_size）
```

**逐行解释**：

- `action_chunk_size = 16`：模型一次预测多少步动作。这是 ACT 论文的核心参数。ALOHA 原论文使用 100（50Hz 控制），但这个值应该根据具体任务的频率和延迟需求调整。

- `n_action_steps = 16`：每次推理后实际执行的步数。必须 ≤ action_chunk_size。例如：
  - `n_action_steps = 1`（AKA-Sim2Real 默认）：每次只执行 chunk 的第一步，下一帧重新预测。这是最保守的策略，也是 Temporal Ensembling 模式下的唯一合法值。
  - `n_action_steps > 1`：每次执行 chunk 的前 N 步，然后等待 N 步后再推理。这适用于推理延迟较高的场景。

### 1.1.7 `__init__` 参数：相机与 CVAE

```python
    # 相机数量
    num_cameras: int = 1,
    # CVAE 参数
    latent_dim: int = 32,  # 隐变量 z 的维度
    use_cvae: bool = True,  # 是否使用 CVAE
    kl_weight: float = 0.1,  # KL 散度损失权重
```

**逐行解释**：

- `num_cameras = 1`：相机数量（AKA-Sim2Real 只有前视视角）。多相机场景下，每个相机的图像独立通过 ResNet 编码后拼接。支持多相机是 LeRobot 的设计，因为 ALOHA 有 4 个相机。

- `latent_dim = 32`：潜变量 $z$ 的维度。这个维度的选择是一个权衡：
  - 太小（如 8）：无法捕获足够的多模态信息
  - 太大（如 128）：KL 散度难以优化，且推理时采样的随机性更大
  - 32 是 CVAE 文献中常用的中间值

- `use_cvae = True`：是否启用 CVAE。设为 False 则退化为纯确定性模型（类似原始 Transformer 行为克隆）。

- `kl_weight = 0.1`：KL 散度在总损失中的权重。这是一个关键的超参数：
  - 值太小：潜变量退化为无信息先验，CVAE 失去多模态能力
  - 值太大：模型过于随机，重建精度下降
  - 0.1 是 LeRobot 调优后的值

### 1.1.8 `__init__` 参数：Temporal Ensembling

```python
    # Temporal Ensembling 参数
    use_temporal_ensembling: bool = True,  # 是否使用时间集成
    temporal_ensembling_coeff: float = 0.01,  # 时间集成衰减系数
```

**逐行解释**：

- `use_temporal_ensembling = True`：是否启用时间集成。

- `temporal_ensembling_coeff = 0.01`：指数衰减系数 $m$。计算公式为 $w_i = e^{-m \cdot i}$。这是 LeRobot ACT 原版的值（**建议不要改**）：
  - 当 $m = 0.01$，$w_0 = 1.0$，$w_7 \approx 0.93$——接近均匀加权，略微偏向旧动作
  - 当 $m = 0$：所有动作完全均匀加权
  - 当 $m < 0$（如 -0.01）：更重视**新**动作——这会让输出更快响应变化

  为什么 LeRobot 选择 "略微偏向旧动作"（$m > 0$）？因为机器人动作通常需要平滑性，稍加偏向历史预测可以减少高频抖动。

### 1.1.9 `__init__` 参数：Spatial Softmax

```python
    # Spatial Softmax 参数
    use_spatial_softmax: bool = True,  # 是否使用 Spatial Softmax
    spatial_softmax_temperature: float = 1.0,  # Spatial Softmax 温度参数
```

**逐行解释**：

- `use_spatial_softmax = True`：是否对视觉特征使用 Spatial Softmax（而非简单的全局平均池化）。Spatial Softmax 从特征图的每个空间位置计算 2D 关键点坐标，保留空间信息。

- `spatial_softmax_temperature = 1.0`：Softmax 的温度参数。温度越高 → 关键点分布越"软"（扩散），温度越低 → 关键点分布越"硬"（集中）。1.0 表示标准 Softmax。

  注意：在当前的 `modeling_act.py` 实现中，Spatial Softmax 相关代码被注释掉了（`_compute_spatial_softmax` 方法未在 `forward` 中调用），改用直接 Flatten + 2D 位置编码的方案。这是因为 ResNet18 输出的特征图本身已经很丰富，Spatial Softmax 带来的提升有限，但增加了计算量。

### 1.1.10 `__init__` 方法体

```python
    self.image_size = image_size
    self.in_channels = in_channels
    self.state_dim = state_dim
    # ... (所有参数都赋值给 self)
    self._validate()
```

**逐行解释**：

- 每个 `self.xxx = xxx`：将构造函数参数保存为实例属性。这是标准的配置模式——参数通过 `__init__` 传入，通过 `self.xxx` 访问。

- `self._validate()`：构造配置后立即执行参数校验。这体现了 **Fail Fast** 原则——配置错误应该在模型构造时就抛出异常，而不是在训练到一半时才崩溃。

### 1.1.11 `_validate()` — 参数校验

```python
def _validate(self):
    """参数校验"""
    if self.use_temporal_ensembling and self.n_action_steps > 1:
        raise ValueError(
            f"Temporal Ensembling 模式下 n_action_steps 必须为 1，"
            f"当前 n_action_steps={self.n_action_steps}"
        )
```

**逐行解释**：

- **第一个校验**：Temporal Ensembling 需要 `n_action_steps = 1`。为什么？
  - Temporal Ensembling 的工作原理是：每次推理产生一个 chunk，执行第一步，剩下的 7 步用于和下一次推理的预测做加权融合。
  - 如果你一次性执行多步（`n_action_steps > 1`），这些步之间就没有"重叠"可以融合——Temporal Ensembling 失去了意义。
  - 这个校验防止用户同时启用两个冲突的功能。

```python
    if self.n_action_steps > self.action_chunk_size:
        raise ValueError(
            f"n_action_steps ({self.n_action_steps}) 不能大于 "
            f"action_chunk_size ({self.action_chunk_size})"
        )
```

**解释**：

- **第二个校验**：逻辑约束——你不可能执行比预测更多的步数。`n_action_steps` 是实际执行的步数，它必须是 chunk 的一个子集。

### 1.1.12 `temporal_ensemble_coeff` — 兼容属性

```python
@property
def temporal_ensemble_coeff(self) -> float:
    """兼容性别名"""
    return self._temporal_ensembling_coeff if hasattr(self, '_temporal_ensembling_coeff') \
        else self.__dict__.get('temporal_ensembling_coeff', 0.01)
```

**逐行解释**：

- `@property`：这是一个属性（property），访问时不需要括号——`config.temporal_ensemble_coeff` 而非 `config.temporal_ensemble_coeff()`。

- 这是一个**向后兼容**的设计：
  1. `hasattr(self, '_temporal_ensembling_coeff')`：检查是否存在旧命名风格（下划线前缀）的属性。
  2. `self.__dict__.get('temporal_ensembling_coeff', 0.01)`：回退到检查新命名风格。
  3. 如果都不存在，返回默认值 0.01。

  为什么需要这个？因为不同版本的代码可能用不同的命名约定。这个属性确保无论模型文件是用旧名字还是新名字保存的，都能正确读取系数。

---

## 1.2 defaults.py — 默认配置与工厂函数

`defaults.py` 定义了 AKA-Sim2Real 的实际默认配置（覆盖 `ACTConfig.__init__` 的通用默认值），并提供配置构建工具。

### 1.2.1 导入

```python
from __future__ import annotations
from typing import Any
from .configuration_act import ACTConfig
```

**逐行解释**：

- `from __future__ import annotations`：启用 PEP 563 的延迟注解求值。这使得类型注解中的类名可以在定义之前引用——在 `TYPE_CHECKING` 场景下特别有用（虽然本文件没用到，但这是项目的统一风格）。

- `from typing import Any`：`Any` 类型表示"任意类型"。用于 `dict[str, Any]`，因为配置字典的值可能是 `int`、`float`、`bool`、`Tuple` 等。

- `from .configuration_act import ACTConfig`：相对导入。`.` 表示当前包目录（`policies/models/act/`）。

### 1.2.2 DEFAULT_ACT_CONFIG

```python
DEFAULT_ACT_CONFIG: dict[str, Any] = {
    "state_dim": 2,            # [vel_left, vel_right]
    "action_dim": 3,           # [left_vel, right_vel, gripper_target]
    "action_chunk_size": 8,    # 预测8步未来动作
    "n_action_steps": 1,       # Temporal Ensembling 模式下必须为 1
    "hidden_dim": 512,
    "num_attention_heads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 3200,
    "use_cvae": True,
    "kl_weight": 0.1,
    "use_temporal_ensembling": True,
    "temporal_ensembling_coeff": 0.01,
    "use_spatial_softmax": True,
    "latent_dim": 32,
    "num_cameras": 1,
}
```

**逐行解释**（聚焦与 `ACTConfig` 默认值的差异）：

- `state_dim = 2`：AKA-Sim2Real 的车辆状态是 2 维——`[左轮速度, 右轮速度]`。这与 ALOHA 的 7 维关节状态不同。

- `action_dim = 3`：动作空间是 3 维——`[左轮速度指令, 右轮速度指令, 夹爪目标]`。注意在 AKA-Sim2Real 中实际使用的训练参数是 `action_dim = 2`（不含夹爪），这个 3 是预留的。

- `action_chunk_size = 8`：预测 8 步动作。如果控制频率是 10Hz，这对应 0.8 秒的预测视野。

- `n_action_steps = 1`：每次只执行第一步（与 Temporal Ensembling 兼容）。

- `num_encoder_layers = 4`、`num_decoder_layers = 4`：相比 `ACTConfig.__init__` 的默认值 6 层，AKA-Sim2Real 使用更浅的网络（4 层）。原因是：车辆控制任务比 ALOHA 双臂操作简单，更少的层数可以加快推理速度且不易过拟合。

**设计模式说明**：这里使用一个大字典而非在 `ACTConfig.__init__` 中直接设置默认值，遵循了 **关注点分离** 原则：
- `ACTConfig.__init__`：定义通用默认值（对齐 LeRobot/ALOHA 的场景）
- `defaults.py`：定义本项目特有的默认值（面向车辆控制）

### 1.2.3 build_act_config() — 配置构建器

```python
def build_act_config(**overrides: Any) -> ACTConfig:
    config_dict = DEFAULT_ACT_CONFIG.copy()
    config_dict.update(overrides)
    return ACTConfig(**config_dict)
```

**逐行解释**：

- `**overrides: Any`：关键字可变参数。允许调用者覆盖默认配置中的任意项。例如 `build_act_config(state_dim=3, hidden_dim=256)`。

- `DEFAULT_ACT_CONFIG.copy()`：**必须 copy**！如果你直接修改 `DEFAULT_ACT_CONFIG`，由于它是模块级变量（全局单例），后续的 `build_act_config()` 调用会受影响。`.copy()` 创建浅拷贝，对于这个全是基本类型的字典来说足够安全。

- `config_dict.update(overrides)`：将调用者提供的覆盖值合并到默认配置中。如果 key 已存在则覆盖，否则添加。

- `ACTConfig(**config_dict)`：使用字典展开的方式构造 `ACTConfig` 实例。Python 的 `**` 操作符将字典的 key-value 对展开为关键字参数。

### 1.2.4 act_config_to_dict() — 配置序列化

```python
def act_config_to_dict(config: ACTConfig) -> dict[str, Any]:
    return {
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        # ... 所有字段
    }
```

**逐行解释**：

- 这个函数是 `build_act_config` 的反向操作：`ACTConfig → dict`。

- 为什么需要手动列出所有字段，而不是用 `config.__dict__`？因为：
  1. **显式优于隐式**：明确控制哪些字段需要序列化（避免把 `_inference_latent_mu` 等内部状态也序列化进去）。
  2. **稳定性**：如果 `ACTConfig` 新增了内部属性，不会意外泄漏到保存的配置中。
  3. **排序保证**：手动列出保证了 JSON 输出中字段的顺序（dict 在 Python 3.7+ 中保持插入顺序）。

---

## 1.3 关键设计决策

### 为什么 encoder/decoder 层数（4 层）比原论文（6 层）少？

1. **任务复杂度**：车辆控制（2 维动作）远低于 ALOHA 双臂操作（14 维动作）
2. **数据量**：AKA-Sim2Real 的数据集通常比 ALOHA 小，更浅的网络不容易过拟合
3. **推理延迟**：在浏览器端的模拟器中，需要实时推理（~30Hz），更少的层数意味着更低的延迟

### 为什么 `dim_feedforward = 3200` 而非 2048？

这是 LeRobot 调优后的值。FFN 维度越大，模型容量越大。3200 相比 2048 大约增加了 56% 的参数量。LeRobot 团队发现这个值在高精度操作任务中表现更好。

---

## 课后思考

1. 如果控制频率从 10Hz 提升到 30Hz，`action_chunk_size` 应该如何调整？为什么？
2. 什么情况下应该将 `use_cvae` 设为 False？
3. Temporal Ensembling 的 `n_action_steps` 校验为什么是必要的？如果强行设为 2 会发生什么？
