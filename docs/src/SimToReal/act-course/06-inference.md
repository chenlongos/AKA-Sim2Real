# 第 6 讲：推理运行时系统

> 对应源文件：
> - `backend/services/inference/runtime.py` — 推理运行时
> - `backend/services/inference/checkpoint.py` — 检查点加载
> - `backend/services/inference/preprocess.py` — 预处理
> - `backend/services/inference/execution.py` — 时序集成策略

## 学习目标

- 掌握模型从磁盘加载到推理输出的完整数据流
- 理解 Singleton 运行时模式的设计意图
- 掌握观测归一化→模型推理→动作反归一化的完整 pipeline

---

## 6.1 checkpoint.py — 检查点加载

### 6.1.1 ACTNormalizationStats — 归一化统计量数据类

```python
@dataclass
class ACTNormalizationStats:
    state_q01: torch.Tensor = field(default_factory=lambda: torch.zeros(2))
    state_q99: torch.Tensor = field(default_factory=lambda: torch.ones(2))
    action_q01: torch.Tensor = field(default_factory=lambda: torch.zeros(2))
    action_q99: torch.Tensor = field(default_factory=lambda: torch.ones(2))
```

**逐行解释**：

- `@dataclass`：Python 的 `dataclasses.dataclass` 装饰器。自动生成 `__init__`, `__repr__`, `__eq__`。比普通类少写大量样板代码。

- `field(default_factory=lambda: torch.zeros(2))`：默认值用 `default_factory` 而非直接 `torch.zeros(2)`。这是因为：
  - Python 的 dataclass 默认值在类定义时只求值一次
  - 如果直接用 `torch.zeros(2)`，所有实例会共享同一个 tensor 对象
  - `default_factory` 在每次实例化时调用 lambda，每个实例拥有独立的 tensor

- 默认值 `zeros(2)` 和 `ones(2)`：当没有加载 stats.json 时，`q01=0, q99=1` 使得归一化 `2*(x-0)/(1-0)-1 = 2x-1`——相当于简单的恒等映射到 [-1, 1]。

### 6.1.2 ACTCheckpointBundle — 检查点数据包

```python
@dataclass
class ACTCheckpointBundle:
    model_config: "ACTConfig"
    state_dict: dict
    inference_latent_mu: Optional[torch.Tensor] = None
    inference_latent_log_sigma: Optional[torch.Tensor] = None
```

**逐行解释**：

- `model_config: "ACTConfig"`：字符串形式的类型注解（前向引用）。因为文件顶部使用了 `from __future__ import annotations`，所有注解都是字符串，在 TYPE_CHECKING 时才求值。这避免了循环导入。

- 这个 dataclass 的作用是将"加载检查点"这个操作的结果打包为一个不可变的值对象（Value Object），后续的 `instantiate_model` 从这个包中读取所有需要的信息。

### 6.1.3 `get_default_device()` — 设备选择

```python
def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

**逐行解释**：

- 设备选择优先级：CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU。
- `torch.cuda.is_available()`：检查是否有 NVIDIA GPU（需要安装 CUDA 版本的 PyTorch）。
- `torch.backends.mps.is_available()`：检查是否有 Apple M 系列芯片的 GPU（macOS 12.3+ 且 PyTorch 1.12+）。
- 返回字符串而非 `torch.device` 对象——字符串在后续代码中可以直接传给 `.to()` 方法。

### 6.1.4 `resolve_default_model_path()` — 默认路径解析

```python
def resolve_default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "output" / "train" / "final_model.pt"
    if not model_path.exists():
        model_path = project_root / "output" / "train" / "model.pt"
    return model_path
```

**逐行解释**：

- `Path(__file__).resolve().parents[3]`：从当前文件向上 4 级目录：
  - `execution.py` → `inference/` → `services/` → `backend/` → 项目根目录
  - `.resolve()` 解析符号链接，`.parents[3]` 取第 4 层父目录。

- 回退策略：先尝试 `final_model.pt`（训练完保存的最终模型），不存在则尝试 `model.pt`（兼容旧命名）。

### 6.1.5 `load_checkpoint_bundle()` — 加载检查点

```python
def load_checkpoint_bundle(model_path: Optional[str], device: str) -> ACTCheckpointBundle:
    project_root = Path(__file__).resolve().parents[3]
    if model_path is not None:
        path = Path(model_path) if Path(model_path).is_absolute() else project_root / model_path
    else:
        path = resolve_default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
```

**逐行解释**：

- `Path(model_path).is_absolute()`：判断用户提供的路径是否为绝对路径。
  - 绝对路径：直接使用
  - 相对路径：相对于项目根目录拼接

- `raise FileNotFoundError(...)`：Fail Fast——文件不存在立即报错，比后续 `KeyError` 或 `RuntimeError` 更容易定位问题。

```python
    checkpoint = torch.load(path, map_location=device, weights_only=False)
```

**逐行解释**：

- `torch.load(path, map_location=device)`：加载 PyTorch 序列化文件。`map_location` 指定将 tensor 加载到哪个设备。这避免了"GPU 模型加载到 CPU 机器"的问题。

- `weights_only=False`：允许加载包含非张量数据（如配置字典、Python 对象）。`weights_only=True`（PyTorch 2.0+）更安全但限制更多。这里因为检查点包含 dict、str 等非张量数据，必须设为 False。

```python
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_config = checkpoint.get("config", {})
        inference_latent_mu = checkpoint.get("inference_latent_mu")
        inference_latent_log_sigma = checkpoint.get("inference_latent_log_sigma")
    else:
        state_dict = checkpoint
        checkpoint_config = {}
        inference_latent_mu = None
        inference_latent_log_sigma = None
```

**逐行解释**：

- 兼容两种检查点格式：
  - **新格式**（`train_act.py` 输出）：`{"model_state_dict": ..., "config": ..., "inference_latent_mu": ..., ...}`
  - **旧格式**（仅权重）：直接是 `state_dict`

- `checkpoint.get("config", {})`：如果旧格式没有 config，返回空字典——`build_act_config(**{})` 使用全部默认值。

```python
    return ACTCheckpointBundle(
        model_config=build_act_config(**checkpoint_config),
        state_dict=state_dict,
        inference_latent_mu=inference_latent_mu,
        inference_latent_log_sigma=inference_latent_log_sigma,
    )
```

### 6.1.6 `load_stats()` — 加载归一化统计

```python
def load_stats(stats_dir: Optional[str]) -> ACTNormalizationStats:
    project_root = Path(__file__).resolve().parents[3]
    if stats_dir is not None:
        data_dir = Path(stats_dir) if Path(stats_dir).is_absolute() else project_root / stats_dir
    else:
        data_dir = project_root / "output" / "dataset"
    stats_path = data_dir / "meta" / "stats.json"
    if not stats_path.exists():
        return ACTNormalizationStats()
```

**解释**：如果 stats.json 不存在，返回默认统计量（q01=0, q99=1），相当于不做 QUANTILES 归一化。

```python
    with open(stats_path, "r") as f:
        stats = json.load(f)

    state_entry = stats.get("observation.state", {})
    action_entry = stats.get("action", {})

    state_q01 = torch.tensor(state_entry.get("q01", [0.0, 0.0]), dtype=torch.float32)
    state_q99 = torch.tensor(state_entry.get("q99", [1.0, 1.0]), dtype=torch.float32)
    action_q01 = torch.tensor(action_entry.get("q01", [0.0, 0.0]), dtype=torch.float32)
    action_q99 = torch.tensor(action_entry.get("q99", [1.0, 1.0]), dtype=torch.float32)

    return ACTNormalizationStats(
        state_q01=state_q01, state_q99=state_q99,
        action_q01=action_q01, action_q99=action_q99,
    )
```

**逐行解释**：

- `stats.get("observation.state", {})`：使用 `.get()` 提供默认空字典。如果 stats.json 的结构不完整，不会抛出 KeyError。
- `state_entry.get("q01", [0.0, 0.0])`：每级都提供合理默认值。
- 这种**多层防御性提取**确保了系统在任何不完整的 stats.json 下都能正常运行。

### 6.1.7 `instantiate_model()` — 从检查点实例化模型

```python
def instantiate_model(bundle: ACTCheckpointBundle, device: str) -> "ACTModel":
    from policies.models.act.modeling_act import ACTModel as PyACTModel

    model = PyACTModel(bundle.model_config)
    model.load_state_dict(bundle.state_dict)
    if bundle.inference_latent_mu is not None and bundle.inference_latent_log_sigma is not None:
        model.set_inference_latent(bundle.inference_latent_mu, bundle.inference_latent_log_sigma)
    model = model.to(device)
    model.eval()
    return model
```

**逐行解释**：

- `from policies.models.act.modeling_act import ACTModel as PyACTModel`：延迟导入 + 重命名。延迟导入避免了循环依赖（`checkpoint.py` 和 `modeling_act.py` 之间）。重命名为 `PyACTModel` 是为了与可能的其他 `ACTModel` 引用区分。

- `model.load_state_dict(bundle.state_dict)`：将权重加载到模型中。PyTorch 的 `load_state_dict` 使用**严格的键名匹配**（`strict=True` 默认）——如果保存的权重中有一个键在模型中不存在，会抛出错误。这是一种安全特性：避免"静默加载了不完整的权重"。

- `model.set_inference_latent(mu, log_sigma)`：设置 CVAE 推理分布。这是在加载后的关键步骤——如果没有这一步，即使训练了 CVAE，推理时也用噪声采样。

- `model.eval()`：设为评估模式（关闭 Dropout 等）。

---

## 6.2 preprocess.py — 预处理

### 6.2.1 ACTPreprocessor 类的初始化

```python
class ACTPreprocessor:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
```

**逐行解释**：

- `transforms.Compose([...])`：torchvision 的图像预处理管道。`Compose` 按顺序应用列表中的每个变换。

- `transforms.Resize((224, 224))`：缩放图像到 ResNet18 的标准输入尺寸。默认使用双线性插值（`InterpolationMode.BILINEAR`）。

- `transforms.ToTensor()`：将 PIL Image (uint8, [0, 255]) 转换为 float tensor ([0.0, 1.0])，同时自动将通道顺序从 `[H, W, C]` 转为 `[C, H, W]`。

- `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`：ImageNet 均值/标准差归一化。注意这与 `ACTDataset` 中的手动归一化逻辑等价，但使用 torchvision 的标准化实现更简洁。

### 6.2.2 `process_image()` — 图像处理

```python
def process_image(self, image_input: Union[str, Image.Image, None], device: str) -> torch.Tensor:
    if image_input is None:
        return torch.randn(1, 1, 3, 224, 224).to(device)
```

**逐行解释**：

- 输入为 `None`（无图像）时：创建随机噪声张量 `[1, 1, 3, 224, 224]`。形状解释：
  - dim 0 = batch = 1
  - dim 1 = num_cameras = 1
  - dim 2-4 = 3×224×224 (RGB 图像)
  使用随机噪声是故障恢复策略——至少能产生一些输出（可能是随机的），比整个系统崩溃好。

```python
    try:
        if isinstance(image_input, str):
            if "," in image_input:
                image_input = image_input.split(",")[1]
            image_data = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            return torch.randn(1, 1, 3, 224, 224).to(device)
        return self.image_transform(image).unsqueeze(0).unsqueeze(0).to(device)
    except Exception:
        return torch.randn(1, 1, 3, 224, 224).to(device)
```

**逐行解释**：

- `if "," in image_input: image_input = image_input.split(",")[1]`：处理 Data URL 格式 `data:image/png;base64,iVBORw0...`。`,` 之后的部分才是 base64 编码图像数据。

- `base64.b64decode(image_input)`：Base64 解码。将 ASCII 字符串转换回二进制数据。

- `io.BytesIO(image_data)`：将 bytes 包装为内存中的文件对象，`Image.open()` 可以从这个虚拟文件读取。

- `image.convert("RGB")`：统一转换为 RGB 格式（处理 RGBA、灰度等）。

- `.unsqueeze(0).unsqueeze(0)`：两次 unsqueeze。第一次添加 batch 维度，第二次添加 camera 维度。最终形状：`[1, 1, 3, 224, 224]`。

- `try/except Exception`：捕获所有可能的图像处理错误（损坏的数据、不支持的格式等），回退到随机噪声。这在生产环境中很重要——一个坏图像不应该拖垮整个推理 pipeline。

### 6.2.3 `normalize_state()` — 状态归一化

```python
def normalize_state(self, state: list, stats: ACTNormalizationStats, device: str) -> torch.Tensor:
    """使用 QUANTILES 归一化状态：2 * (x - q01) / (q99 - q01) - 1"""
    state_tensor = torch.tensor(state[:2], dtype=torch.float32).unsqueeze(0).to(device)
    q01 = stats.state_q01.unsqueeze(0).to(device)
    q99 = stats.state_q99.unsqueeze(0).to(device)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)
    return 2 * (state_tensor - q01) / denom - 1
```

**逐行解释**：

- `state[:2]`：**只取前 2 个维度**。这与 AKA-Sim2Real 的实际状态空间 `[vel_left, vel_right]` 对应。如果传入的状态列表更长（如包含夹爪状态），多余的维度被截断。

- `.unsqueeze(0)`：添加 batch 维度 `[2] → [1, 2]`。

- 归一化公式与 `ACTDataset` 中的完全一致（见第 2 讲）。

### 6.2.4 `denormalize_action()` — 动作反归一化

```python
def denormalize_action(self, action: torch.Tensor, stats: ACTNormalizationStats, device: str) -> torch.Tensor:
    """使用 QUANTILES 反归一化动作：(x + 1) / 2 * (q99 - q01) + q01"""
    q01 = stats.action_q01.unsqueeze(0).to(device)
    q99 = stats.action_q99.unsqueeze(0).to(device)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)
    return (action + 1) / 2 * denom + q01
```

**逐行解释**：

- 反归一化公式是归一化公式的**逆运算**：
  归一化：$x' = 2 \cdot \frac{x - q_{01}}{q_{99} - q_{01}} - 1$
  反归一化：$x = \frac{x' + 1}{2} \cdot (q_{99} - q_{01}) + q_{01}$

- 推导验证：
  - 当 $x' = -1$：$x = 0/2 * \Delta + q_{01} = q_{01}$ ✓（最小值映射回 1% 分位）
  - 当 $x' = 1$：$x = 2/2 * \Delta + q_{01} = q_{99}$ ✓（最大值映射回 99% 分位）

---

## 6.3 execution.py — 时序集成策略

```python
@dataclass
class TemporalEnsemblingPolicy:
    """Online action-chunk fusion policy for ACT inference."""
    decay: float = 0.5
    step: int = 0
    predictions: Deque[dict] = field(default_factory=deque)
```

**逐行解释**：

- `decay = 0.5`：一种简化版 Temporal Ensembling 的衰减系数（与 `ACTTemporalEnsembler` 不同，这里用幂律衰减 $0.5^{rel\_idx}$）。

- `step = 0`：当前推理步数（自 episode 开始以来的计数）。

- `predictions: Deque[dict]`：双端队列，存储历史预测。每个预测是一个 dict：`{"start_step": int, "actions": Tensor[k, d_a]}`。

  注意：这个类`TemporalEnsemblingPolicy`与`modeling_act.py`中的`ACTTemporalEnsembler`是**两个独立的实现**。`execution.py`中的版本更简单（幂律衰减权重），适用于快速原型；`modeling_act.py`中的版本更精确（指数加权），对齐了 LeRobot。当前 runtime 使用的是 `ACTTemporalEnsembler`，这个类作为备选方案保留。

```python
def reset(self):
    self.step = 0
    self.predictions.clear()
```

**解释**：在 episode 开始时调用，清除所有历史预测。不清除的话，上一个 episode 的动作会污染新 episode 的起始动作。

```python
def blend(self, action_chunk: torch.Tensor) -> torch.Tensor:
    if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
        return action_chunk
```

**逐行解释**：

- 防御性检查：只处理 `[1, k, d_a]` 格式的 chunk。如果格式不对，直接返回原值（不做融合）。

```python
    current_step = self.step
    chunk_size = action_chunk.shape[1]
    prediction = action_chunk[0].detach().clone().cpu()
    self.predictions.append({
        "start_step": current_step,
        "actions": prediction,
    })
```

**逐行解释**：

- `.detach().clone().cpu()`：断开计算图 + 深拷贝 + 移回 CPU。存储在 `predictions` 中的张量需要在 GPU 推理循环之外持久化，移到 CPU 避免 GPU 内存泄漏。

```python
    min_valid_start = current_step - chunk_size + 1
    while self.predictions and self.predictions[0]["start_step"] < min_valid_start:
        self.predictions.popleft()
```

**逐行解释**：

- `min_valid_start = current_step - chunk_size + 1`：有效窗口的边界。只有 `start_step >= current_step - k + 1` 的预测才可能覆盖当前步。例如 k=8，当前步=20：只有 start_step ≥ 13 的预测能覆盖步 20。

- `while ... popleft()`：从左侧移除过期的预测。双端队列使得 O(1) 的左侧弹出效率。

```python
    candidates = []
    weights = []
    for pred in self.predictions:
        rel_idx = current_step - pred["start_step"]
        if 0 <= rel_idx < pred["actions"].shape[0]:
            weight = self.decay ** rel_idx
            candidates.append(pred["actions"][rel_idx])
            weights.append(weight)
```

**逐行解释**：

- `rel_idx = current_step - pred["start_step"]`：相对索引——"这个预测产生于多少步之前"。rel_idx=0 表示"刚刚产生的预测"。

- `0 <= rel_idx < chunk_size`：确保当前步骤在预测的有效范围内。

- `weight = self.decay ** rel_idx`：幂律衰减权重。例如 decay=0.5：
  - rel_idx=0（最新）：$w = 0.5^0 = 1.0$
  - rel_idx=1：$w = 0.5^1 = 0.5$
  - rel_idx=7（最旧）：$w = 0.5^7 \approx 0.008$

  这比标准 Temporal Ensembling 更激进地偏向最新预测（衰减更快）。

```python
    if candidates:
        weight_tensor = torch.tensor(weights, dtype=prediction.dtype).unsqueeze(1)
        candidate_tensor = torch.stack(candidates, dim=0)
        blended = (candidate_tensor * weight_tensor).sum(dim=0) / weight_tensor.sum()
        action_chunk = action_chunk.clone()
        action_chunk[0, 0] = blended.to(action_chunk.device)
```

**逐行解释**：

- 加权平均公式：
  $$a_{blended} = \frac{\sum_{j} w_j \cdot a_j}{\sum_{j} w_j}$$

- `action_chunk[0, 0] = blended.to(action_chunk.device)`：只替换 chunk 的第一个动作（当前步），其他位置保持不变。

```python
    self.step += 1
    return action_chunk
```

**解释**：步数递增，返回融合后的动作 chunk。调用方（`get_action`）会取 `[:, 0]` 获取当前动作。

---

## 6.4 runtime.py — 推理运行时

### 6.4.1 ACTInferenceRuntime 类

```python
class ACTInferenceRuntime:
    def __init__(self):
        self.model: Optional["ACTModel"] = None
        self.device = get_default_device()
        self.stats = ACTNormalizationStats()
        self.preprocessor = ACTPreprocessor()
        self.temporal_ensembler: Optional[ACTTemporalEnsembler] = None
        self.action_chunk_size = 8
```

**逐行解释**：

- 运行时是一个**有状态对象**，持有模型、设备、预处理器的引用。
- `model: Optional["ACTModel"] = None`：初始时模型未加载。类型注解 `Optional` 表示可以为 None。
- `self.stats = ACTNormalizationStats()`：默认统计量（q01=0, q99=1），在 `_load_stats` 中被覆盖。
- `self.action_chunk_size = 8`：默认值，在 `load_model` 时从配置中更新。

### 6.4.2 `load_model()` — 模型加载与组装

```python
def load_model(self, model_path=None, stats_dir=None):
    logger.info("加载 ACT 模型...")
    self.device = get_default_device()
    bundle = load_checkpoint_bundle(model_path, self.device)
    self.model = instantiate_model(bundle, self.device)
    self.reset_inference_context()
    self._load_stats(stats_dir)
```

**逐行解释**：

- 顺序：设备选择 → 加载检查点 → 实例化模型 → 重置推理上下文 → 加载归一化统计。每个步骤依赖上一步的结果。

```python
    if self.should_use_temporal_ensembling():
        self.action_chunk_size = getattr(self.model.config, "action_chunk_size", 8)
        coeff = float(getattr(self.model.config, "temporal_ensembling_coeff", 0.01))
        self.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=coeff, chunk_size=self.action_chunk_size,
        )
```

**逐行解释**：

- `getattr(self.model.config, "action_chunk_size", 8)`：安全获取属性，不存在时默认为 8。
- `float(...)`：显式类型转换——确保系数是 `float` 类型（配置中可能是 `int` 0）。
- 初始化 `ACTTemporalEnsembler`（而非 `TemporalEnsemblingPolicy`）——使用 LeRobot 对齐的实现。

### 6.4.3 `infer()` — 单步推理

```python
def infer(self, state: list, image=None, user_id=None) -> list:
    if self.model is None:
        logger.warning(f"ACT 模型未加载，返回随机动作")
        return [[0.0, 0.0]]
```

**逐行解释**：

- 模型未加载时返回 `[[0.0, 0.0]]`（零动作）。这比抛异常更好——UI 层可以继续运行并提示用户加载模型。

```python
    with torch.no_grad():
        state_tensor = self.preprocessor.normalize_state(state, self.stats, self.device)
        image_tensor = self.process_image(image)

        use_temporal = self.should_use_temporal_ensembling()
        action = self.model.get_action(
            image_tensor, state_tensor,
            use_temporal_ensembling=use_temporal,
            temporal_ensembler=self.temporal_ensembler,
        )
```

**逐行解释**：

- `torch.no_grad()`：推理不需要梯度——节省内存和计算。
- `normalize_state(...)`：原始状态 → [-1, 1] 归一化状态。
- `process_image(...)`：原始图像 → 归一化 tensor。
- `model.get_action(...)`：参见第 4 讲 4.5.6 节。

```python
        action = self.preprocessor.denormalize_action(action, self.stats, self.device)
        return action[0].cpu().tolist()
```

**逐行解释**：

- `denormalize_action(action, ...)`：将 [-1, 1] 模型输出映射回实际动作范围。
- `.cpu()`：将 tensor 从 GPU 移回 CPU。
- `.tolist()`：PyTorch tensor → Python list。例如 `tensor([0.5, -0.3])` → `[0.5, -0.3]`。这是 API 响应所需的格式。

### 6.4.4 单例模式

```python
_runtime = ACTInferenceRuntime()

def get_act_runtime() -> ACTInferenceRuntime:
    return _runtime
```

**逐行解释**：

- `_runtime`：模块级变量（模块单例）。Python 的模块只在第一次 import 时初始化——所有后续 import 都共享同一个 `_runtime` 实例。
- 这种模式确保整个应用中只有一个模型加载在内存中——避免多次加载模型的巨大内存浪费（每个模型实例 ~100MB+）。
- 为什么不直接用全局变量而包装在函数中？函数提供了依赖注入（DI）的可能性——未来可以替换为不同的运行时实例而不影响调用方。

```python
def load_act_model(model_path=None, stats_dir=None):
    return _runtime.load_model(model_path=model_path, stats_dir=stats_dir)

def act_inference(state, image=None, user_id=None):
    return _runtime.infer(state, image, user_id)
```

**逐行解释**：

- 这些是**便利函数（convenience functions）**——让调用方可以直接 `import load_act_model` 而不需要先 `get_act_runtime()`。
- 这种模式在 FastAPI 架构中很常见——API 路由通过简单的函数调用就能触发推理，而不需要管理运行时对象的生命周期。

---

## 6.5 完整推理数据流

回顾一次完整推理的数据流动：

```
用户输入（HTTP/WebSocket）
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ACTPreprocessor                                     │
│                                                     │
│  state: [0.5, -0.3]  ──► normalize_state()  ──► [-0.2, 0.1] │
│  image: "base64..."   ──► process_image()   ──► [1,1,3,224,224] │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ACTModel.get_action()                               │
│                                                     │
│  forward() → { action: [B, k, d_a], ... }          │
│                                                     │
│  if temporal_ensemble:                              │
│    ACTTemporalEnsembler.update() → [B, d_a]        │
│  else:                                              │
│    action[:, 0] → [B, d_a]                         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ACTPreprocessor                                     │
│                                                     │
│  denormalize_action() → [0.3, -0.5]                │
└─────────────────────────────────────────────────────┘
    │
    ▼
返回给控制器/模拟器
```

---

## 课后思考

1. 为什么 `process_image()` 在异常时返回随机噪声而不是 `None` 或抛出异常？这种容错策略的利弊是什么？
2. 模块级单例 `_runtime` 在多线程环境（FastAPI 的 async workers）中是否安全？需要考虑什么并发问题？
3. `normalize_state()` 中 `state[:2]` 的硬编码有什么风险？如何改进？
