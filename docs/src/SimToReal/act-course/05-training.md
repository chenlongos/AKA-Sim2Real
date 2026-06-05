# 第 5 讲：训练流程

> 对应源文件：`policies/models/act/train_act.py`

## 学习目标

- 掌握 ACT 训练循环的完整流程
- 理解 CVAE latent 统计量收集的原理
- 掌握检查点保存策略

---

## 5.1 导入与路径设置

```python
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import io
import base64
```

**逐行解释**：

- `argparse`：命令行参数解析。`train_act.py` 作为独立脚本运行，通过命令行传递超参数。
- `Path`（pathlib）：跨平台的文件路径操作。比 `os.path` 更现代、更可读。
- `PIL.Image`：Pillow 库的图像加载。用于从文件中读取 PNG/JPEG 图像。
- `base64`：Base64 编解码。虽然训练脚本中不直接使用，但导入是为了兼容数据可能来源于 base64 编码的场景。

```python
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

**逐行解释**：

- `Path(__file__)`：当前文件的绝对路径。`__file__` 是 Python 的模块级变量，指向当前 `.py` 文件的路径。
- `.parent.parent.parent`：向上三级目录——`train_act.py → act/ → models/ → policies/`，即项目根目录。
- `sys.path.insert(0, str(project_root))`：将项目根目录插入 Python 导入路径的**最前面**（索引 0）。这确保 `from policies.models.act.modeling_act import ...` 能找到正确的模块，而不会与系统中已安装的同名包冲突。

```python
logger = logging.getLogger("backend.services.training.orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
```

**逐行解释**：

- `logging.getLogger("backend.services.training.orchestrator")`：使用与训练编排器相同的 logger 名称。这意味着训练脚本的日志会与后端服务的训练日志合并，统一管理。

- `if not logger.handlers`：防御性检查。如果 logger 已经被其他模块配置过了（有 handlers），就不重复配置，避免日志重复输出。

- `logging.basicConfig(level=logging.INFO)`：设置日志级别为 INFO。这会输出 epoch 进度、损失值等信息。

### 本地导入

```python
from policies.models.act.modeling_act import ACTModel, ACTConfig
from policies.models.act.ACTDataset import ACTDataset
from policies.models.act.defaults import build_act_config, act_config_to_dict
```

**逐行解释**：

- 这三个导入是项目内部的，放在 `sys.path` 修改之后，确保路径正确。
- `ACTModel` 和 `ACTConfig` 从 `modeling_act` 导入，而不是从 `configuration_act`——因为 `ACTModel` 是训练的入口类。
- `build_act_config` 用于构建配置，`act_config_to_dict` 用于序列化到检查点。

---

## 5.2 `load_dataset()` — 数据加载

### 5.2.1 函数签名

```python
def load_dataset(data_dir: str = "dataset") -> Dict[str, torch.Tensor]:
```

**解释**：返回一个字典，包含三个键：`"observation.image"`, `"observation.state"`, `"action"`。这是 LeRobot 数据集的标准格式。

### 5.2.2 加载统计信息

```python
    data_path = Path(data_dir)

    with open(data_path / "meta" / "stats.json", "r") as f:
        stats = json.load(f)
```

**逐行解释**：

- `data_path / "meta" / "stats.json"`：`/` 操作符是 `pathlib.Path` 的特有语法——路径拼接。等价于 `os.path.join(data_dir, "meta", "stats.json")`。
- `json.load(f)`：从文件中读取 JSON。`stats.json` 包含训练数据集的 QUANTILES 百分位数统计量（1%/99%分位）。

### 5.2.3 加载 Parquet 数据文件

```python
    import pandas as pd

    parquet_files = sorted(data_path.glob("data/chunk-*/file-*.parquet"))
    print(f"找到 {len(parquet_files)} 个数据文件")
```

**逐行解释**：

- `import pandas as pd`：延迟导入（Lazy Import）。只有在实际调用 `load_dataset()` 时才导入 pandas。这避免了在不使用该函数时加载这个重量级库。
- `data_path.glob("data/chunk-*/file-*.parquet")`：通配符匹配。LeRobot 数据集存储为 `data/chunk-000/file-000.parquet`, `data/chunk-000/file-001.parquet`, ...
- `sorted(...)`：按文件名排序，确保数据文件按正确的时间顺序加载。

### 5.2.4 逐文件加载

```python
    images = []
    states = []
    actions = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
```

**逐行解释**：

- `pd.read_parquet(parquet_file)`：读取 Apache Parquet 格式文件。Parquet 是列式存储格式，比 CSV/JSON 更高效。
- 使用列表 `[]` 逐个追加，最后用 `torch.stack` 合并——这是处理大量数据时的内存友好做法。

```python
        for img_path in df["observation.image"]:
            full_path = data_path / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                img = img.resize((224, 224))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)
            else:
                images.append(torch.zeros(3, 224, 224))
```

**逐行解释**：

- `Image.open(full_path).convert("RGB")`：Pillow 打开图像并确保是 RGB 格式（处理 RGBA/灰度等异常格式）。
- `img.resize((224, 224))`：缩放到 ResNet18 的标准输入尺寸。
- `np.array(img)`：Pillow Image → numpy 数组，形状 `[H, W, C]`。
- `torch.from_numpy(...)`：numpy → PyTorch tensor（零拷贝）。注意 `from_numpy` 创建的 tensor 与 numpy 数组共享内存！
- `.permute(2, 0, 1)`：通道顺序从 `[H, W, C]` 转为 PyTorch 标准 `[C, H, W]`。
- `.float() / 255.0`：uint8 [0, 255] → float32 [0.0, 1.0]。
- `else` 分支：图像文件缺失时的容错处理。使用零张量（全黑图像）。这种做法至少能继续训练而不崩溃，但训练质量会受影响。

```python
        for state_list in df["observation.state"]:
            state = torch.tensor(state_list, dtype=torch.float32)
            states.append(state)
```

**逐行解释**：

- `torch.tensor(state_list, dtype=torch.float32)`：从 Python list 创建 float32 张量。Parquet 中可能存储为 list 或 numpy array，`torch.tensor` 能处理多种输入。

```python
        for action_data in df["action"]:
            action = torch.tensor(action_data, dtype=torch.float32)
            actions.append(action)
```

### 5.2.5 合并与返回

```python
    images = torch.stack(images)  # [N, 3, 224, 224]
    states = torch.stack(states)  # [N, state_dim]
    actions = torch.stack(actions)  # [N, action_dim]

    print(f"加载完成: {len(images)} 个样本")
    print(f"  图像形状: {images.shape}")
    print(f"  状态形状: {states.shape}")
    print(f"  动作形状: {actions.shape}")
```

**逐行解释**：

- `torch.stack(images)`：将图像列表堆叠为一个大的张量。`stack` 创建新维度（dim=0），要求列表中所有张量形状一致。如果 List 中某个元素是形状不一致的张量，`stack` 会报错。

- 打印形状信息用于调试——让用户确认数据加载正确。

```python
    return {
        "observation.image": images,
        "observation.state": states,
        "action": actions,
    }
```

**解释**：返回 LeRobot 标准格式的字典。注意是扁平键名 `"observation.image"` 而非嵌套字典，这与 `ACTDataset.__getitem__` 中的解包逻辑对应。

---

## 5.3 `train()` — 训练主函数

### 5.3.1 函数签名

```python
def train(
    data_dir: str = "dataset",
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    state_dim: int = 2,
    action_dim: int = 2,
    action_chunk_size: int = 8,
    hidden_dim: int = 512,
) -> ACTModel:
```

**逐行解释**：

- `epochs = 50`：完整遍历数据集 50 次。
- `batch_size = 8`：每次更新使用 8 个样本。注意这个值较小——机器人数据集通常不如 NLP/视觉数据集那么大，小 batch 也是合理的。
- `lr = 1e-4`：学习率 $10^{-4}$。这是 Adam 优化器的常用学习率。
- `state_dim = 2, action_dim = 2`：AKA-Sim2Real 的实际维度（左右轮速度）。
- `action_chunk_size = 8`：预测 8 步动作。

### 5.3.2 加载数据与归一化统计

```python
    data = load_dataset(data_dir)

    stats_path = Path(data_dir) / "meta" / "stats.json"
    state_q01, state_q99, action_q01, action_q99 = None, None, None, None
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        state_q01 = torch.tensor(stats["observation.state"]["q01"], dtype=torch.float32)
        state_q99 = torch.tensor(stats["observation.state"]["q99"], dtype=torch.float32)
        action_q01 = torch.tensor(stats["action"]["q01"], dtype=torch.float32)
        action_q99 = torch.tensor(stats["action"]["q99"], dtype=torch.float32)
```

**逐行解释**：

- `stats_path.exists()`：防御性检查——stats.json 可能不存在（数据采集阶段可能没有生成统计量）。
- `stats["observation.state"]["q01"]`：JSON 中嵌套访问。stats.json 的结构：
  ```json
  {
    "observation.state": {"q01": [...], "q99": [...]},
    "action": {"q01": [...], "q99": [...]}
  }
  ```
- 如果 stats.json 不存在，所有归一化参数为 `None`，`ACTDataset` 会跳过归一化步骤。

### 5.3.3 构建配置与模型

```python
    config = build_act_config(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
    )

    model = ACTModel(config)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

**逐行解释**：

- `build_act_config(...)`：使用 `defaults.py` 的工厂函数，只覆盖与默认值不同的参数。
- `sum(p.numel() for p in model.parameters())`：计算模型总参数量。`numel()` 返回张量中元素的数量。对于典型配置（hidden_dim=512, 4 encoder + 4 decoder layers），参数量大约在 20-30M。

### 5.3.4 创建 DataLoader

```python
    dataset = ACTDataset(
        data, action_chunk_size=action_chunk_size, normalize_images=True,
        state_q01=state_q01, state_q99=state_q99,
        action_q01=action_q01, action_q99=action_q99,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
```

**逐行解释**：

- `shuffle=True`：每个 epoch 随机打乱数据顺序。这是训练中的标准做法——防止模型记忆数据的顺序。
- `num_workers=0`：使用主进程加载数据（不启用多进程）。这是因为：
  1. 机器人数据集通常较小，多进程加载的开销可能大于收益
  2. 多进程与某些文件系统/OS 有兼容性问题
  3. 调试更简单（单进程的错误信息更清晰）

### 5.3.5 创建优化器与损失函数

```python
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
```

**逐行解释**：

- `optim.Adam`：Adam 优化器。自适应学习率方法，结合了 Momentum 和 RMSProp：
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  $$\theta_t = \theta_{t-1} - \alpha \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

- `nn.MSELoss()`：均方误差损失。注意这个 criterion 实际上在训练循环中**没有被使用**——取而代之的是 `F.l1_loss`（见下文）。这是一个开发过程中的遗留（可能最初计划用 MSE，后来改为 L1）。

### 5.3.6 设备与模式设置

```python
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")
```

**逐行解释**：

- `model.train()`：设置训练模式。启用 Dropout 等正则化技术。
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")`：优先使用 GPU（CUDA），回退到 CPU。
- `model = model.to(device)`：将模型和所有参数/缓冲区移动到目标设备。注意必须重新赋值——`to()` 返回新模型（虽然原地操作时是同一个对象，但保险起见使用返回值）。

### 5.3.7 CVAE Latent 统计收集设置

```python
    all_mu = []
    all_log_sigma = []
    latent_collection_epochs = min(5, epochs // 2)
```

**逐行解释**：

- `all_mu, all_log_sigma`：收集最后几个 epoch 中产生的潜变量分布参数。存为 Python 列表，最后合并。

- `latent_collection_epochs = min(5, epochs // 2)`：用训练的最后几个 epoch（最多 5 个）来收集 latent 统计量。为什么用**最后**几个 epoch？
  - 训练初期模型不稳定，潜变量分布还在剧烈变化
  - 训练后期模型收敛，潜变量分布趋于稳定
  - 用最后几个 epoch 的统计量更能代表推理时的分布

### 5.3.8 训练循环

```python
    for epoch in range(epochs):
        total_loss = 0
        total_l1_loss = 0
        total_kl_loss = 0
        num_batches = 0
```

**逐行解释**：

- 每个 epoch 开始前重置累积统计量。这些变量用于计算每个 epoch 的平均损失（用于日志输出和监控训练进度）。

```python
        for batch in dataloader:
            images = batch["observation"]["image"].to(device)
            states = batch["observation"]["state"].to(device)
            actions = batch["action"].to(device)
```

**逐行解释**：

- `.to(device)`：将每个 batch 移动到 GPU/CPU。如果不做这一步，数据和模型不在同一设备上，PyTorch 会报 `Expected all tensors to be on the same device` 错误。

#### 前向传播

```python
            optimizer.zero_grad()

            output = model(
                images, states,
                action_target=actions,
                infer_cvae=False,
            )
```

**逐行解释**：

- `optimizer.zero_grad()`：**清零梯度**。PyTorch 的梯度默认是累加的（`.backward()` 把新梯度**加**到 `.grad` 属性上）。如果不清零，多次 backward 的梯度会累加，导致错误。

- `infer_cvae=False`：训练模式下，即使不在 `self.training` 条件下，也不走 CVAE 推理路径。`infer_cvae=False` 告诉 `forward()`："不要尝试从先验分布采样 z，我们已经在训练时用 action_target 编码了 z"。

```python
            predicted_actions = output["action"]
            kl_loss = output.get("kl_loss")
            mu = output.get("mu")
            log_sigma_x2 = output.get("log_sigma_x2")
```

**逐行解释**：

- `output.get("kl_loss")`：使用 `.get()` 而非 `[]` 索引。如果 `use_cvae=False`，`kl_loss` 为 `None`，`.get()` 返回 `None` 而非抛出 `KeyError`。

#### 收集 latent 统计

```python
            if config.use_cvae and mu is not None and log_sigma_x2 is not None:
                if epoch >= epochs - latent_collection_epochs:
                    all_mu.append(mu.detach().cpu())
                    all_log_sigma.append(log_sigma_x2.detach().cpu())
```

**逐行解释**：

- `epoch >= epochs - latent_collection_epochs`：只在最后几个 epoch 收集。例如 epochs=50, latent_collection_epochs=5 → 从 epoch 45 开始收集。

- `.detach()`：**断开计算图**。`.detach()` 阻止梯度从 latents 统计量回传到模型。不 detach 的后果：这些保存的 tensor 持有对计算图的引用，可能导致内存泄漏。

- `.cpu()`：将 tensor 移动到 CPU。因为这些统计量需要在 GPU 训练循环之外使用（保存到文件时），移到 CPU 确保兼容性。

#### 损失计算

```python
            l1_loss = F.l1_loss(predicted_actions, actions)

            if kl_loss is not None:
                loss = l1_loss + kl_loss * config.kl_weight
                total_kl_loss += kl_loss.item()
            else:
                loss = l1_loss

            total_l1_loss += l1_loss.item()
```

**逐行解释**：

- `F.l1_loss(predicted_actions, actions)`：L1 损失（平均绝对误差）：
  $$\text{L1} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

  为什么用 L1 而非 MSE（L2）？
  - L1 对异常值（outliers）不那么敏感——单个大误差不会主导损失
  - 在机器人控制中，偶尔的大误差比持续的小误差更危险，L1 更鲁棒
  - L1 在高维动作空间中产生更清晰的梯度信号

- `loss = l1_loss + kl_loss * config.kl_weight`：**总损失 = 重建损失 + 0.1 × KL 散度**。KL 权重 0.1 使得模型优先学习好的重建，同时温和地约束潜变量分布。

- `.item()`：将标量 tensor 转换为 Python float。这既节约 GPU 内存（不再持有这些 tensor），又方便日志输出。

#### 反向传播与优化

```python
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
```

**逐行解释**：

- `loss.backward()`：**反向传播**。计算损失对每个参数的梯度 $\frac{\partial L}{\partial \theta}$。梯度存储在 `param.grad` 中。

- `optimizer.step()`：**参数更新**。使用 Adam 算法根据梯度更新所有参数。

- `total_loss += loss.item()`：累积损失用于 epoch 统计。

### 5.3.9 Epoch 日志与检查点

```python
        avg_loss = total_loss / num_batches
        avg_l1 = total_l1_loss / num_batches
        if config.use_cvae:
            avg_kl = total_kl_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, KL: {avg_kl:.6f})")
```

**逐行解释**：

- `avg_loss = total_loss / num_batches`：计算 epoch 平均损失。
- 日志格式 `{avg_loss:.6f}`：保留 6 位小数。有助于观察训练的微小变化。

```python
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
```

**逐行解释**：

- `(epoch + 1) % 10 == 0`：每 10 个 epoch 保存一个检查点。使用 `epoch + 1` 因为 epoch 从 0 开始计数。
- `checkpoint_path.parent.mkdir(parents=True, exist_ok=True)`：确保输出目录存在。`parents=True` 递归创建父目录，`exist_ok=True` 如果目录已存在不报错。
- `torch.save(model.state_dict(), checkpoint_path)`：保存模型权重。只保存 `state_dict()` 而非整个模型对象——避免 pickle 依赖和版本不兼容。

### 5.3.10 保存最终模型

```python
    if config.use_cvae and len(all_mu) > 0:
        all_mu_tensor = torch.cat(all_mu, dim=0)
        all_log_sigma_tensor = torch.cat(all_log_sigma, dim=0)

        latent_mu_mean = all_mu_tensor.mean(dim=0)
        latent_log_sigma_mean = all_log_sigma_tensor.mean(dim=0)

        final_path = Path(output_dir) / "final_model.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'inference_latent_mu': latent_mu_mean,
            'inference_latent_log_sigma': latent_log_sigma_mean,
            'config': act_config_to_dict(config),
        }
        torch.save(checkpoint, final_path)
```

**逐行解释**：

- `torch.cat(all_mu, dim=0)`：将收集到的所有 $\mu$ 向量沿 batch 维度拼接。如果收集了 5 个 epoch × 10 batches/epoch × 8 samples/batch = 400 个样本：`[400, 32]`。

- `all_mu_tensor.mean(dim=0)`：沿 batch 维度取平均，得到 `[32]` 的均值向量。这就是推理时使用的潜变量分布均值。

- 检查点字典的结构：`model_state_dict`（权重）+ `inference_latent_mu`（推理均值）+ `inference_latent_log_sigma`（推理方差）+ `config`（配置）。加载时通过 `load_checkpoint_bundle` 解析。

```python
    else:
        final_path = Path(output_dir) / "final_model.pt"
        torch.save(model.state_dict(), final_path)
```

**解释**：无 CVAE 时只保存权重。

---

## 5.4 `export_to_huggingface()` — 导出

```python
def export_to_huggingface(model, config, output_dir="huggingface_model", model_name="aka-sim-act"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "pytorch_model.bin")

    # 保存 config.json
    config_dict = {
        "model_type": "act",
        "state_dim": config.state_dim,
        ...
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
```

**逐行解释**：

- HuggingFace 格式约定：`pytorch_model.bin`（权重）+ `config.json`（配置）是 HF 的标准文件命名。
- `json.dump(..., indent=2)`：格式化输出，便于人类阅读和调试。

### 生成的推理代码、README

```python
    inference_code = '''...'''   # 模板推理代码
    with open(output_path / "inference.py", "w") as f:
        f.write(inference_code)

    readme = f'''...'''          # 模板 README（使用 f-string 插入实际配置值）
    with open(output_path / "README.md", "w") as f:
        f.write(readme)
```

**解释**：导出的推理代码是一个模板/存根（stub），需要用户根据实际情况修改。它的作用是指明推理 API 的使用方式。`f'''...'''` 是 f-string 三重引号——在运行时将配置值插入模板文本。

---

## 5.5 `main()` — 命令行入口

```python
def main():
    parser = argparse.ArgumentParser(description="训练ACT模型")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--export_hf", action="store_true")

    args = parser.parse_args()

    model = train(
        data_dir=args.data_dir, output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    if args.export_hf:
        config = build_act_config(state_dim=7, action_dim=5, action_chunk_size=16, hidden_dim=512)
        export_to_huggingface(model, config)

if __name__ == "__main__":
    main()
```

**逐行解释**：

- `argparse.ArgumentParser`：Python 标准库的命令行解析器。
- `action="store_true"`：`--export_hf` 是一个布尔标志——出现则为 True，否则为 False。
- `if __name__ == "__main__"`：Python 的标准入口保护。当文件被直接运行时（`python train_act.py`）执行 `main()`；被 import 时不会执行。
- `export_to_huggingface` 中硬编码了 `state_dim=7, action_dim=5` 而非使用训练时的配置——这是一个已知的限制，导出 HF 格式时需要手动调整参数来匹配训练配置。

---

## 课后思考

1. 为什么 latent 统计量在**最后几个** epoch 收集，而不是在整个训练过程中收集？
2. 如果 `batch_size` 太小（如 1）或太大（如 256）会导致什么问题？LayerNorm 和 Adam 对此分别有何影响？
3. `torch.save(model.state_dict())` vs `torch.save(model)` 各有什么优缺点？什么场景下选择哪种？
