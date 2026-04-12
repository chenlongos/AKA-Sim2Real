# 模型推理

模型推理阶段，系统使用训练好的 ACT 模型替代人工操控，实现小车自主驾驶。系统同时支持**模拟器（Sim）**和**真实小车（Real）**两种模式。

---

## 推理前提

- 已完成[模型训练](training.md)，`output/train/model.pt` 存在
- 后端和前端均已启动

---

## 加载模型

### 方式一：自动加载（推荐）

后端启动时会自动尝试加载 `output/train/model.pt`，无需手动操作。可通过以下接口确认模型状态：

```bash
curl http://localhost:8000/health
```

返回示例：

```json
{
  "model_loaded": true,
  "device": "mps"
}
```

### 方式二：前端手动加载

在界面上点击 **"加载模型"** 按钮，系统会加载最新的模型检查点。

### 方式三：REST API

```bash
curl -X POST http://localhost:8000/api/inference/load
```

---

## 启动自动推理

### 前端操作

点击 **"自动推理"** 按钮，系统进入自动驾驶模式：

1. 摄像头每帧捕获当前图像
2. 图像与车辆状态输入 ACT 模型
3. 模型输出动作块（action chunk，8 个时间步）
4. 取第一个动作执行，循环往复

### Socket.IO 事件

推理通过 Socket.IO 实时传输控制指令：

| 事件 | 方向 | 说明 |
|------|------|------|
| `start_inference` | 客户端 → 服务端 | 启动自动推理 |
| `stop_inference` | 客户端 → 服务端 | 停止自动推理 |
| `inference_action` | 服务端 → 客户端 | 下发控制动作 |

---

## 推理流程详解

```
摄像头图像 ──┐
             ├──▶ ACTInferenceRuntime.infer()
车辆状态   ──┘         │
                       ├── 图像预处理（ResNet18 归一化）
                       ├── 状态归一化
                       ├── Transformer 编码解码
                       ├── 输出 action chunk [8, 2]
                       ├── （可选）时序集成（Temporal Ensembling）
                       └── 动作反归一化 → 发送给小车
```

**图像预处理**：图像统一缩放并按 ImageNet 均值/方差归一化后送入 ResNet18 视觉编码器。

**状态归一化**：使用数据集中保存的 `state_mean` 和 `state_std` 进行 Z-score 归一化。

**动作反归一化**：输出的动作使用 `action_mean` 和 `action_std` 还原为真实控制量。

---

## 时序集成（Temporal Ensembling）

时序集成是 ACT 论文中提出的平滑技术，通过对多个历史 action chunk 的加权融合来减少控制抖动。

**启用方式**（设置环境变量）：

```bash
export ACT_TEMPORAL_ENSEMBLING=1
python backend/main.py
```

**衰减权重**由 `temporal_ensembling_weight` 配置参数控制（默认 0.5），值越小融合越平滑。

> **注意**：当前默认关闭时序集成，适合大多数场景。如控制输出存在抖动，可尝试开启。

---

## Sim 与 Real 模式

系统同时支持两个 Socket.IO 命名空间：

| 命名空间 | 用途 |
|----------|------|
| `/sim` | 模拟器推理，控制虚拟小车 |
| `/real` | 真实小车推理，通过硬件接口控制实体小车 |

两个命名空间使用相同的推理逻辑，仅底层执行器（模拟器控制器 vs 真实硬件驱动）不同。

---

## 推理 REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 查询模型加载状态 |
| `POST` | `/api/inference/load` | 加载模型 |
| `POST` | `/api/inference/reset` | 重置推理上下文（清空时序集成缓存） |

---

## 推理调试

查看后端日志中的推理输出，确认模型正常运行：

```
INFO: 加载 ACT 模型...
INFO: 状态归一化: mean=[0.0, 0.0], std=[1.0, 1.0]
INFO: 动作归一化: mean=[0.0, 0.0], std=[1.0, 1.0]
INFO: ACT 模型加载完成，使用设备: mps
```

若模型未加载时触发推理，系统会返回零动作 `[0.0, 0.0]` 并打印警告日志，而非抛出异常，保证系统鲁棒性。
