# ACT 模型架构

ACT（Action Chunking Transformer）是一种基于 Transformer 的模仿学习策略，通过预测**动作块（action chunk）**而非单步动作来实现平滑、稳定的控制输出。

---

## 整体架构图

```
                     ┌──────────────────────────────────────┐
                     │           ACTModel                   │
                     │                                      │
  图像输入            │  ┌──────────────┐                   │
  (H×W×3) ─────────▶│  │  RGBEncoder   │──▶ 视觉特征 (D)   │
                     │  │  (ResNet18)   │                   │
                     │  └──────────────┘         │          │
                     │                           │          │
  状态输入            │  ┌──────────────┐         │          │
  [vel_l, vel_r] ──▶│  │ StateEncoder  │──▶ 状态特征 (D)   │
                     │  │    (MLP)      │         │          │
                     │  └──────────────┘         │          │
                     │                           ▼          │
  ┌──────────┐       │               ┌────────────────┐     │
  │   CVAE   │──────▶│               │ Transformer    │     │
  │ (训练时)  │  z   │               │ Encoder-Decoder│     │
  └──────────┘       │               └────────┬───────┘     │
                     │                        │             │
                     │                        ▼             │
                     │               ┌────────────────┐     │
                     │               │  动作预测头     │     │
                     │               │  (Linear)      │     │
                     │               └────────────────┘     │
                     │                        │             │
                     └────────────────────────┼─────────────┘
                                              │
                                              ▼
                              action_chunk [chunk_size, action_dim]
                              （默认 8 步 × 2 维）
```

---

## 核心组件

### 1. RGBEncoder（视觉编码器）

- 骨干网络：**ResNet18**（ImageNet 预训练）
- 输入：归一化后的 RGB 图像
- 输出：展平的视觉特征向量，投影到 `hidden_dim`（默认 512）
- 图像预处理：resize → ImageNet 均值/方差归一化

### 2. StateEncoder（状态编码器）

- 结构：简单 MLP（多层感知机）
- 输入：归一化后的车辆状态 `[vel_left, vel_right]`（维度 2）
- 输出：状态特征向量（维度 `hidden_dim`）

### 3. CVAE（条件变分自编码器）

CVAE 在**训练阶段**引入潜变量 `z`，为动作预测提供多模态表达能力。

```
训练时：
  encoder: (观测, 动作序列) → (μ, σ) → z ~ N(μ, σ²)

推理时：
  z ~ N(latent_mean, latent_std)  # 从训练分布中采样
  或 z = 0                        # 使用零向量（确定性推理）
```

KL 散度损失约束潜变量分布接近标准正态分布：

```
KL Loss = KL(N(μ, σ²) ∥ N(0, I))
```

### 4. Transformer Encoder-Decoder

| 参数 | 默认值 |
|------|--------|
| `hidden_dim` | 512 |
| `num_attention_heads` | 8 |
| `num_encoder_layers` | 4 |
| `num_decoder_layers` | 4 |
| `dim_feedforward` | 3200 |

- **编码器**：将视觉特征 + 状态特征 + CVAE 潜变量拼接，经多层自注意力编码为上下文表示
- **解码器**：以可学习的动作查询（action queries）为输入，通过交叉注意力从编码器上下文中提取信息，输出 `action_chunk_size` 个动作向量

### 5. 动作预测头

线性层将 Transformer 解码器的输出投影到动作空间（维度 `action_dim=2`），输出归一化后的动作块。

---

## 动作块（Action Chunking）

ACT 的核心创新之一是**不预测单步动作，而是一次性预测 `action_chunk_size` 步的动作序列**（默认 8 步）。

**优势**：
- 缓解复合误差（compound error）
- 提升长序列动作的时间一致性
- 减少控制器与环境之间的高频交互需求

**执行策略**：推理时每次只执行 action chunk 的第一步动作（或使用时序集成），下一帧重新预测。

---

## 时序集成（Temporal Ensembling）

时序集成通过加权融合多个历史 action chunk 的预测，进一步平滑控制输出。

```
当前动作 = α × 当前 chunk[0] + (1-α) × 历史融合动作
```

权重衰减系数 `α`（`temporal_ensembling_weight`，默认 0.5）控制历史信息的保留程度：
- 值越大：跟随最新预测，响应速度快
- 值越小：保留更多历史，控制更平滑

---

## 代码层次结构

```
policies/models/act/
├── modeling_act.py         # ACTModel、RGBEncoder、StateEncoder、CVAE、Transformer 层
├── configuration_act.py    # ACTConfig 数据类
├── defaults.py             # DEFAULT_ACT_CONFIG 与 build_act_config()
└── train_act.py            # 独立训练脚本

backend/services/inference/
├── checkpoint.py           # 检查点加载、模型实例化、统计量读取
├── preprocess.py           # 图像预处理、状态/动作归一化与反归一化
├── execution.py            # TemporalEnsemblingPolicy（时序集成）
└── runtime.py              # ACTInferenceRuntime（推理运行时装配）
```

---

## 训练损失汇总

| 损失项 | 权重 | 说明 |
|--------|------|------|
| L1 重建损失 | 1.0 | 预测动作块与真实动作的 L1 误差 |
| KL 散度 | 0.1（`kl_weight`）| CVAE 潜变量正则化 |

**总损失** = L1_loss + 0.1 × KL_loss

---

## 参考资料

- [ACT 原论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [ResNet 论文：Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Transformer 原论文：Attention Is All You Need](https://arxiv.org/abs/1706.03762)
