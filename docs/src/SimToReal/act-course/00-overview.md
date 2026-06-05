# 第 0 讲：课程概述与数学基础

## 本课程的目标

本课程将逐行讲解 AKA-Sim2Real 项目中的 ACT（Action Chunking Transformer）实现代码。学完本课程后，你将能够：

1. 理解 ACT 论文中每个公式在代码中的具体实现位置
2. 掌握 Transformer 在机器人控制中的端到端应用方式
3. 能够独立修改、调试、优化 ACT 模型

## 前置知识

- 熟悉 Python 和 PyTorch 基本操作
- 了解 Transformer 的 Encoder-Decoder 结构（Self-Attention, Cross-Attention）
- 了解变分自编码器（VAE）的重参数化技巧
- 了解模仿学习（Imitation Learning / Behavior Cloning）的基本概念

## ACT 论文核心思想回顾

ACT（Action Chunking Transformer）来自论文 *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao et al., 2023)。其核心创新点：

### 1. 动作分块（Action Chunking）

传统行为克隆每一步预测一个动作 $a_t$，这会导致**复合误差（compounding error）**——每一步的小误差会随时间累积，最终偏离训练分布。

ACT 的解决方案：一次预测未来 $k$ 步的动作序列（称为一个 "chunk"）：

$$
\hat{a}_{t:t+k} = \pi\sb{\theta}(o_t)
$$

推理时，每次只执行 chunk 的第一个动作，然后重新预测。这相当于每次预测都"向前看"了 $k$ 步，减少了短视（myopic）行为。

### 2. CVAE（条件变分自编码器）

模仿学习中，同一个观测下可能有多种合理的动作（多模态性）。标准的回归模型会学习这些动作的"平均值"，导致模型输出模糊不清。

CVAE 引入一个潜变量 $z$ 来捕获这种多模态性：

- **训练时**：$z$ 从编码器 $q(z | o, a)$ 中采样（利用了"未来动作"的信息）
- **推理时**：$z$ 从先验分布 $p(z) = \mathcal{N}(0, I)$ 中采样

### 3. 时间集成（Temporal Ensembling）

由于每个时间步都会预测一个完整的 chunk，相邻时间步的预测会有重叠。Temporal Ensembling 利用指数加权平均来融合这些重叠的预测，进一步平滑输出：

$$ a_t^{(i)} = \sum_{j=0}^{i} w_j \cdot \hat{a}_t^{(i,j)} $$

其中 $w_j = \exp(-m \cdot j)$，$m$ 是一个小的衰减系数。

## 代码文件地图

本课程按以下顺序讲解代码文件：

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| 第 1 讲 | `configuration_act.py` + `defaults.py` | 模型超参数配置 |
| 第 2 讲 | `ACTDataset.py` | 数据加载与 QUANTILES 归一化 |
| 第 3 讲 | `modeling_act.py` (上) | 位置编码、视觉编码器、状态编码器 |
| 第 4 讲 | `modeling_act.py` (下) | Transformer Encoder/Decoder、CVAE、ACTModel |
| 第 5 讲 | `train_act.py` | 训练循环与检查点管理 |
| 第 6 讲 | `runtime.py` + `checkpoint.py` + `preprocess.py` + `execution.py` | 推理运行时系统 |

## 数学符号约定

| 符号 | 含义 | 代码中对应 |
|------|------|-----------|
| $B$ | batch 大小 | `batch_size` |
| $C$ | 图像通道数 | `in_channels`（默认 3）|
| $H, W$ | 图像高宽 | `image_size`（默认 224×224）|
| $d_s$ | 状态维度 | `state_dim`（默认 2: 左右轮速度）|
| $d_a$ | 动作维度 | `action_dim`（默认 2: 左右轮速度指令）|
| $k$ | action chunk 大小 | `action_chunk_size`（默认 8）|
| $d_h$ | Transformer 隐藏维度 | `hidden_dim`（默认 512）|
| $d_z$ | CVAE 潜变量维度 | `latent_dim`（默认 32）|

在阅读后续章节时，请随时回到这里查阅符号定义。
