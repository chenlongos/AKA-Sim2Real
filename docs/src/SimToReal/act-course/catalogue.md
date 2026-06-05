# ACT 算法精讲

本课程将 ACT（Action Chunking Transformer）的所有实现代码逐行拆解讲解，以算法课的方式带你深入理解每一行代码背后的数学原理和设计决策。

## 课程目录

| 章节 | 标题 | 核心内容 |
|------|------|---------|
| [第 0 讲](00-overview.md) | 课程概述与数学基础 | ACT 核心思想、代码地图、符号约定 |
| [第 1 讲](01-configuration.md) | 配置系统 | ACTConfig 25 个超参数详解、参数校验、默认配置 |
| [第 2 讲](02-dataset.md) | 数据集与归一化 | ACTDataset 滑窗采样、QUANTILES 百分位数归一化 |
| [第 3 讲](03-modeling-part1.md) | 模型实现（上） | Temporal Ensembling、2D 正弦位置编码、ResNet18 视觉编码器、状态编码器 |
| [第 4 讲](04-modeling-part2.md) | 模型实现（下） | Transformer Encoder/Decoder（Pre-Norm）、CVAE 重参数化、ACTModel 完整前向传播 |
| [第 5 讲](05-training.md) | 训练流程 | Parquet 数据加载、训练循环、CVAE latent 统计收集、检查点保存 |
| [第 6 讲](06-inference.md) | 推理运行时系统 | 检查点加载、预处理管线、Temporal Ensembling 策略、Singletone 运行时 |

## 如何使用本课程

### 推荐阅读路径

1. **初学者**：按顺序 0→1→2→3→4→5→6 阅读
2. **想快速理解核心**：先读第 0 讲和第 4 讲（ACTModel.forward()）
3. **想部署推理**：读第 1 讲（配置）、第 6 讲（推理系统）
4. **想调优训练**：读第 2 讲（数据集）、第 5 讲（训练）

### 代码对照阅读

每讲的标题下方标注了对应的源文件。建议打开源文件和本课程并排阅读——先看课程讲解，再看代码实现。

## 参考资料

- [ACT 原论文: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [LeRobot ACT 实现（HuggingFace）](https://github.com/huggingface/lerobot)
- [Transformer: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ResNet: Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114)
