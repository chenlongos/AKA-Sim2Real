# 模型训练

采集到足够的演示数据后，即可训练 ACT（Action Chunking Transformer）模型。训练过程由后端的训练编排器（Training Orchestrator）自动完成，无需手动干预。

---

## 训练前提

- 已完成[数据采集](data-collection.md)，数据保存在 `output/dataset/`
- 数据集中至少有 **100 个样本**

---

## 启动训练

在前端界面点击 **"开始训练"** 按钮，或通过 REST API 触发：

```bash
curl -X POST http://localhost:8000/api/training/start
```

---

## 训练流程详解

训练器（`backend/services/training/orchestrator.py`）执行以下步骤：

```
数据集加载
    │
    ▼
构建 ACT 配置
    │
    ▼
初始化模型（ACTModel）
    │
    ▼
训练循环（L1 Loss + KL Loss）
    │
    ▼
保存模型检查点
output/train/model.pt（含 CVAE 潜变量统计）
```

### 损失函数

ACT 使用两个损失项联合训练：

| 损失 | 公式 | 说明 |
|------|------|------|
| **重建损失** | L1(predicted_actions, gt_actions) | 动作预测的主要监督信号 |
| **KL 散度** | KL(q(z\|obs,action) ∥ p(z\|obs)) | CVAE 正则项，权重为 `kl_weight=0.1` |

总损失：`Loss = L1_loss + kl_weight × KL_loss`

---

## 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_dim` | 2 | 状态维度（左右轮速） |
| `action_dim` | 2 | 动作维度（左右轮指令） |
| `action_chunk_size` | 8 | 每次预测的动作步数 |
| `hidden_dim` | 512 | Transformer 隐层维度 |
| `num_attention_heads` | 8 | 多头注意力头数 |
| `num_encoder_layers` | 4 | Transformer 编码器层数 |
| `num_decoder_layers` | 4 | Transformer 解码器层数 |
| `dim_feedforward` | 3200 | 前馈网络中间维度 |
| `kl_weight` | 0.1 | KL 散度损失权重 |
| `latent_dim` | 32 | CVAE 潜变量维度 |
| `use_cvae` | True | 是否启用 CVAE |

---

## 训练输出

训练完成后，模型保存在：

```
output/train/
├── model.pt           # 完整模型检查点（含 CVAE 潜变量统计）
└── final_model.pt     # 最终模型（如启用了 CVAE 则包含 latent stats）
```

检查点内容：

```python
{
    "model_state_dict": ...,   # 模型权重
    "config": { ... },         # ACT 配置字典
    "latent_mean": ...,        # CVAE 潜变量均值（用于推理时采样）
    "latent_std":  ...,        # CVAE 潜变量标准差
}
```

---

## 查看训练状态

通过 REST API 查询训练进度：

```bash
# 获取训练状态
curl http://localhost:8000/api/training/status

# 停止训练
curl -X POST http://localhost:8000/api/training/stop
```

---

## 单独运行训练脚本（高级）

如需在命令行直接运行训练（不通过 Web 界面）：

```bash
python policies/models/act/train_act.py \
    --dataset_dir output/dataset \
    --output_dir output/train
```

---

## 下一步

训练完成后，继续进行 [模型推理](inference.md)，让小车自主行驶。
