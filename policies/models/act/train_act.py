"""
ACT 模型训练脚本
用于从采集的数据训练ACT模型并导出为HuggingFace格式
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import io
import base64

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from policies.models.act.modeling_act import ACTModel, ACTConfig
from policies.models.act.ACTDataset import ACTDataset


def load_dataset(data_dir: str = "dataset") -> Dict[str, torch.Tensor]:
    """
    加载导出的数据集

    Args:
        data_dir: 数据集目录

    Returns:
        包含训练数据的字典
    """
    data_path = Path(data_dir)

    # 加载统计信息
    with open(data_path / "meta" / "stats.json", "r") as f:
        stats = json.load(f)

    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32)

    # 加载所有parquet文件
    import pandas as pd
    import glob

    parquet_files = sorted(data_path.glob("data/chunk-*/file-*.parquet"))
    print(f"找到 {len(parquet_files)} 个数据文件")

    images = []
    states = []
    actions = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        # 加载图像
        for img_path in df["observation.image"]:
            full_path = data_path / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                img = img.resize((224, 224))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)
            else:
                # 如果图像不存在，使用零填充
                images.append(torch.zeros(3, 224, 224))

        # 加载状态
        for state_list in df["observation.state"]:
            state = torch.tensor(state_list, dtype=torch.float32)
            states.append(state)

        # 加载动作
        for action_list in df["action"]:
            action = torch.tensor(action_list, dtype=torch.float32)
            actions.append(action)

    # 转换为tensor
    images = torch.stack(images)  # [N 224, , 3,224]
    states = torch.stack(states)  # [N, state_dim]
    actions = torch.stack(actions)  # [N, action_chunk_size, action_dim]

    print(f"加载完成: {len(images)} 个样本")
    print(f"  图像形状: {images.shape}")
    print(f"  状态形状: {states.shape}")
    print(f"  动作形状: {actions.shape}")

    return {
        "observation.image": images,
        "observation.state": states,
        "action": actions,
    }


def train(
    data_dir: str = "dataset",
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    state_dim: int = 7,
    action_dim: int = 5,
    action_chunk_size: int = 16,
    hidden_dim: int = 512,
) -> ACTModel:
    """
    训练ACT模型

    Args:
        data_dir: 数据集目录
        output_dir: 模型输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        state_dim: 状态维度
        action_dim: 动作维度
        action_chunk_size: 动作分块大小
        hidden_dim: 隐藏层维度

    Returns:
        训练好的模型
    """
    print("=" * 50)
    print("开始训练ACT模型")
    print("=" * 50)

    # 加载数据
    data = load_dataset(data_dir)

    # 创建配置
    config = ACTConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_attention_heads=8,
        dim_feedforward=3200,  # 根据 LeRobot
        use_cvae=False,  # 简化训练
        use_temporal_ensembling=False,
        use_spatial_softmax=True,
        latent_dim=32,
    )

    # 创建模型
    model = ACTModel(config)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建数据集和数据加载器
    dataset = ACTDataset(
        data,
        action_chunk_size=action_chunk_size,
        normalize_images=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练循环
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            images = batch["observation"]["image"].to(device)
            states = batch["observation"]["state"].to(device)
            actions = batch["action"].to(device)

            # 前向传播
            optimizer.zero_grad()

            # 预测动作
            predicted_actions = model.get_action(
                images,
                states,
                use_temporal_ensembling=False,
            )

            # 计算损失
            loss = criterion(predicted_actions, actions)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  保存检查点: {checkpoint_path}")

    # 保存最终模型
    final_path = Path(output_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n模型已保存到: {final_path}")

    return model


def export_to_huggingface(
    model: ACTModel,
    config: ACTConfig,
    output_dir: str = "huggingface_model",
    model_name: str = "aka-sim-act",
) -> str:
    """
    导出模型为HuggingFace格式

    Args:
        model: 训练好的模型
        config: 模型配置
        output_dir: 输出目录
        model_name: 模型名称

    Returns:
        输出目录路径
    """
    from huggingface_hub import HfApi, create_repo

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    torch.save(model.state_dict(), output_path / "pytorch_model.bin")

    # 保存配置
    config_dict = {
        "model_type": "act",
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "action_chunk_size": config.action_chunk_size,
        "hidden_dim": config.hidden_dim,
        "num_attention_heads": config.num_attention_heads,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "image_size": config.image_size,
        "in_channels": config.in_channels,
        "num_cameras": config.num_cameras,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 保存推理代码
    inference_code = '''"""
ACT 模型推理代码
"""
import torch
from pathlib import Path
import json
from typing import List

# 加载配置
config_path = Path(__file__).parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

# 加载模型 (需要自行实现加载逻辑)
# from act_model import ACTModel
# model = ACTModel(config)
# model.load_state_dict(torch.load("pytorch_model.bin"))
# model.eval()

def infer(state: List[float], image_path: str = None) -> List[List[float]]:
    """
    推理函数

    Args:
        state: 状态向量 [state_dim]
        image_path: 图像路径 (可选)

    Returns:
        预测的动作序列 [action_chunk_size, action_dim]
    """
    # TODO: 实现推理逻辑
    pass
'''

    with open(output_path / "inference.py", "w") as f:
        f.write(inference_code)

    # 保存 README
    readme = f'''# {model_name}

AKA-Sim ACT (Action Chunking Transformer) 模型

## 模型配置

- state_dim: {config.state_dim}
- action_dim: {config.action_dim}
- action_chunk_size: {config.action_chunk_size}
- hidden_dim: {config.hidden_dim}
- num_encoder_layers: {config.num_encoder_layers}
- num_decoder_layers: {config.num_decoder_layers}
- num_attention_heads: {config.num_attention_heads}

## 使用方法

```python
import torch
from act_model_pytorch import ACTModel, ACTConfig

# 加载配置
config = ACTConfig(
    state_dim={config.state_dim},
    action_dim={config.action_dim},
    action_chunk_size={config.action_chunk_size},
    hidden_dim={config.hidden_dim},
)

# 创建模型
model = ACTModel(config)

# 加载权重
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# 推理
with torch.no_grad():
    state = torch.randn(1, {config.state_dim})
    image = torch.randn(1, 1, 3, 224, 224)
    action = model.get_action(image, state)
```
'''

    with open(output_path / "README.md", "w") as f:
        f.write(readme)

    print(f"\nHuggingFace 模型已导出到: {output_path}")
    print(f"  - pytorch_model.bin: 模型权重")
    print(f"  - config.json: 模型配置")
    print(f"  - inference.py: 推理代码")
    print(f"  - README.md: 使用说明")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="训练ACT模型")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--export_hf", action="store_true", help="导出为HuggingFace格式")

    args = parser.parse_args()

    # 训练模型
    model = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 导出为HuggingFace格式
    if args.export_hf:
        config = ACTConfig(
            state_dim=7,
            action_dim=5,
            action_chunk_size=16,
            hidden_dim=512,
        )
        export_to_huggingface(model, config)


if __name__ == "__main__":
    main()
