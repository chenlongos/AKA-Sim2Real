"""ACT 最小可执行测试。"""

from pathlib import Path
import sys
import tempfile

import torch

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from policies.models.act.modeling_act import ACTModel
from policies.models.act.ACTDataset import ACTDataset
from policies.models.act.configuration_act import ACTConfig
from policies.models.act.defaults import act_config_to_dict


def test_act_config():
    """测试配置类"""
    print("测试 ACT 配置...")
    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
    )
    assert config.state_dim == 7
    assert config.action_dim == 7
    assert config.action_chunk_size == 16
    print("配置测试通过!")


def test_act_model_forward():
    """测试 ACT 模型前向传播"""
    print("\n测试 ACT 模型前向传播...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    model = ACTModel(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    images = torch.randn(batch_size, 1, 3, 224, 224)
    state = torch.randn(batch_size, 7)
    action_target = torch.randn(batch_size, 16, 7)

    output = model(images, state, action_target)

    assert output["action"].shape == (batch_size, 16, 7), f"Expected shape {(batch_size, 16, 7)}, got {output['action'].shape}"
    print(f"前向传播测试通过! 输出形状: {output['action'].shape}")


def test_act_model_get_action():
    """测试获取动作"""
    print("\n测试获取动作...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
    )

    model = ACTModel(config)
    model.eval()

    images = torch.randn(1, 1, 3, 224, 224)
    state = torch.randn(1, 7)

    with torch.no_grad():
        action = model.get_action(images, state)

    assert action.shape == (1, 16, 7), f"Expected shape {(1, 16, 7)}, got {action.shape}"
    print(f"获取动作测试通过! 动作形状: {action.shape}")


def test_act_dataset_temporal_actions():
    """测试逐时刻动作格式的数据集切片。"""
    print("\n测试数据集...")

    # 创建模拟数据 - 使用与 ACTDataset 兼容的格式
    # action 应该是 [total_timesteps, action_dim]，每个时间步一个动作
    num_samples = 100
    action_chunk_size = 16
    data = {
        "observation.image": torch.randn(num_samples, 1, 3, 224, 224),
        "observation.state": torch.randn(num_samples, 7),
        "action": torch.randn(num_samples, 7),  # [total_timesteps, action_dim]
    }

    dataset = ACTDataset(data, action_chunk_size=16)

    assert len(dataset) == num_samples - 16 + 1, f"Expected length {num_samples - 16 + 1}, got {len(dataset)}"

    sample = dataset[0]
    assert "observation" in sample
    assert "action" in sample
    assert sample["action"].shape == (16, 7)

    print(f"数据集测试通过! 数据集大小: {len(dataset)}")


def test_act_dataset_chunked_actions():
    """测试已 chunk 化动作格式的数据集读取。"""
    print("\n测试 chunked action 数据集...")

    data = {
        "observation.image": torch.randn(20, 1, 3, 224, 224),
        "observation.state": torch.randn(20, 7),
        "action": torch.randn(20, 16, 7),
    }

    dataset = ACTDataset(data, action_chunk_size=16)
    assert len(dataset) == 20
    assert dataset[0]["action"].shape == (16, 7)
    print("chunked action 数据集测试通过!")


def test_checkpoint_roundtrip():
    """测试 checkpoint 保存/加载格式。"""
    print("\n测试 checkpoint roundtrip...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    model = ACTModel(config)
    state_dict = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"
        checkpoint = {
            "model_state_dict": state_dict,
            "config": act_config_to_dict(config),
        }
        torch.save(checkpoint, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded_model = ACTModel(ACTConfig(**loaded["config"]))
        reloaded_model.load_state_dict(loaded["model_state_dict"])

    print("checkpoint roundtrip 测试通过!")


def main():
    """运行所有测试"""
    print("=" * 50)
    print("开始测试 ACT PyTorch 实现")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)

    # 运行测试
    test_act_config()
    test_act_model_forward()
    test_act_model_get_action()
    test_act_dataset_temporal_actions()
    test_act_dataset_chunked_actions()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
