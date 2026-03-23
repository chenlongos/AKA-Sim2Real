"""ACT minimal executable tests."""

from pathlib import Path

import torch

from policies.models.act.modeling_act import ACTModel
from policies.models.act.ACTDataset import ACTDataset
from policies.models.act.configuration_act import ACTConfig
from policies.models.act.defaults import act_config_to_dict


def test_act_config():
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

    assert output["action"].shape == (batch_size, 16, 7)
    print(f"前向传播测试通过! 输出形状: {output['action'].shape}")


def test_act_model_get_action():
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

    assert action.shape == (1, 16, 7)
    print(f"获取动作测试通过! 动作形状: {action.shape}")


def test_act_dataset_temporal_actions():
    print("\n测试数据集...")

    num_samples = 100
    data = {
        "observation.image": torch.randn(num_samples, 1, 3, 224, 224),
        "observation.state": torch.randn(num_samples, 7),
        "action": torch.randn(num_samples, 7),
    }

    dataset = ACTDataset(data, action_chunk_size=16)

    assert len(dataset) == num_samples - 16 + 1
    sample = dataset[0]
    assert "observation" in sample
    assert "action" in sample
    assert sample["action"].shape == (16, 7)
    print(f"数据集测试通过! 数据集大小: {len(dataset)}")


def test_act_dataset_chunked_actions():
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


def test_checkpoint_roundtrip(tmp_path: Path):
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

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": state_dict,
        "config": act_config_to_dict(config),
    }
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded_model = ACTModel(ACTConfig(**loaded["config"]))
    reloaded_model.load_state_dict(loaded["model_state_dict"])

    print("checkpoint roundtrip 测试通过!")
