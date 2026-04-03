"""Training orchestration for ACT models."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from policies.models.act.defaults import act_config_to_dict, build_act_config
from policies.models.act.modeling_act import ACTModel

from backend.services.training.dataset import SimpleDataset
from backend.services.training.dataset_loader import load_dataset
from backend.services.training.progress import TrainingCallbacks
from backend.services.training.state import training_state

logger = logging.getLogger(__name__)


def _extract_checkpoint_payload(checkpoint):
    """兼容旧格式 state_dict 和新格式 checkpoint dict。"""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    return checkpoint, {}


def _train_model_sync(
    sio_server,
    loop,
    data_dir: str = "output/dataset",
    output_dir: str = None,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    resume_from: str = None,
) -> Optional[ACTModel]:
    """训练 ACT 模型。"""
    training_state["is_running"] = True
    training_state["epoch"] = 0
    training_state["total_epochs"] = epochs
    training_state["loss"] = 0.0
    training_state["progress"] = 0.0

    callbacks = TrainingCallbacks(sio_server, loop=loop, namespace="/")

    try:
        logger.info("=" * 50)
        logger.info("开始训练ACT模型")
        if resume_from:
            logger.info(f"从已有模型继续训练: {resume_from}")
        logger.info("=" * 50)

        data = load_dataset(data_dir)
        action_dim = data["action"].shape[-1]
        raw_state_dim = data["observation.state"].shape[-1]
        state_dim = 2

        logger.info(f"action_dim: {action_dim}, state_dim: {state_dim} (原始: {raw_state_dim})")

        config = build_act_config(
            state_dim=2,
            action_dim=action_dim,
            action_chunk_size=8,
        )
        model = ACTModel(config)

        if resume_from:
            resume_path = Path(resume_from)
            if resume_path.exists():
                checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
                state_dict, checkpoint_config = _extract_checkpoint_payload(checkpoint)
                if checkpoint_config:
                    logger.info(f"检测到 checkpoint 配置: {checkpoint_config}")
                model.load_state_dict(state_dict)
                logger.info(f"已加载已有模型: {resume_path}")
            else:
                logger.warning(f"指定的可模型文件不存在: {resume_path}，从头开始训练")

        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        stats = data.get("stats")
        dataset = SimpleDataset(data, stats=stats)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        device = torch.device(device)
        model = model.to(device)
        logger.info(f"使用设备: {device}")

        model.train()
        total_batches = len(dataloader)

        all_mu = []
        all_log_sigma = []
        latent_collection_epochs = min(5, epochs // 2)

        for epoch in range(epochs):
            callbacks.on_epoch_start(epoch, epochs)

            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                images = batch["observation"]["image"].to(device)
                states = batch["observation"]["state"].to(device)
                actions = batch["action"].to(device)

                optimizer.zero_grad()
                output = model(images, states, action_target=actions, infer_cvae=False)
                predicted_actions = output["action"]
                kl_loss = output.get("kl_loss")
                mu = output.get("mu")
                log_sigma_x2 = output.get("log_sigma_x2")

                if config.use_cvae and mu is not None and log_sigma_x2 is not None and epoch >= epochs - latent_collection_epochs:
                    all_mu.append(mu.detach().cpu())
                    all_log_sigma.append(log_sigma_x2.detach().cpu())

                l1_loss = criterion(predicted_actions, actions)
                loss = l1_loss + kl_loss * config.kl_weight if kl_loss is not None else l1_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                callbacks.on_batch_end(loss.item(), batch_idx, total_batches)

            avg_loss = total_loss / total_batches
            callbacks.on_epoch_end(epoch, avg_loss)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        if output_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            output_path = project_root / "output" / "train"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if config.use_cvae and len(all_mu) > 0:
            all_mu_tensor = torch.cat(all_mu, dim=0)
            all_log_sigma_tensor = torch.cat(all_log_sigma, dim=0)
            latent_mu_mean = all_mu_tensor.mean(dim=0)
            latent_log_sigma_mean = all_log_sigma_tensor.mean(dim=0)

            logger.info(f"CVAE Latent: mu={latent_mu_mean.mean().item():.4f}, log_sigma={latent_log_sigma_mean.mean().item():.4f}")

            final_path = output_path / "final_model.pt"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "inference_latent_mu": latent_mu_mean,
                "inference_latent_log_sigma": latent_log_sigma_mean,
                "config": act_config_to_dict(config),
            }
            torch.save(checkpoint, final_path)
            logger.info(f"模型已保存到: {final_path} (包含 CVAE latent 统计)")
        else:
            final_path = output_path / "model.pt"
            torch.save(model.state_dict(), final_path)
            logger.info(f"模型已保存到: {final_path}")

        callbacks.on_train_end(str(final_path))
        return model
    except Exception as exc:
        logger.error(f"训练失败: {exc}")
        training_state["is_running"] = False
        raise


async def train_model(
    sio_server,
    data_dir: str = "output/dataset",
    output_dir: str = None,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    resume_from: str = None,
) -> Optional[ACTModel]:
    """训练 ACT 模型（异步包装器，在线程池中运行以避免阻塞事件循环）"""
    import asyncio

    loop = asyncio.get_running_loop()

    # 在线程池中运行同步训练函数
    return await loop.run_in_executor(
        None,
        lambda: _train_model_sync(
            sio_server,
            loop,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            resume_from=resume_from,
        )
    )
