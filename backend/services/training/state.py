"""Shared training progress state."""

# 按用户隔离的训练状态
# 格式: { user_id: training_state }
_user_training_states: dict[str, dict] = {}


def _ensure_user_state(user_id: str) -> dict:
    """确保用户训练状态存在"""
    if user_id not in _user_training_states:
        _user_training_states[user_id] = {
            "is_running": False,
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "progress": 0.0,
        }
    return _user_training_states[user_id]


def get_training_state(user_id: str) -> dict:
    """Return training progress state for user."""
    return _ensure_user_state(user_id)


def stop_training(user_id: str):
    """Stop training for user."""
    if user_id in _user_training_states:
        _user_training_states[user_id]["is_running"] = False
