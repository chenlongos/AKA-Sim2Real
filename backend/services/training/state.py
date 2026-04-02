"""Shared training progress state."""

training_state = {
    "is_running": False,
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "progress": 0.0,
}


def get_training_state():
    """Return training progress state."""
    return training_state
