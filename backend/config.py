"""
AKA-Sim 后端 - 配置模块
"""

import os


class Config:
    """应用配置"""

    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # 模型配置
    MODEL_PATH = os.getenv("MODEL_PATH", None)
    STATE_DIM = int(os.getenv("STATE_DIM", "3"))  # [x_vel, y_vel, theta_vel]
    ACTION_DIM = int(os.getenv("ACTION_DIM", "3"))  # [x_vel, y_vel, theta_vel]
    ACTION_CHUNK_SIZE = int(os.getenv("ACTION_CHUNK_SIZE", "8"))
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "512"))

    # 模拟配置
    MAP_WIDTH = 800
    MAP_HEIGHT = 600


config = Config()
