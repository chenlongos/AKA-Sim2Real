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
    STATE_DIM = int(os.getenv("STATE_DIM", "2"))  # [vel_left, vel_right] 轮子速度
    ACTION_DIM = int(os.getenv("ACTION_DIM", "2"))  # [vel_left, vel_right] 轮子速度
    ACTION_CHUNK_SIZE = int(os.getenv("ACTION_CHUNK_SIZE", "8"))
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "512"))

    # Redis 配置（多 worker + Socket.IO 必需）
    REDIS_URL = os.getenv("REDIS_URL", None)

    # CORS 配置（逗号分隔多个来源）
    _cors = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5175,https://act.chenlongrobot.com"
    )
    # CNB 预览环境自动注入域名，无需手动配 CORS
    _cnb_preview = os.getenv("CNB_VSCODE_PREVIEW_URL", "")
    if _cnb_preview:
        _cnb_preview = _cnb_preview.rstrip("/")
        _cors = f"{_cors},{_cnb_preview}"
    CORS_ORIGINS = _cors.split(",")

    # 模拟配置
    MAP_WIDTH = 800
    MAP_HEIGHT = 600


config = Config()
