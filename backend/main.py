"""
AKA-Sim 后端服务
FastAPI + Socket.IO 实现
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from socketio import AsyncServer

# 添加仓库根目录到路径，兼容 `python backend/main.py`
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.services import inference
from backend import api
from backend.api import router as api_router
from backend.config import config
from backend.sio_handlers import SimNamespace, RealNamespace, start_game_loop, set_act_runtime as set_sio_act_runtime
from backend.utils import set_broadcast_sio, setup_socket_logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
runtime = inference.get_act_runtime()
api.set_act_runtime(runtime)
set_sio_act_runtime(runtime)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 50)
    logger.info("AKA-Sim Backend 启动")
    logger.info("=" * 50)

    # 尝试加载模型
    # TODO 没必要
    try:
        runtime.load_model()
    except Exception as e:
        logger.warning(f"模型加载失败: {e}")
        logger.warning("将以无模型模式运行")

    # 启动两个命名空间的状态广播
    start_game_loop(sio, namespace="/sim")
    start_game_loop(sio, namespace="/real")

    yield

    logger.info("=" * 50)
    logger.info("AKA-Sim Backend 关闭")
    logger.info("=" * 50)


# FastAPI 应用
app = FastAPI(
    title="AKA-Sim Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Socket.IO 服务器
sio = AsyncServer(
    async_mode="asgi",
    ping_timeout=60,
    ping_interval=25,
)

# 注册两个独立的命名空间
sio.register_namespace(SimNamespace("/sim"))
sio.register_namespace(RealNamespace("/real"))

# 设置sio_server到api模块
api.set_sio_server(sio)

# 设置日志广播 - 初始化两个命名空间
set_broadcast_sio(sio, namespace="/sim")
set_broadcast_sio(sio, namespace="/real")
setup_socket_logging(level=logging.INFO)

# 测试日志
logger.info("后端服务启动，日志广播系统已就绪")

# 挂载 Socket.IO 到 FastAPI (使用 ASGI 应用)
from socketio.asgi import ASGIApp
app.mount("/socket.io", ASGIApp(sio))

# 注册 API 路由
app.include_router(api_router)


# 主入口
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )
