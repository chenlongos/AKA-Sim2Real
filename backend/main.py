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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services import act_model
import api
from api import router as api_router
from config import config
from sio_handlers import SimNamespace, start_game_loop

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 50)
    logger.info("AKA-Sim Backend 启动")
    logger.info("=" * 50)

    # 尝试加载模型
    # TODO 没必要
    try:
        act_model.load_act_model()
    except Exception as e:
        logger.warning(f"模型加载失败: {e}")
        logger.warning("将以无模型模式运行")

    # 启动状态广播
    start_game_loop(sio)

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
sio.register_namespace(SimNamespace("/"))

# 设置sio_server到api模块
api.set_sio_server(sio)

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
