"""
日志广播系统 - 将日志通过 Socket.IO 发送到前端
支持按命名空间隔离
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

# 全局 Socket.IO 服务器引用
_sio_server = None
# 按命名空间跟踪状态: { namespace: { sids: set(), ... } }
_namespace_states: dict[str, dict[str, Any]] = {}


def set_broadcast_sio(sio_server, namespace: str = "/"):
    """设置 Socket.IO 服务器用于广播日志"""
    global _sio_server
    _sio_server = sio_server
    # 初始化命名空间状态
    if namespace not in _namespace_states:
        _namespace_states[namespace] = {
            "sids": set(),
        }


def add_connected_sid(sid: str, namespace: str = "/"):
    """添加连接的sid"""
    if namespace not in _namespace_states:
        _namespace_states[namespace] = {"sids": set()}
    _namespace_states[namespace]["sids"].add(sid)


def remove_connected_sid(sid: str, namespace: str = "/"):
    """移除连接的sid"""
    if namespace in _namespace_states:
        _namespace_states[namespace]["sids"].discard(sid)


class SocketIOHandler(logging.Handler):
    """将日志发送到前端的日志处理器"""

    # 需要过滤的日志消息前缀
    FILTERED_PREFIXES = [
        "已重置 ACT 推理时序上下文",
        "[on_action] 用户控制，退出推理模式:",
        "[游戏循环] frame=",
    ]

    def emit(self, record: logging.LogRecord):
        if _sio_server is None:
            return

        try:
            # 格式化消息以检查是否需要过滤
            try:
                message = record.getMessage()
            except Exception:
                message = str(record.msg)

            # 检查是否需要过滤
            for prefix in self.FILTERED_PREFIXES:
                if message.startswith(prefix):
                    return

            log_entry = self.format_log(record)

            # 给每个命名空间的每个连接单独发送日志
            for namespace, ns_state in _namespace_states.items():
                sids = ns_state.get("sids", set())
                for sid in sids:
                    try:
                        asyncio.create_task(
                            _sio_server.emit(
                                "log_message",
                                log_entry,
                                room=sid,
                                namespace=namespace,
                            )
                        )
                    except Exception:
                        pass
        except Exception:
            # 避免日志处理失败影响主程序
            pass

    def format_log(self, record: logging.LogRecord) -> dict[str, Any]:
        """格式化日志记录为字典"""
        # 获取时间戳（本地时间）
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # 格式化消息
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)

        # 添加异常信息（如果有）
        if record.exc_info:
            import traceback
            exc_text = "".join(traceback.format_exception(*record.exc_info))
            message = f"{message}\n{exc_text}"

        return {
            "timestamp": timestamp,
            "level": record.levelname,
            "levelno": record.levelno,
            "message": message,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }


# 全局处理器实例
_socket_handler: SocketIOHandler | None = None


def setup_socket_logging(level: int = logging.INFO):
    """设置 Socket.IO 日志广播"""
    global _socket_handler

    if _socket_handler is not None:
        return

    root_logger = logging.getLogger()

    # 创建并添加 Socket.IO 处理器
    _socket_handler = SocketIOHandler()
    _socket_handler.setLevel(level)

    # 使用简单的格式（由 format_log 处理详细格式）
    formatter = logging.Formatter("%(message)s")
    _socket_handler.setFormatter(formatter)

    root_logger.addHandler(_socket_handler)
    logging.info("Socket.IO 日志广播已启用")


def remove_socket_logging():
    """移除 Socket.IO 日志广播"""
    global _socket_handler

    if _socket_handler is None:
        return

    root_logger = logging.getLogger()
    root_logger.removeHandler(_socket_handler)
    _socket_handler = None
