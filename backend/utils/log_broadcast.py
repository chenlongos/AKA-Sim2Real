"""
日志广播系统 - 将日志通过 Socket.IO 发送到前端
支持按命名空间隔离

注意：logging handler 的 emit 可能在没有 event loop 的线程中被调用
因此日志广播主要用于简单的开始/结束事件
训练进度通过 TrainingCallbacks.on_epoch_end 发送
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

# 全局 Socket.IO 服务器引用
_sio_server = None
# 主线程的 event loop（用于跨线程调度）
_main_loop = None
# 按命名空间跟踪状态: { namespace: { sids: set(), ... } }
_namespace_states: dict[str, dict[str, Any]] = {}


def set_broadcast_sio(sio_server, namespace: str = "/"):
    """设置 Socket.IO 服务器用于广播日志"""
    global _sio_server, _main_loop
    _sio_server = sio_server
    # 保存主线程的 loop
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
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


def emit_log_message(log_entry: dict, namespace: str = "/", sid: str = None):
    """手动发送日志消息（供 TrainingCallbacks 等调用）"""
    if _sio_server is None:
        return
    target_sid = sid or _namespace_states.get(namespace, {}).get("sids", set())
    if isinstance(target_sid, set) and target_sid:
        target_sid = next(iter(target_sid))
    if target_sid:
        _sio_server.emit("log_message", log_entry, room=target_sid, namespace=namespace)


class SocketIOHandler(logging.Handler):
    """将日志发送到前端的日志处理器 - 仅广播关键事件"""

    # 只广播这些模块的日志（白名单）
    BROADCAST_LOGGERS = {
        "backend.services.training.orchestrator",
        "backend.sio_handlers.domains.episode.events",
        "backend.api.domains.training.routes",
        "backend.api.domains.episode.routes",
    }

    def emit(self, record: logging.LogRecord):
        if _sio_server is None:
            return

        # 只广播白名单中的日志
        if record.name not in self.BROADCAST_LOGGERS:
            return

        try:
            log_entry = self.format_log(record)

            # 只发给同一命名空间的连接
            for namespace, ns_state in _namespace_states.items():
                sids = ns_state.get("sids", set())
                for sid in sids:
                    try:
                        self._emit_async(log_entry, namespace, sid)
                    except Exception:
                        pass
        except Exception:
            pass

    def _emit_async(self, log_entry: dict, namespace: str, sid: str):
        """通过主线程的 loop 异步发送"""
        if _main_loop is None:
            return
        def _do_emit():
            asyncio.ensure_future(
                _sio_server.emit(
                    "log_message",
                    log_entry,
                    room=sid,
                    namespace=namespace,
                )
            )
        _main_loop.call_soon_threadsafe(_do_emit)

    def format_log(self, record: logging.LogRecord) -> dict[str, Any]:
        """格式化日志记录为字典"""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)

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


def setup_socket_logging(level: int = logging.WARNING):
    """设置 Socket.IO 日志广播"""
    global _socket_handler

    if _socket_handler is not None:
        return

    _socket_handler = SocketIOHandler()
    _socket_handler.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    _socket_handler.setFormatter(formatter)

    # 直接添加到需要广播的模块的 logger
    for logger_name in SocketIOHandler.BROADCAST_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.addHandler(_socket_handler)
        logger.setLevel(logging.INFO)


def remove_socket_logging():
    """移除 Socket.IO 日志广播"""
    global _socket_handler

    if _socket_handler is None:
        return

    root_logger = logging.getLogger()
    root_logger.removeHandler(_socket_handler)
    _socket_handler = None