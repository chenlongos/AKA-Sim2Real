import {useCallback, useEffect, useRef, useState} from "react";
import {socket} from "../../api/socket.ts";

const MAX_LOGS = 500;

interface LogEntry {
    id: string;
    timestamp: string;
    level: "DEBUG" | "INFO" | "WARNING" | "ERROR";
    levelno: number;
    message: string;
    logger: string;
    module: string;
    line: number;
}

interface LogConsoleProps {
    className?: string;
}

const levelBgColors: Record<string, string> = {
    DEBUG: "bg-gray-800",
    INFO: "",
    WARNING: "bg-yellow-900/30",
    ERROR: "bg-red-900/30",
};

export const LogConsole = ({className = ""}: LogConsoleProps) => {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [shouldScroll, setShouldScroll] = useState(true);
    const logContainerRef = useRef<HTMLDivElement>(null);

    // 日志更新时，如果当前在底部则滚动到底部
    useEffect(() => {
        if (shouldScroll && logContainerRef.current) {
            const container = logContainerRef.current;
            container.scrollTop = container.scrollHeight;
        }
    }, [logs, shouldScroll]);

    // 监听滚动事件，判断是否应该自动滚动
    const handleScroll = useCallback(() => {
        if (!logContainerRef.current) return;
        const {scrollTop, scrollHeight, clientHeight} = logContainerRef.current;
        // 当距离底部小于 100px 时，认为是在底部
        const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;
        setShouldScroll(isAtBottom);
    }, []);

    // 监听 socket 日志
    useEffect(() => {
        let logIdCounter = 0;

        const handleLogMessage = (data: Omit<LogEntry, "id">) => {
            console.log("收到日志消息:", data);

            const entry: LogEntry = {
                ...data,
                id: `log-${++logIdCounter}`,
            };

            setLogs(prev => {
                const newLogs = [...prev, entry];
                if (newLogs.length > MAX_LOGS) {
                    return newLogs.slice(newLogs.length - MAX_LOGS);
                }
                return newLogs;
            });
        };

        console.log("开始监听 log_message 事件");
        socket.on("log_message", handleLogMessage);

        // 测试：添加一条本地日志
        const testEntry: LogEntry = {
            id: `log-local-${Date.now()}`,
            timestamp: new Date().toLocaleTimeString("zh-CN", {hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3}),
            level: "INFO",
            levelno: 20,
            message: "前端日志控制台已就绪，等待后端日志...",
            logger: "frontend.LogConsole",
            module: "LogConsole",
            line: 0,
        };
        setLogs([testEntry]);

        return () => {
            socket.off("log_message", handleLogMessage);
        };
    }, []);

    // 清空日志
    const clearLogs = useCallback(() => {
        setLogs([]);
    }, []);

    return (
        <div className={`flex flex-col bg-gray-900 border-2 border-gray-700 rounded-lg overflow-hidden ${className}`}>
            {/* 自定义滚动条样式 */}
            <style>{`
                .log-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .log-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                .log-scrollbar::-webkit-scrollbar-thumb {
                    background-color: rgba(156, 163, 175, 0.4);
                    border-radius: 3px;
                }
                .log-scrollbar::-webkit-scrollbar-thumb:hover {
                    background-color: rgba(156, 163, 175, 0.6);
                }
            `}</style>

            {/* 工具栏 */}
            <div className="flex items-center justify-between bg-gray-800 px-3 py-2 border-b border-gray-700">
                <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-gray-300">后台日志</span>
                    <span className="text-xs text-gray-500">
                        ({logs.length}/{MAX_LOGS})
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    {/* 级别过滤 */}
                    {/* <select
                        value={filterLevel}
                        onChange={(e) => setFilterLevel(e.target.value)}
                        className="text-xs bg-gray-700 text-gray-300 border border-gray-600 rounded px-1 py-0.5"
                    >
                        <option value="ALL">ALL</option>
                        <option value="DEBUG">DEBUG</option>
                        <option value="INFO">INFO</option>
                        <option value="WARNING">WARNING</option>
                        <option value="ERROR">ERROR</option>
                    </select> */}

                    {/* 清空 */}
                    <div
                        onClick={clearLogs}
                        className="text-xs px-1 py-0.5 rounded text-white cursor-pointer"
                    >
                        清空
                    </div>
                </div>
            </div>

            {/* 日志内容 */}
            <div
                ref={logContainerRef}
                onScroll={handleScroll}
                className="flex-1 overflow-y-auto font-mono text-xs p-2 space-y-0.5 log-scrollbar"
            >
                {logs.length === 0 ? (
                    <div className="text-gray-600 text-center py-4">
                        等待日志...
                    </div>
                ) : (
                    logs.map((log) => (
                        <div
                            key={log.id}
                            className={`${levelBgColors[log.level]} rounded px-1 py-0.5 wrap-break-word`}
                        >
                            <span className="text-gray-500">[{log.timestamp}]</span>
                            {/* <span className={`ml-1 font-semibold ${levelColors[log.level]}`}>
                                [{log.level}]
                            </span> */}
                            {/* <span className="text-gray-400 ml-1">
                                [{log.module}:{log.line}]
                            </span> */}
                            <span className="text-gray-200 ml-1 whitespace-pre-wrap">
                                {log.message}
                            </span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
