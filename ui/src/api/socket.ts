import {io, Socket} from "socket.io-client";

// 生成随机clientId
function generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Socket 工厂函数 - 为不同命名空间创建独立的 socket 实例
export function createSocket(namespace: string = "/"): Socket {
    const clientId = generateClientId();
    return io(namespace, {
        path: "/socket.io",
        transports: ["polling", "websocket"],
        upgrade: true,
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 500,
        timeout: 20000,
        auth: {
            clientId: clientId
        }
    });
}

// 创建 Sim 和 Real 页面专用的 socket 实例
export const simSocket = createSocket("/sim");
export const realSocket = createSocket("/real");

// 为了向后兼容，保留默认的全局 socket（使用 /sim 命名空间）
export const socket = simSocket;
export const clientId = generateClientId();

// ============ 通用 Socket 函数（接受 socket 参数） ============

// 发送当前按下的按键列表
export const sendActions = (socket: Socket, actions: string[]) => {
    socket.emit('action', actions);
}

export const resetCar = (socket: Socket) => {
    socket.emit('reset_car_state');
}

export const getCarState = (socket: Socket) => {
    socket.emit('get_car_state');
}

// 发送图像数据用于训练数据采集（Socket方式 - Sim页面使用）
export const sendImageData = (
    socket: Socket,
    imageData: string,
    actions: string[],
    options?: { carIP?: string; timestamp?: number },
) => {
    socket.emit('collect_data', {
        image: imageData,
        actions: actions,
        car_ip: options?.carIP,
        timestamp: options?.timestamp ?? Date.now(),
    });
}

// 设置当前采集轮次
export const setEpisode = (socket: Socket, episodeId: number) => {
    socket.emit('set_episode', episodeId);
}

// 获取所有轮次信息
export const getEpisodes = (socket: Socket) => {
    socket.emit('get_episodes');
}

// 删除指定轮次的数据
export const deleteEpisode = (socket: Socket, episodeId: number) => {
    socket.emit('delete_episode', { episode_id: episodeId });
}

// 开始新的 episode
export const startEpisode = (socket: Socket, episodeId: number, taskName: string = "default") => {
    socket.emit('start_episode', { episode_id: episodeId, task_name: taskName });
}

// 结束当前 episode
export const endEpisode = (socket: Socket, episodeId?: number) => {
    socket.emit('end_episode', { episode_id: episodeId });
}

// 完成 episode 并保存到磁盘
export const finalizeEpisode = (socket: Socket, episodeId?: number) => {
    socket.emit('finalize_episode', { episode_id: episodeId });
}

// 获取当前 episode 状态
export const getEpisodeStatus = (socket: Socket) => {
    socket.emit('get_episode_status');
}

export const runInferenceWithSocket = (
    socket: Socket,
    state: number[],
    image?: string,
    timeoutMs: number = 10000,
) => {
    return new Promise<{ success: boolean; action?: unknown; error?: string }>((resolve, reject) => {
        const timer = window.setTimeout(() => {
            socket.off('act_infer_result', handleResult);
            reject(new Error('推理超时'));
        }, timeoutMs);

        const handleResult = (data: { success: boolean; action?: unknown; error?: string }) => {
            window.clearTimeout(timer);
            socket.off('act_infer_result', handleResult);
            resolve(data);
        };

        socket.on('act_infer_result', handleResult);
        socket.emit('act_infer', { state, image });
    });
};

// 监听训练进度
export const onTrainingProgress = (socket: Socket, callback: (data: {
    is_running: boolean;
    epoch: number;
    total_epochs: number;
    loss: number;
    progress: number;
}) => void) => {
    socket.on('training_progress', callback);
    return () => socket.off('training_progress', callback);
};