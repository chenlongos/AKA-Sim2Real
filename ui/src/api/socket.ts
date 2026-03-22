import {io} from "socket.io-client";

// 连接到后端服务器（使用相对路径，通过 Vite 代理）
export const socket = io("", {
    path: "/socket.io",
    transports: ["polling", "websocket"],
    upgrade: true,
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 500,
    timeout: 20000
});

// 发送当前按下的按键列表
export const sendActions = (actions: string[]) => {
    socket.emit('action', actions);
}

// 直接发送速度命令 [vel_left, vel_right]
export const sendVelocity = (velocity: [number, number]) => {
    socket.emit('velocity_action', velocity);
}

export const resetCar = () => {
    socket.emit('reset_car_state');
}

export const getCarState = () => {
    socket.emit('get_car_state');
}

// 发送图像数据用于训练数据采集
export const sendImageData = (imageData: string, actions: string[]) => {
    socket.emit('collect_data', {
        image: imageData,
        actions: actions
    });
}

// 设置当前采集轮次
export const setEpisode = (episodeId: number) => {
    socket.emit('set_episode', episodeId);
}

// 获取所有轮次信息
export const getEpisodes = () => {
    socket.emit('get_episodes');
}

// 删除指定轮次的数据
export const deleteEpisode = (episodeId: number) => {
    socket.emit('delete_episode', { episode_id: episodeId });
}

// 开始新的 episode
export const startEpisode = (episodeId: number, taskName: string = "default") => {
    socket.emit('start_episode', { episode_id: episodeId, task_name: taskName });
}

// 结束当前 episode
export const endEpisode = (episodeId?: number) => {
    socket.emit('end_episode', { episode_id: episodeId });
}

// 完成 episode 并保存到磁盘
export const finalizeEpisode = (episodeId?: number) => {
    socket.emit('finalize_episode', { episode_id: episodeId });
}

// 获取当前 episode 状态
export const getEpisodeStatus = () => {
    socket.emit('get_episode_status');
}

// 暂停数据采集
export const pauseCollection = () => {
    socket.emit('pause_collection');
}

// 训练相关
export const startTraining = (params: {
    data_dir?: string;
    output_dir?: string;
    epochs?: number;
    batch_size?: number;
    lr?: number;
    episode_ids?: number[];  // 指定使用哪些轮次
    resume_from?: string;   // 从已有模型继续训练
}) => {
    return fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    }).then(res => res.json());
};

export const getTrainingStatus = () => {
    return fetch('/api/train/status').then(res => res.json());
};

export const stopTraining = () => {
    return fetch('/api/train/stop', { method: 'POST' }).then(res => res.json());
};

// 推理相关
export const loadTrainedModel = () => {
    return fetch('/api/act/load_trained', { method: 'POST' }).then(res => {
        if (!res.ok) {
            return res.json().then(err => {
                throw new Error(err.detail || '加载失败')
            })
        }
        return res.json()
    })
};

export const runInference = (state: number[], image?: string) => {
    return fetch('/api/act/run_inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state, image }),
    }).then(res => res.json());
};

export const runInferenceWithSocket = (
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
export const onTrainingProgress = (callback: (data: {
    is_running: boolean;
    epoch: number;
    total_epochs: number;
    loss: number;
    progress: number;
}) => void) => {
    socket.on('training_progress', callback);
    return () => socket.off('training_progress');
};
