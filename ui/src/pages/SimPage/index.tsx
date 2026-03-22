import {useCallback, useEffect, useRef, useState} from "react"
import {
    socket,
    sendActions,
    resetCar,
    getCarState,
    setEpisode,
    getEpisodes,
    deleteEpisode,
    startTraining,
    stopTraining,
    loadTrainedModel,
    runInferenceWithSocket,
    startEpisode,
    endEpisode,
    finalizeEpisode,
    getEpisodeStatus,
    sendImageData,
} from "../../api/socket.ts";
import type {CarState, Obstacle} from "../../models/types.ts";
import {TopDownView} from "./TopDownView.tsx";
import {FirstPersonView, type FirstPersonViewRef} from "./FirstPersonView.tsx";
import {TrainingControl} from "./TrainingControl.tsx";
import {InferenceControl} from "./InferenceControl.tsx";

const SEND_INTERVAL = 50 // 发送控制指令间隔(ms)

const SimPage = () => {
    const keys = useRef<Record<string, boolean>>({})
    const firstPersonViewRef = useRef<FirstPersonViewRef>(null)
    const [carState, setCarState] = useState<CarState>({
        x: 400,
        y: 300,
        angle: -Math.PI / 2,
        vel_left: 0,
        vel_right: 0,
    })
    const [obstacles, setObstacles] = useState<Obstacle[]>([
        {x: 300, y: 200, width: 80, height: 80},
    ])
    const [collectedCount, setCollectedCount] = useState(0)
    const [isTraining, setIsTraining] = useState(false)
    const [trainingProgress, setTrainingProgress] = useState({epoch: 0, total_epochs: 50, loss: 0, progress: 0})
    const [trainingEpochs, setTrainingEpochs] = useState(50)
    const [currentEpisode, setCurrentEpisode] = useState(1)
    const [episodeCounts, setEpisodeCounts] = useState<Record<number, number>>({})
    const [resumeTraining, setResumeTraining] = useState(false)
    const [isModelLoaded, setIsModelLoaded] = useState(false)
    const [inferenceResult, setInferenceResult] = useState<string[]>([])
    const [isInferring, setIsInferring] = useState(false)
    const [autoInference, setAutoInference] = useState(false)
    const autoInferenceRef = useRef(false)  // ref 版本用于动画循环，避免闭包竞争
    const inferenceTimerRef = useRef<number | null>(null)

    // Episode 管理状态
    const [isRecording, setIsRecording] = useState(false)
    const [episodeTaskName, setEpisodeTaskName] = useState("default")

    // 监听后端车辆状态更新
    useEffect(() => {
        // 连接 Socket.IO
        socket.connect()

        // 监听连接
        socket.on("connected", (data) => {
            console.log("Connected:", data)
            // 连接后获取初始状态
            getCarState()
        })

        // 监听车辆状态更新
        socket.on("car_state_update", (state: CarState) => {
            setCarState(state)
        })

        // 监听采集计数更新
        socket.on("collection_count", (data: {
            count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setCollectedCount(data.count)
            if (data.exported) {
                // alert(`数据已导出到: ${data.output_path}`)
            } else if (data.error) {
                alert(`导出失败: ${data.error}`)
            }
        })

        // 监听轮次信息（只同步各轮次的数据量，不同步轮次号，由前端控制）
        socket.on("episode_info", (data: {
            current_episode: number;
            episodes: Record<number, number>;
            buffer_size?: number
        }) => {
            setEpisodeCounts(data.episodes)
        })

        // 监听 episode 状态
        socket.on("episode_status", (data: {
            episode_id: number;
            is_recording: boolean;
            frame_count: number;
            task_name: string
        }) => {
            setIsRecording(data.is_recording)
            setCollectedCount(data.frame_count)
            setEpisodeTaskName(data.task_name)
        })

        // 监听 episode 开始
        socket.on("episode_started", (data: { episode_id: number; task_name: string; frame_count: number }) => {
            setIsRecording(true)
            setCollectedCount(0)
            setEpisodeTaskName(data.task_name)
        })

        // 监听 episode 结束
        socket.on("episode_ended", (data: {
            episode_id: number;
            frame_count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setIsRecording(false)
            setCollectedCount(data.frame_count)
            if (data.exported) {
                // alert(`Episode ${data.episode_id} 数据已导出到: ${data.output_path}`)
            } else if (data.error) {
                alert(`导出失败: ${data.error}`)
            }
        })

        // 监听 episode 完成
        socket.on("episode_finalized", (data: {
            episode_id: number;
            frame_count: number;
            output_path?: string;
            error?: string
        }) => {
            if (data.output_path) {
                // alert(`Episode ${data.episode_id} 已保存到: ${data.output_path}`)
            } else if (data.error) {
                alert(`保存失败: ${data.error}`)
            }
        })

        // 监听训练进度
        socket.on("training_progress", (data: {
            is_running: boolean;
            epoch: number;
            total_epochs: number;
            loss: number;
            progress: number
        }) => {
            setIsTraining(data.is_running)
            setTrainingProgress({
                epoch: data.epoch,
                total_epochs: data.total_epochs,
                loss: data.loss,
                progress: data.progress
            })
        })

        // 获取初始轮次信息
        getEpisodes()
        // 获取初始 episode 状态
        getEpisodeStatus()

        return () => {
            socket.off("connected")
            socket.off("car_state_update")
            socket.off("collection_count")
            socket.off("training_progress")
            socket.off("episode_info")
            socket.off("episode_status")
            socket.off("episode_started")
            socket.off("episode_ended")
            socket.off("episode_finalized")
            socket.off("collection_paused")
            socket.off("collection_resumed")
            socket.disconnect()
        }
    }, [])

    const sendCommand = (cmd: string[]) => {
        // 发送动作到后端
        if (cmd.length === 0) {
            return  // 空动作不发任何命令
        }
        sendActions(cmd)
    }

    const handleSetEpisode = (episodeId: number) => {
        if (episodeId < 1) return

        // 保存当前轮次（因为 handleEndEpisode 会修改 currentEpisode）
        const episodeToDelete = currentEpisode

        if (isRecording) {
            if (!confirm('正在录制中，切换轮次将结束当前录制。是否继续?')) {
                return
            }
            handleEndEpisode()
        }

        // 重置帧数
        setCollectedCount(0)

        // 先设置目标轮次（后端会清空该轮次的数据）
        setEpisode(episodeId)

        // 如果是回退到之前的轮次，删除之前轮次的数据
        if (episodeId < episodeToDelete) {
            deleteEpisode(episodeToDelete)
            // 刷新轮次列表
            getEpisodes()
        }
        setCurrentEpisode(episodeId)
    }

    const handleStartEpisode = () => {
        // 检查是否有上一轮的数据需要保存
        if (episodeCounts[currentEpisode] && episodeCounts[currentEpisode] > 0) {
            finalizeEpisode(currentEpisode)
            // 刷新轮次列表
            getEpisodes()
        }

        // 开始新录制（使用当前轮次，不改变轮次）
        startEpisode(currentEpisode, episodeTaskName)
    }

    const handleEndEpisode = () => {
        // 结束录制并自动保存数据（endEpisode会自动导出，所以不需要再调用finalizeEpisode）
        endEpisode(currentEpisode)
        // 轮次自动+1
        setCurrentEpisode(currentEpisode + 1)
        // 重置帧数
        setCollectedCount(0)
        // 刷新轮次列表
        getEpisodes()
    }

    const handleStartTraining = async () => {
        try {
            const result = await startTraining({
                data_dir: 'dataset',
                output_dir: 'checkpoints',
                epochs: trainingEpochs,
                batch_size: 8,
                lr: 1e-4,
                resume_from: resumeTraining ? 'output/train/model.pt' : undefined,
            })
            if (!result.success) {
                alert(result.message)
            }
        } catch (e) {
            alert('启动训练失败')
        }
    }

    const handleStopTraining = async () => {
        try {
            await stopTraining()
        } catch (e) {
            alert('停止训练失败')
        }
    }

    const handleLoadModel = async () => {
        try {
            const result = await loadTrainedModel()
            if (result.success) {
                setIsModelLoaded(true)
                alert('模型加载成功')
            } else {
                alert('模型加载失败: ' + result.message)
            }
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e)
            alert('加载模型失败: ' + msg)
        }
    }

    const doInference = async () => {
        // 真实小车模式：状态输入是左右轮速度 [vel_left, vel_right]
        const state: [number, number] = [carState.vel_left, carState.vel_right]
        const imageBase64 = firstPersonViewRef.current?.getImageData()
        const result = await runInferenceWithSocket(state, imageBase64)
        if (result.success && result.action) {
            const actionChunks = Array.isArray(result.action) ? result.action : [result.action]
            if (actionChunks.length === 0 || !Array.isArray(actionChunks[0])) {
                console.error("Invalid action shape:", result.action)
                return
            }

            // 兼容 [chunk, dim] 和 [batch, chunk, dim] 两种返回格式
            const firstChunk = Array.isArray(actionChunks[0][0])
                ? actionChunks[0][0]
                : actionChunks[0]

            const velLeftTarget = firstChunk[0]
            const velRightTarget = firstChunk[1]

            if (typeof velLeftTarget !== 'number' || typeof velRightTarget !== 'number') {
                console.error("Invalid velocity values:", { velLeftTarget, velRightTarget, firstChunk })
                return
            }

            const velStr = `v=[${velLeftTarget.toFixed(2)}, ${velRightTarget.toFixed(2)}]`
            setInferenceResult([velStr])
        } else if (!result.success) {
            throw new Error(result.error || '推理失败')
        }
    }

    const handleInference = async () => {
        if (!isModelLoaded) {
            alert('请先加载模型')
            return
        }
        setIsInferring(true)
        try {
            await doInference()
        } catch (e) {
            alert('推理失败')
        }
        setIsInferring(false)
    }

    const handleAutoInference = async () => {
        if (!isModelLoaded) {
            alert('请先加载模型')
            return
        }
        if (autoInference) {
            setAutoInference(false)
            autoInferenceRef.current = false  // 同步更新 ref
            if (inferenceTimerRef.current) {
                clearInterval(inferenceTimerRef.current)
                inferenceTimerRef.current = null
            }
        } else {
            setAutoInference(true)
            autoInferenceRef.current = true  // 同步更新 ref
            await doInference()
            inferenceTimerRef.current = window.setInterval(async () => {
                await doInference()
            }, 50)
        }
    }

    // 获取当前按下的动作列表
    const getCurrentActions = useCallback((): string[] => {
        const keyMap: Record<string, string> = {
            'ArrowUp': 'forward',
            'KeyW': 'forward',
            'ArrowDown': 'backward',
            'KeyS': 'backward',
            'ArrowLeft': 'left',
            'KeyA': 'left',
            'ArrowRight': 'right',
            'KeyD': 'right',
        }

        const actions: string[] = []
        for (const [code, action] of Object.entries(keyMap)) {
            if (keys.current[code]) {
                actions.push(action)
            }
        }
        return actions
    }, [])

    // 键盘事件处理
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            const active = document.activeElement as HTMLElement | null
            if (active) {
                const tag = active.tagName
                if (active.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
                    return
                }
            }
            if (e.code.startsWith("Arrow")) {
                e.preventDefault()
            }
            keys.current[e.code] = true
        }
        const handleKeyUp = (e: KeyboardEvent) => {
            const active = document.activeElement as HTMLElement | null
            if (active) {
                const tag = active.tagName
                if (active.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
                    return
                }
            }
            if (e.code.startsWith("Arrow")) {
                e.preventDefault()
            }
            keys.current[e.code] = false
        }

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)
        }
    }, [])

    // 定时发送动作
    useEffect(() => {
        let lastSendTime = 0;
        let rafId: number

        const loop = (currentTime: number) => {
            // 自动推理模式下完全跳过，不发任何命令
            if (autoInferenceRef.current) {
                rafId = window.requestAnimationFrame(loop)
                return
            }

            if (currentTime - lastSendTime >= SEND_INTERVAL) {
                const actions = getCurrentActions()
                // 只有有实际动作时才发送，停止时什么都不发
                if (actions.length > 0) {
                    sendCommand(actions)
                }
                lastSendTime = currentTime
            }
            rafId = window.requestAnimationFrame(loop)
        }

        rafId = window.requestAnimationFrame(loop)
        return () => window.cancelAnimationFrame(rafId)
    }, [getCurrentActions])

    return (
        <div className="flex flex-col gap-3 p-4 h-screen overflow-hidden">
            <h1 className="text-center font-bold">AKA-Sim 模拟器</h1>
            <div className="flex gap-5 flex-1 items-stretch">
                <div className="w-64 flex flex-col h-full">
                    <TrainingControl
                        collectedCount={collectedCount}
                        isTraining={isTraining}
                        trainingProgress={trainingProgress}
                        trainingEpochs={trainingEpochs}
                        resumeTraining={resumeTraining}
                        episodeCounts={episodeCounts}
                        currentEpisode={currentEpisode}
                        isRecording={isRecording}
                        onStartTraining={handleStartTraining}
                        onStopTraining={handleStopTraining}
                        onSetTrainingEpochs={setTrainingEpochs}
                        onSetResumeTraining={setResumeTraining}
                        onSetEpisode={handleSetEpisode}
                        onEndEpisode={handleEndEpisode}
                        onStartEpisode={handleStartEpisode}
                        onResetCar={resetCar}
                    />

                    <InferenceControl
                        isModelLoaded={isModelLoaded}
                        isInferring={isInferring}
                        autoInference={autoInference}
                        inferenceResult={inferenceResult}
                        onLoadModel={handleLoadModel}
                        onInference={handleInference}
                        onAutoInference={handleAutoInference}
                    />
                </div>
                <div className="flex-1 flex flex-col h-full">
                    <TopDownView
                        carState={carState}
                        obstacles={obstacles}
                        onObstaclesChange={setObstacles}
                        collectedCount={collectedCount}
                        resetCar={resetCar}
                        sendCommand={sendCommand}
                    />
                </div>
                <div className="w-90 flex flex-col h-full">
                    <FirstPersonView
                        ref={firstPersonViewRef}
                        carState={carState}
                        obstacles={obstacles}
                        isRecording={isRecording}
                        onCollect={(imageData, actions) => sendImageData(imageData, actions)}
                        getCurrentActions={getCurrentActions}
                    />
                </div>
            </div>
        </div>
    )
}

export default SimPage;
