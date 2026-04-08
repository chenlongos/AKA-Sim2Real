import {useCallback, useEffect, useRef, useState} from "react"
import {
    simSocket,
    sendActionVector,
    resetCar,
    getCarState,
    setEpisode,
    getEpisodes,
    deleteEpisode,
    runInferenceWithSocket,
    startEpisode,
    endEpisode,
    finalizeEpisode,
    getEpisodeStatus,
    sendImageData,
    onTrainingProgress,
} from "../../api/socket.ts";
import {startTraining, stopTraining, loadTrainedModel} from "../../api/api";
import type {CarState, Obstacle} from "../../models/types.ts";
import {TopDownView} from "./TopDownView.tsx";
import {RightPanel, type RightPanelRef} from "./RightPanel.tsx";
import {TrainingControl} from "./TrainingControl.tsx";
import {InferenceControl} from "./InferenceControl.tsx";
import {useSimCarStore} from "../../stores/simCarStore.ts";
import {getContinuousActionFromDiscreteActions, SIM_KEY_TO_ACTION} from "./actionMapping.ts";

const SEND_INTERVAL = 50 // 发送控制指令间隔(ms)

const SimPage = () => {
    const keys = useRef<Record<string, boolean>>({})
    const lastSentActionVectorRef = useRef<[number, number]>([0, 0])
    const firstPersonViewRef = useRef<RightPanelRef>(null)
    const carState = useSimCarStore((state) => state.carState)
    const [obstacles, setObstacles] = useState<Obstacle[]>([
        {x: 300, y: 200, width: 80, height: 80},
    ])
    const [collectedCount, setCollectedCount] = useState(0)
    const [isTraining, setIsTraining] = useState(false)
    const [trainingProgress, setTrainingProgress] = useState({epoch: 0, total_epochs: 50, loss: 0, progress: 0})
    const [trainingEpochs, setTrainingEpochs] = useState(50)
    const [collectionFps, setCollectionFps] = useState(30)
    const [currentEpisode, setCurrentEpisode] = useState(1)
    const [episodeCounts, setEpisodeCounts] = useState<Record<number, number>>({})
    const [resumeTraining, setResumeTraining] = useState(false)
    const [isModelLoaded, setIsModelLoaded] = useState(false)
    const [inferenceResult, setInferenceResult] = useState<string[]>([])
    const [isInferring, setIsInferring] = useState(false)
    const [autoInference, setAutoInference] = useState(false)
    const autoInferenceRef = useRef(false)  // ref 版本用于动画循环，避免闭包竞争
    const inferenceTimerRef = useRef<number | null>(null)
    const setCarState = useSimCarStore((state) => state.setCarState)
    const resetSimCarState = useSimCarStore((state) => state.resetCarState)

    // Episode 管理状态
    const [isRecording, setIsRecording] = useState(false)
    const [episodeTaskName, setEpisodeTaskName] = useState("default")

    // 监听后端车辆状态更新
    useEffect(() => {
        // 监听连接
        simSocket.on("connected", (data) => {
            console.log("Connected:", data)
            // 连接后获取初始状态
            getCarState(simSocket)
        })

        // 监听车辆状态更新
        simSocket.on("car_state_update", (state: CarState) => {
            setCarState(state)
        })

        // 监听采集计数更新
        simSocket.on("collection_count", (data: {
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
        simSocket.on("episode_info", (data: {
            current_episode: number;
            episodes: Record<number, number>;
            buffer_size?: number
        }) => {
            setEpisodeCounts(data.episodes)
        })

        // 监听 episode 状态
        simSocket.on("episode_status", (data: {
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
        simSocket.on("episode_started", (data: { episode_id: number; task_name: string; frame_count: number }) => {
            setIsRecording(true)
            setCollectedCount(0)
            setEpisodeTaskName(data.task_name)
        })

        // 监听 episode 结束
        simSocket.on("episode_ended", (data: {
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
        simSocket.on("episode_finalized", (data: {
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
        const unsubscribeTrainingProgress = onTrainingProgress(simSocket, (data: {
            is_running: boolean;
            epoch: number;
            total_epochs: number;
            loss: number;
            progress: number;
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
        getEpisodes(simSocket)
        // 获取初始 episode 状态
        getEpisodeStatus(simSocket)

        return () => {
            simSocket.off("connected")
            simSocket.off("car_state_update")
            simSocket.off("collection_count")
            simSocket.off("training_progress")
            simSocket.off("episode_info")
            simSocket.off("episode_status")
            simSocket.off("episode_started")
            simSocket.off("episode_ended")
            simSocket.off("episode_finalized")
            simSocket.off("collection_paused")
            simSocket.off("collection_resumed")
            unsubscribeTrainingProgress()
            resetSimCarState()
        }
    }, [resetSimCarState, setCarState])

    const sendCommand = (action: [number, number]) => {
        sendActionVector(simSocket, action)
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
        setEpisode(simSocket, episodeId)

        // 如果是回退到之前的轮次，删除之前轮次的数据
        if (episodeId < episodeToDelete) {
            deleteEpisode(simSocket, episodeToDelete)
            // 刷新轮次列表
            getEpisodes(simSocket)
        }
        setCurrentEpisode(episodeId)
    }

    const handleStartEpisode = () => {
        // 检查是否有上一轮的数据需要保存
        if (episodeCounts[currentEpisode] && episodeCounts[currentEpisode] > 0) {
            finalizeEpisode(simSocket, currentEpisode)
            // 刷新轮次列表
            getEpisodes(simSocket)
        }

        // 开始新录制（使用当前轮次，不改变轮次）
        startEpisode(simSocket, currentEpisode, episodeTaskName)
    }

    const handleEndEpisode = () => {
        // 结束录制并自动保存数据（endEpisode会自动导出，所以不需要再调用finalizeEpisode）
        endEpisode(simSocket, currentEpisode)
        // 轮次自动+1
        setCurrentEpisode(currentEpisode + 1)
        // 重置帧数
        setCollectedCount(0)
        // 刷新轮次列表
        getEpisodes(simSocket)
    }

    const handleStartTraining = async () => {
        try {
            const result = await startTraining({
                data_dir: 'output/dataset',
                output_dir: 'output/train',
                epochs: trainingEpochs,
                batch_size: 8,
                lr: 1e-4,
                resume_from: resumeTraining ? 'output/train/model.pt' : undefined,
            })
            if (!result.success) {
                alert(result.message)
            }
        } catch {
            alert('启动训练失败')
        }
    }

    const handleStopTraining = async () => {
        try {
            await stopTraining()
        } catch {
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
        const result = await runInferenceWithSocket(simSocket, state, imageBase64)
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
        } catch {
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

    const getCurrentActions = useCallback((): string[] => {
        const actions: string[] = []
        for (const [code, action] of Object.entries(SIM_KEY_TO_ACTION)) {
            if (keys.current[code]) {
                actions.push(action)
            }
        }
        return actions
    }, [])

    const getCurrentActionVector = useCallback((): [number, number] => {
        return getContinuousActionFromDiscreteActions(getCurrentActions())
    }, [getCurrentActions])

    // 键盘事件处理
    useEffect(() => {
        const clearKeysAndStop = () => {
            keys.current = {}
            lastSentActionVectorRef.current = [0, 0]
            sendActionVector(simSocket, [0, 0])
        }

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
        const handleWindowBlur = () => {
            clearKeysAndStop()
        }
        const handleVisibilityChange = () => {
            if (document.hidden) {
                clearKeysAndStop()
            }
        }

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)
        window.addEventListener('blur', handleWindowBlur)
        document.addEventListener('visibilitychange', handleVisibilityChange)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)
            window.removeEventListener('blur', handleWindowBlur)
            document.removeEventListener('visibilitychange', handleVisibilityChange)
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
                const actionVector = getCurrentActionVector()
                const lastActionVector = lastSentActionVectorRef.current
                const changed = actionVector[0] !== lastActionVector[0]
                    || actionVector[1] !== lastActionVector[1]

                if (changed || actionVector[0] !== 0 || actionVector[1] !== 0) {
                    sendCommand(actionVector)
                    lastSentActionVectorRef.current = actionVector
                }
                lastSendTime = currentTime
            }
            rafId = window.requestAnimationFrame(loop)
        }

        rafId = window.requestAnimationFrame(loop)
        return () => window.cancelAnimationFrame(rafId)
    }, [getCurrentActions])

    return (
        <div className="flex flex-col h-screen bg-slate-950 overflow-hidden">
            {/* 顶部标题栏 */}
            <div className="flex items-center justify-between px-6 py-2 bg-slate-900/50 border-b border-slate-800">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-linear-to-br from-violet-600 to-blue-600 flex items-center justify-center shadow-lg shadow-violet-900/20">
                        <span className="text-white font-bold text-sm">SIM</span>
                    </div>
                    <div>
                        <h2 className="text-base font-bold text-slate-100">AKA ACT 小车模拟器</h2>
                        <p className="text-xs text-slate-500">Action Chunking with Transformers - Educational Edition</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700">
                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"/>
                        <span className="text-xs text-slate-300">Simulation Active</span>
                    </div>
                    <span className="text-xs text-slate-500 font-mono">60 FPS</span>
                </div>
            </div>

            {/* 主内容区 */}
            <div className="flex flex-1 overflow-hidden p-4 gap-4">
                {/* 左侧面板 - 训练和推理控制 */}
                <div className="w-72 flex flex-col gap-3 min-w-0">
                    <TrainingControl
                        collectedCount={collectedCount}
                        isTraining={isTraining}
                        trainingProgress={trainingProgress}
                        trainingEpochs={trainingEpochs}
                        collectionFps={collectionFps}
                        resumeTraining={resumeTraining}
                        episodeCounts={episodeCounts}
                        currentEpisode={currentEpisode}
                        isRecording={isRecording}
                        onStartTraining={handleStartTraining}
                        onStopTraining={handleStopTraining}
                        onSetTrainingEpochs={setTrainingEpochs}
                        onSetCollectionFps={(fps) => setCollectionFps(Math.max(1, Math.min(60, fps)))}
                        onSetResumeTraining={setResumeTraining}
                        onSetEpisode={handleSetEpisode}
                        onEndEpisode={handleEndEpisode}
                        onStartEpisode={handleStartEpisode}
                        onResetCar={() => resetCar(simSocket)}
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

                {/* 中间 - 俯视图 */}
                <div className="flex-1 min-w-0 flex flex-col">
                    <TopDownView
                        obstacles={obstacles}
                        onObstaclesChange={setObstacles}
                        collectedCount={collectedCount}
                        resetCar={() => resetCar(simSocket)}
                        sendCommand={sendCommand}
                    />
                </div>

                {/* 右侧 - 第一视角 + 日志 */}
                <div className="w-96 flex flex-col min-w-0">
                    <RightPanel
                        ref={firstPersonViewRef}
                        obstacles={obstacles}
                        isRecording={isRecording}
                        collectionFps={collectionFps}
                        onCollect={(imageData) => sendImageData(simSocket, imageData, {
                            state: {
                                vel_left: carState.vel_left,
                                vel_right: carState.vel_right,
                            },
                            action: getCurrentActionVector(),
                        })}
                    />
                </div>
            </div>
        </div>
    )
}

export default SimPage;
