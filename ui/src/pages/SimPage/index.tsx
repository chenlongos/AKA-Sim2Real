import {useCallback, useEffect, useRef, useState} from "react"
import {
    simSocket,
    sendActionVector,
    setEpisode,
    getEpisodes,
    deleteEpisode,
    runInferenceWithSocket,
    startEpisode,
    endEpisode,
    getEpisodeStatus,
    sendImageData,
    onTrainingProgress,
} from "../../api/socket.ts";
import {startTraining, stopTraining, loadTrainedModel, listDatasetDirs, listModels} from "../../api/api";
import type {Obstacle} from "../../models/types.ts";
import {TopDownView} from "./TopDownView.tsx";
import {RightPanel, type RightPanelRef} from "./RightPanel.tsx";
import {TrainingControl} from "./TrainingControl.tsx";
import {InferenceControl} from "./InferenceControl.tsx";
import {useSimCarStore} from "../../stores/simCarStore.ts";
import {getContinuousActionFromDiscreteActions, SIM_KEY_TO_ACTION} from "./actionMapping.ts";
import {showToast} from "../../lib/toast.ts";
import {SIMULATION, getDatasetPath, getTrainPath, getModelPath} from "../../lib/constants.ts";
import {init, send} from "../../services/reportService.ts";

const SimPage = () => {
    const keys = useRef<Record<string, boolean>>({})
    const lastSentActionVectorRef = useRef<[number, number, number]>([0, 0, 0])
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
    const [resumeTraining, setResumeTraining] = useState(false)
    const [isModelLoaded, setIsModelLoaded] = useState(false)
    const [inferenceResult, setInferenceResult] = useState<string[]>([])
    const [isInferring, setIsInferring] = useState(false)
    const [autoInference, setAutoInference] = useState(false)
    const autoInferenceRef = useRef(false)  // ref 版本用于动画循环，避免闭包竞争
    const inferenceTimerRef = useRef<number | null>(null)
    const resetSimCarState = useSimCarStore((state) => state.resetCarState)

    // Episode 管理状态
    const [isRecording, setIsRecording] = useState(false)
    const [episodeTaskName, setEpisodeTaskName] = useState("default")
    const [datasetName, setDatasetName] = useState("default")
    const [datasets, setDatasets] = useState<string[]>([])
    const [models, setModels] = useState<string[]>([])
    const [selectedModel, setSelectedModel] = useState<string>("")
    const userId = useSimCarStore.getState().userId
    const episodeStartTimeRef = useRef<number>(0)
    const trainingPathsRef = useRef<{ dataset_path: string; model_path: string }>({ dataset_path: "", model_path: "" })

    // 监听后端事件
    useEffect(() => {
        init("sim");
        send("action.enter");

        // 监听连接
        simSocket.on("connected", (data) => {
            console.log("Connected:", data)
        })

        // 加载用户的数据集列表
        const loadDatasets = async () => {
            try {
                const result = await listDatasetDirs(userId)
                if (result.datasets && result.datasets.length > 0) {
                    setDatasets(result.datasets)
                    if (!datasetName || !result.datasets.includes(datasetName)) {
                        setDatasetName(result.datasets[0])
                    }
                }
            } catch {
                // 忽略错误，使用默认数据集
            }
        }

        // 监听采集计数更新
        simSocket.on("collection_count", (data: {
            count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setCollectedCount(data.count)
            if (data.error) {
                showToast.error(`导出失败: ${data.error}`)
            }
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
            episodeStartTimeRef.current = Date.now()
            send("collection.episode_started", {
                episode_id: data.episode_id,
                task_name: data.task_name,
                dataset_name: datasetName,
            })
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
            const durationMs = episodeStartTimeRef.current ? Date.now() - episodeStartTimeRef.current : 0
            send("collection.episode_ended", {
                episode_id: data.episode_id,
                frame_count: data.frame_count,
                duration_ms: durationMs,
            })
            if (!data.error) {
                loadDatasets()
            } else {
                showToast.error(`导出失败: ${data.error}`)
            }
        })

        // 监听 episode 完成
        simSocket.on("episode_finalized", (data: {
            episode_id: number;
            frame_count: number;
            output_path?: string;
            error?: string
        }) => {
            send("collection.episode_finalized", {
                episode_id: data.episode_id,
                frame_count: data.frame_count,
                output_path: data.output_path || "",
            })
            if (data.error) {
                showToast.error(`保存失败: ${data.error}`)
            } else {
                loadDatasets()
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

            if (data.is_running && data.progress < 1) {
                send("training.epoch_progress", {
                    epoch: data.epoch,
                    total_epochs: data.total_epochs,
                    loss: data.loss,
                    progress: data.progress,
                })
            } else if (!data.is_running && data.progress >= 1) {
                send("training.completed", {
                    total_epochs: data.total_epochs,
                    final_loss: data.loss,
                    dataset_path: trainingPathsRef.current.dataset_path,
                    model_path: trainingPathsRef.current.model_path,
                })
            }
        })

        // 获取初始轮次信息
        getEpisodes(simSocket, userId)
        // 获取初始 episode 状态
        getEpisodeStatus(simSocket, userId)
        loadDatasets()

        return () => {
            simSocket.off("connected")
            simSocket.off("collection_count")
            simSocket.off("training_progress")
            simSocket.off("episode_started")
            simSocket.off("episode_ended")
            simSocket.off("episode_finalized")
            simSocket.off("collection_paused")
            simSocket.off("collection_resumed")
            unsubscribeTrainingProgress()
            resetSimCarState()
            send("action.leave");
        }
    }, [resetSimCarState, userId])

    // 加载模型列表
    useEffect(() => {
        setIsModelLoaded(false)
        setSelectedModel("")
        setIsModelLoaded(false)
        const loadModels = async () => {
            try {
                const result = await listModels(userId, datasetName)
                if (result.models) {
                    setModels(result.models)
                } else {
                    setModels([])
                }
            } catch {
                setModels([])
            }
        }
        loadModels()
    }, [userId, datasetName])

    const handleRefreshModels = async () => {
        try {
            const result = await listModels(userId, datasetName)
            if (result.models) {
                setModels(result.models)
            }
        } catch {
            // ignore
        }
    }

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

        send("action.switch_episode", {
            from_episode: episodeToDelete,
            to_episode: episodeId,
        })

        // 先设置目标轮次（后端会清空该轮次的数据）
        setEpisode(simSocket, userId, episodeId)

        // 如果是回退到之前的轮次，删除之前轮次的数据
        if (episodeId < episodeToDelete) {
            deleteEpisode(simSocket, userId, episodeToDelete)
            // 刷新轮次列表
            getEpisodes(simSocket, userId)
        }
        setCurrentEpisode(episodeId)
    }

    const handleStartEpisode = () => {
        send("action.start_collection", {
            episode_id: currentEpisode,
            dataset_name: datasetName,
            fps: collectionFps,
        })
        // 开始新录制（使用当前轮次，不改变轮次）
        startEpisode(simSocket, userId, currentEpisode, episodeTaskName)
    }

    const handleSelectDataset = (name: string) => {
        setDatasetName(name)
        // 切换数据集后重新加载该数据集的 episode 信息
        getEpisodes(simSocket, userId)
    }

    const handleEndEpisode = () => {
        send("action.end_collection", {
            episode_id: currentEpisode,
            frame_count: collectedCount,
        })
        // 结束录制并自动保存数据（endEpisode会自动导出，所以不需要再调用finalizeEpisode）
        endEpisode(simSocket, userId, currentEpisode)
        // 轮次自动+1
        setCurrentEpisode(currentEpisode + 1)
        // 重置帧数
        setCollectedCount(0)
        // 刷新轮次列表
        getEpisodes(simSocket, userId)
    }

    const handleStartTraining = async () => {
        try {
            const userId = useSimCarStore.getState().userId
            const datasetPath = getDatasetPath(userId, datasetName)
            const modelPath = getModelPath(userId, datasetName)
            trainingPathsRef.current = { dataset_path: datasetPath, model_path: modelPath }

            send("action.start_training", {
                dataset_name: datasetName,
                epochs: trainingEpochs,
                batch_size: 8,
                lr: 1e-4,
                resume: resumeTraining,
            })

            const result = await startTraining(userId, {
                data_dir: datasetPath,
                output_dir: getTrainPath(userId, datasetName),
                epochs: trainingEpochs,
                batch_size: 8,
                lr: 1e-4,
                resume_from: resumeTraining ? modelPath : undefined,
            })
            if (!result.success) {
                showToast.error(result.message || '训练失败')
            }
        } catch {
            showToast.error('启动训练失败')
        }
    }

    const handleStopTraining = async () => {
        try {
            send("action.stop_training", {
                current_epoch: trainingProgress.epoch,
                current_loss: trainingProgress.loss,
            })
            send("training.stopped", {
                ended_epoch: trainingProgress.epoch,
                last_loss: trainingProgress.loss,
            })
            await stopTraining(userId)
        } catch {
            showToast.error('停止训练失败')
        }
    }

    const handleLoadModel = async (modelName: string) => {
        try {
            const userId = useSimCarStore.getState().userId
            const dataDir = getDatasetPath(userId, datasetName)
            const modelPath = `output/train/${userId}/${modelName}/model.pt`
            const result = await loadTrainedModel(userId, dataDir, modelPath)
            if (result.success) {
                setIsModelLoaded(true)
                setSelectedModel(modelName)
                send("action.load_model", { model_name: modelName })
                showToast.success('模型加载成功')
            } else {
                showToast.error('模型加载失败: ' + result.message)
            }
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e)
            showToast.error('加载模型失败: ' + msg)
        }
    }

    const handleUnloadModel = () => {
        setIsModelLoaded(false)
        setSelectedModel("")
    }

    const doInference = useCallback(async () => {
        // 获取最新的 carState（避免闭包问题）
        const currentCarState = useSimCarStore.getState().carState
        // 真实小车模式：状态输入是左右轮速度 [vel_left, vel_right]
        const state: [number, number] = [currentCarState.vel_left, currentCarState.vel_right]
        const imageBase64 = firstPersonViewRef.current?.getImageData()
        const result = await runInferenceWithSocket(simSocket, state, imageBase64, userId)
        if (result.success && result.action) {
            // 新格式：action = [left_vel, right_vel]（单步）
            const actionValues = Array.isArray(result.action) ? result.action : [result.action]
            if (actionValues.length < 2) {
                console.error("Invalid action shape:", result.action)
                return
            }

            const velLeftTarget = actionValues[0]
            const velRightTarget = actionValues[1]

            if (typeof velLeftTarget !== 'number' || typeof velRightTarget !== 'number') {
                console.error("Invalid velocity values:", { velLeftTarget, velRightTarget })
                return
            }

            // 设置目标速度，RAF 循环会持续应用物理
            useSimCarStore.getState().setTargetVelocity(velLeftTarget, velRightTarget)

            const gripperTarget = actionValues.length >= 3 ? actionValues[2] : 0
            const gripperCmd = gripperTarget > 0.5 ? 'grab' : 'release'
            const velStr = `v=[${velLeftTarget.toFixed(2)}, ${velRightTarget.toFixed(2)}] gripper=${gripperCmd}`
            setInferenceResult([velStr])
        } else if (!result.success) {
            throw new Error(result.error || '推理失败')
        }
    }, [])

    const handleInference = async () => {
        if (!isModelLoaded) {
            showToast.warning('请先加载模型')
            return
        }
        setIsInferring(true)
        try {
            await doInference()
        } catch {
            showToast.error('推理失败')
        }
        setIsInferring(false)
    }

    const handleAutoInference = async () => {
        if (!isModelLoaded) {
            showToast.warning('请先加载模型')
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

    const getCurrentActionVector = useCallback((): [number, number, number] => {
        return getContinuousActionFromDiscreteActions(getCurrentActions())
    }, [getCurrentActions])

    // 键盘事件处理
    useEffect(() => {
        const clearKeysAndStop = () => {
            keys.current = {}
            lastSentActionVectorRef.current = [0, 0, 0]
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

    // 定时发送动作并更新本地状态
    useEffect(() => {
        let lastSendTime = 0;
        let rafId: number

        const loop = (currentTime: number) => {
            // 每帧应用物理（通过 store 直接获取最新状态）
            const store = useSimCarStore.getState()
            store.applyPhysics()

            // 自动推理模式下只发推理请求
            if (autoInferenceRef.current) {
                rafId = window.requestAnimationFrame(loop)
                return
            }

            if (currentTime - lastSendTime >= SIMULATION.SEND_INTERVAL_MS) {
                const actionVector = getCurrentActionVector()
                const lastActionVector = lastSentActionVectorRef.current
                const changed = actionVector[0] !== lastActionVector[0]
                    || actionVector[1] !== lastActionVector[1]

                if (changed || actionVector[0] !== 0 || actionVector[1] !== 0) {
                    // 更新目标速度
                    store.setTargetVelocity(actionVector[0], actionVector[1])
                    // 发送命令到后端（用于日志和推理）
                    sendCommand([actionVector[0], actionVector[1]])
                    lastSentActionVectorRef.current = actionVector
                }
                lastSendTime = currentTime
            }
            rafId = window.requestAnimationFrame(loop)
        }

        rafId = window.requestAnimationFrame(loop)
        return () => window.cancelAnimationFrame(rafId)
    }, [getCurrentActionVector, getCurrentActions, sendCommand])

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
                        currentEpisode={currentEpisode}
                        isRecording={isRecording}
                        datasetName={datasetName}
                        datasets={datasets}
                        onStartTraining={handleStartTraining}
                        onStopTraining={handleStopTraining}
                        onSetTrainingEpochs={setTrainingEpochs}
                        onSetCollectionFps={(fps) => setCollectionFps(Math.max(1, Math.min(60, fps)))}
                        onSetResumeTraining={setResumeTraining}
                        onSetEpisode={handleSetEpisode}
                        onEndEpisode={handleEndEpisode}
                        onStartEpisode={handleStartEpisode}
                        onResetCar={() => { send("action.reset_scene"); resetSimCarState(); sendCommand([0, 0]); }}
                        onSetDatasetName={setDatasetName}
                        onSelectDataset={handleSelectDataset}
                    />

                    <InferenceControl
                        isModelLoaded={isModelLoaded}
                        selectedModel={selectedModel}
                        isInferring={isInferring}
                        autoInference={autoInference}
                        inferenceResult={inferenceResult}
                        models={models}
                        onLoadModel={handleLoadModel}
                        onUnloadModel={handleUnloadModel}
                        onInference={handleInference}
                        onAutoInference={handleAutoInference}
                        onRefreshModels={handleRefreshModels}
                    />
                </div>

                {/* 中间 - 俯视图 */}
                <div className="flex-1 min-w-0 flex flex-col">
                    <TopDownView
                        obstacles={obstacles}
                        onObstaclesChange={setObstacles}
                        collectedCount={collectedCount}
                        sendCommand={sendCommand}
                    />
                </div>

                {/* 右侧 - 第一视角 + 日志 */}
                <div className="w-96 flex flex-col min-w-0">
                    <RightPanel
                        ref={firstPersonViewRef}
                        userId={userId}
                        obstacles={obstacles}
                        isRecording={isRecording}
                        collectionFps={collectionFps}
                        onCollect={(imageData) => sendImageData(simSocket, imageData, useSimCarStore.getState().userId, datasetName, {
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
