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
import type {CarState} from "../../models/types.ts";
import {TrainingControl} from "../SimPage/TrainingControl.tsx";
import {InferenceControl} from "../SimPage/InferenceControl.tsx";
import {RealCameraView} from "./RealCameraView.tsx";
import {CarControl} from "./CarControl.tsx";

const SEND_INTERVAL = 50 // 发送控制指令间隔(ms)

const RealPage = () => {
    const keys = useRef<Record<string, boolean>>({})
    const lastSentActionsRef = useRef<string[]>([])
    const [carState, setCarState] = useState<CarState>({
        x: 0,
        y: 0,
        angle: 0,
        vel_left: 0,
        vel_right: 0,
    })
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
    const autoInferenceRef = useRef(false)
    const inferenceTimerRef = useRef<number | null>(null)

    // Episode 管理状态
    const [isRecording, setIsRecording] = useState(false)
    const [episodeTaskName, setEpisodeTaskName] = useState("default")

    // 监听后端车辆状态更新
    useEffect(() => {
        socket.connect()

        socket.on("connected", (data) => {
            console.log("Connected:", data)
            getCarState()
        })

        socket.on("car_state_update", (state: CarState) => {
            setCarState(state)
        })

        socket.on("collection_count", (data: {
            count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setCollectedCount(data.count)
        })

        socket.on("episode_info", (data: {
            current_episode: number;
            episodes: Record<number, number>;
            buffer_size?: number
        }) => {
            setEpisodeCounts(data.episodes)
        })

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

        socket.on("episode_started", (data: { episode_id: number; task_name: string; frame_count: number }) => {
            setIsRecording(true)
            setCollectedCount(0)
            setEpisodeTaskName(data.task_name)
        })

        socket.on("episode_ended", (data: {
            episode_id: number;
            frame_count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setIsRecording(false)
            setCollectedCount(data.frame_count)
        })

        socket.on("episode_finalized", (data: {
            episode_id: number;
            frame_count: number;
            output_path?: string;
            error?: string
        }) => {
            // 数据保存完成
            if (data.output_path) {
                // alert(`Episode ${data.episode_id} 已保存到: ${data.output_path}`)
            } else if (data.error) {
                alert(`保存失败: ${data.error}`)
            }
        })

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

        getEpisodes()
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
            socket.disconnect()
        }
    }, [])

    const sendCommand = (cmd: string[]) => {
        sendActions(cmd)
    }

    const handleSetEpisode = (episodeId: number) => {
        if (episodeId < 1) return

        const episodeToDelete = currentEpisode

        if (isRecording) {
            if (!confirm('正在录制中，切换轮次将结束当前录制。是否继续?')) {
                return
            }
            handleEndEpisode()
        }

        setCollectedCount(0)
        setEpisode(episodeId)

        if (episodeId < episodeToDelete) {
            deleteEpisode(episodeToDelete)
            getEpisodes()
        }
        setCurrentEpisode(episodeId)
    }

    const handleStartEpisode = () => {
        if (episodeCounts[currentEpisode] && episodeCounts[currentEpisode] > 0) {
            finalizeEpisode(currentEpisode)
            getEpisodes()
        }
        startEpisode(currentEpisode, episodeTaskName)
    }

    const handleEndEpisode = () => {
        endEpisode(currentEpisode)
        setCurrentEpisode(currentEpisode + 1)
        setCollectedCount(0)
        getEpisodes()
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
        const state: [number, number] = [carState.vel_left, carState.vel_right]
        const result = await runInferenceWithSocket(state, undefined)
        if (result.success && result.action) {
            const actionChunks = Array.isArray(result.action) ? result.action : [result.action]
            if (actionChunks.length === 0 || !Array.isArray(actionChunks[0])) {
                console.error("Invalid action shape:", result.action)
                return
            }

            const firstChunk = Array.isArray(actionChunks[0][0])
                ? actionChunks[0][0]
                : actionChunks[0]

            const velLeftTarget = firstChunk[0]
            const velRightTarget = firstChunk[1]

            if (typeof velLeftTarget !== 'number' || typeof velRightTarget !== 'number') {
                console.error("Invalid velocity values:", {velLeftTarget, velRightTarget, firstChunk})
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
            autoInferenceRef.current = false
            if (inferenceTimerRef.current) {
                clearInterval(inferenceTimerRef.current)
                inferenceTimerRef.current = null
            }
        } else {
            setAutoInference(true)
            autoInferenceRef.current = true
            await doInference()
            inferenceTimerRef.current = window.setInterval(async () => {
                await doInference()
            }, 50)
        }
    }

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
        const clearKeysAndStop = () => {
            keys.current = {}
            lastSentActionsRef.current = []
            sendActions([])
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
            if (autoInferenceRef.current) {
                rafId = window.requestAnimationFrame(loop)
                return
            }

            if (currentTime - lastSendTime >= SEND_INTERVAL) {
                const actions = getCurrentActions()
                const lastActions = lastSentActionsRef.current
                const changed = actions.length !== lastActions.length
                    || actions.some((action, index) => action !== lastActions[index])

                if (changed || actions.length > 0) {
                    sendCommand(actions)
                    lastSentActionsRef.current = actions
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
            <h1 className="text-center font-bold">AKA-Sim Real 真实小车</h1>
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
                    <RealCameraView isRecording={isRecording} />
                </div>

                <div className="w-90 flex flex-col h-full">
                    <CarControl
                        carState={carState}
                        isRecording={isRecording}
                        getCurrentActions={getCurrentActions}
                        onCollect={(imageData, actions) => sendImageData(imageData, actions)}
                    />
                </div>
            </div>
        </div>
    )
}

export default RealPage;