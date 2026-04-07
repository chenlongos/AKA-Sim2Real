import {useCallback, useEffect, useRef, useState} from "react"
import {
    realSocket,
    sendActions,
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
    onTrainingProgress,
} from "../../api/socket.ts";
import {carHeartbeat, motorStatus, motorDirect, carControl, startTraining, stopTraining, loadTrainedModel, collectImage} from "../../api/api";
import type {CarState} from "../../models/types.ts";
import {TrainingControl} from "../SimPage/TrainingControl.tsx";
import {InferenceControl} from "../SimPage/InferenceControl.tsx";
import {RealCameraView, type CameraDeviceOption, type RealCameraViewRef} from "./RealCameraView.tsx";
import {RealRightPanel, type RealRightPanelRef} from "./RealRightPanel.tsx";

const SEND_INTERVAL = 50 // 发送控制指令间隔(ms)
const COLLECT_INTERVAL = 1000 / 20 // 数据采集间隔(ms)，30fps

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
    const autoInferenceSessionRef = useRef(0)
    const inferenceInFlightRef = useRef(false)
    const inferenceTimerRef = useRef<number | null>(null)
    const topdownCameraViewRef = useRef<RealCameraViewRef | null>(null)
    const fpvCameraViewRef = useRef<RealRightPanelRef | null>(null)
    const collectTimerRef = useRef<number | null>(null)
    const collectInFlightRef = useRef(false)
    const [cameraDevices, setCameraDevices] = useState<CameraDeviceOption[]>([])
    const [topdownCameraId, setTopdownCameraId] = useState("")
    const [fpvCameraId, setFpvCameraId] = useState("")
    const [cameraPermissionError, setCameraPermissionError] = useState("")

    // Episode 管理状态
    const [isRecording, setIsRecording] = useState(false)
    const [episodeTaskName, setEpisodeTaskName] = useState("default")
    const [carIP, setCarIP] = useState("")
    const [carConnected, setCarConnected] = useState(false)

    // 监听后端车辆状态更新
    useEffect(() => {
        realSocket.on("connected", (data) => {
            console.log("Connected:", data)
            getCarState(realSocket)
        })

        realSocket.on("car_state_update", (state: CarState) => {
            setCarState(state)
        })

        realSocket.on("collection_count", (data: {
            count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setCollectedCount(data.count)
        })

        realSocket.on("episode_info", (data: {
            current_episode: number;
            episodes: Record<number, number>;
            buffer_size?: number
        }) => {
            setEpisodeCounts(data.episodes)
        })

        realSocket.on("episode_status", (data: {
            episode_id: number;
            is_recording: boolean;
            frame_count: number;
            task_name: string
        }) => {
            setIsRecording(data.is_recording)
            setCollectedCount(data.frame_count)
            setEpisodeTaskName(data.task_name)
        })

        realSocket.on("episode_started", (data: { episode_id: number; task_name: string; frame_count: number }) => {
            setIsRecording(true)
            setCollectedCount(0)
            setEpisodeTaskName(data.task_name)
        })

        realSocket.on("episode_ended", (data: {
            episode_id: number;
            frame_count: number;
            exported?: boolean;
            output_path?: string;
            error?: string
        }) => {
            setIsRecording(false)
            setCollectedCount(data.frame_count)
        })

        realSocket.on("episode_finalized", (data: {
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
        const unsubscribeTrainingProgress = onTrainingProgress(realSocket, (data: {
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

        getEpisodes(realSocket)
        getEpisodeStatus(realSocket)

        return () => {
            realSocket.off("connected")
            realSocket.off("car_state_update")
            realSocket.off("collection_count")
            realSocket.off("training_progress")
            realSocket.off("episode_info")
            realSocket.off("episode_status")
            realSocket.off("episode_started")
            realSocket.off("episode_ended")
            realSocket.off("episode_finalized")
            unsubscribeTrainingProgress()
        }
    }, [])

    useEffect(() => {
        if (!navigator.mediaDevices?.enumerateDevices || !navigator.mediaDevices?.getUserMedia) {
            // eslint-disable-next-line react-hooks/set-state-in-effect
            setCameraPermissionError("当前浏览器不支持摄像头访问")
            return
        }

        let cancelled = false

        const syncCameraDevices = async () => {
            try {
                const tempStream = await navigator.mediaDevices.getUserMedia({video: true})
                tempStream.getTracks().forEach((track) => track.stop())

                const devices = await navigator.mediaDevices.enumerateDevices()
                if (cancelled) return

                const videoInputs = devices
                    .filter((device) => device.kind === "videoinput")
                    .map((device, index) => ({
                        deviceId: device.deviceId,
                        label: device.label || `摄像头 ${index + 1}`,
                    }))

                setCameraDevices(videoInputs)
                setCameraPermissionError(videoInputs.length === 0 ? "未检测到可用摄像头" : "")

                setTopdownCameraId((current) => {
                    if (videoInputs.some((device) => device.deviceId === current)) {
                        return current
                    }
                    return videoInputs[0]?.deviceId ?? ""
                })
                setFpvCameraId((current) => {
                    if (videoInputs.some((device) => device.deviceId === current)) {
                        return current
                    }
                    return videoInputs[1]?.deviceId ?? videoInputs[0]?.deviceId ?? ""
                })
            } catch (error) {
                if (cancelled) return
                const message = error instanceof Error ? error.message : "摄像头权限获取失败"
                setCameraPermissionError(message)
                setCameraDevices([])
                setTopdownCameraId("")
                setFpvCameraId("")
            }
        }

        syncCameraDevices()

        const handleDeviceChange = () => {
            syncCameraDevices()
        }
        navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange)

        return () => {
            cancelled = true
            navigator.mediaDevices.removeEventListener("devicechange", handleDeviceChange)
        }
    }, [])

    const sendCommand = (cmd: string[]) => {
        sendActions(realSocket, cmd)
    }

    const heartbeatIPRef = useRef("")
    const checkCarHeartbeat = async (ip: string) => {
        if (!ip) {
            setCarConnected(false)
            heartbeatIPRef.current = ""
            return
        }
        heartbeatIPRef.current = ip
        const data = await carHeartbeat(ip)
        if (data.ok && heartbeatIPRef.current === ip) {
            setCarConnected(true)
            return
        }
    }

    const handleCarIPChange = (ip: string) => {
        setCarIP(ip)
        setCarConnected(false)
        checkCarHeartbeat(ip)
    }

    const getRealtimeInferenceState = async (): Promise<[number, number]> => {
        if (!carIP) {
            throw new Error("请先输入小车IP")
        }

        const data = await motorStatus(carIP)

        if (!data?.ok) {
            setCarConnected(false)
            throw new Error(data?.error || data?.detail || data?.message || "获取小车实时状态失败")
        }

        const velLeft = data?.state?.vel_left
        const velRight = data?.state?.vel_right
        if (typeof velLeft !== "number" || typeof velRight !== "number") {
            throw new Error("小车实时状态返回格式不正确")
        }

        setCarConnected(true)
        setCarState((prev) => ({
            ...prev,
            vel_left: velLeft,
            vel_right: velRight,
        }))

        return [velLeft, velRight]
    }

    const sendInferenceActionToCar = async (left: number, right: number) => {
        if (!carIP) {
            throw new Error("请先输入小车IP")
        }

        const mapVelocityToMotorCommand = (value: number) => {
            if (Math.abs(value) < 1e-3) {
                return 0
            }
            const sign = value >= 0 ? 1 : -1
            return sign * Math.round(Math.abs(value) * 250)
        }

        const leftCommand = mapVelocityToMotorCommand(left)
        const rightCommand = mapVelocityToMotorCommand(right)
        const data = await motorDirect(carIP, leftCommand, rightCommand)

        if (!data?.ok) {
            throw new Error(data?.error || data?.detail || data?.message || "发送推理控制到小车失败")
        }

        return {
            leftCommand,
            rightCommand,
        }
    }

    const stopInferenceAndCar = useCallback(() => {
        setAutoInference(false)
        autoInferenceRef.current = false
        autoInferenceSessionRef.current += 1
        inferenceInFlightRef.current = false

        if (inferenceTimerRef.current) {
            clearInterval(inferenceTimerRef.current)
            inferenceTimerRef.current = null
        }

        if (carIP) {
            motorDirect(carIP, 0, 0).catch(() => {})
        }
    }, [carIP])

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
        setEpisode(realSocket, episodeId)

        if (episodeId < episodeToDelete) {
            deleteEpisode(realSocket, episodeToDelete)
            getEpisodes(realSocket)
        }
        setCurrentEpisode(episodeId)
    }

    const handleStartEpisode = () => {
        if (episodeCounts[currentEpisode] && episodeCounts[currentEpisode] > 0) {
            finalizeEpisode(realSocket, currentEpisode)
            getEpisodes(realSocket)
        }
        startEpisode(realSocket, currentEpisode, episodeTaskName)
    }

    const handleEndEpisode = () => {
        endEpisode(realSocket, currentEpisode)
        setCurrentEpisode(currentEpisode + 1)
        setCollectedCount(0)
        getEpisodes(realSocket)
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

    const doInference = async (sessionId?: number) => {
        if (sessionId !== undefined && sessionId !== autoInferenceSessionRef.current) {
            return
        }

        const state = await getRealtimeInferenceState()
        const imageBase64 = fpvCameraViewRef.current?.getImageData()
        const result = await runInferenceWithSocket(realSocket, state, imageBase64)

        if (sessionId !== undefined && (!autoInferenceRef.current || sessionId !== autoInferenceSessionRef.current)) {
            return
        }

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

            if (sessionId !== undefined && (!autoInferenceRef.current || sessionId !== autoInferenceSessionRef.current)) {
                return
            }

            const {leftCommand, rightCommand} = await sendInferenceActionToCar(velLeftTarget, velRightTarget)
            const velStr = `v=[${velLeftTarget.toFixed(2)}, ${velRightTarget.toFixed(2)}] -> motor=[${leftCommand}, ${rightCommand}]`
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
            stopInferenceAndCar()
        } else {
            const sessionId = autoInferenceSessionRef.current + 1
            autoInferenceSessionRef.current = sessionId
            setAutoInference(true)
            autoInferenceRef.current = true
            inferenceInFlightRef.current = true
            try {
                await doInference(sessionId)
            } finally {
                inferenceInFlightRef.current = false
            }
            inferenceTimerRef.current = window.setInterval(async () => {
                if (inferenceInFlightRef.current || !autoInferenceRef.current || sessionId !== autoInferenceSessionRef.current) {
                    return
                }

                inferenceInFlightRef.current = true
                try {
                    await doInference(sessionId)
                } finally {
                    inferenceInFlightRef.current = false
                }
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
        const keyActionMap: Record<string, string> = {
            'ArrowUp': 'up',
            'KeyW': 'up',
            'ArrowDown': 'down',
            'KeyS': 'down',
            'ArrowLeft': 'left',
            'KeyA': 'left',
            'ArrowRight': 'right',
            'KeyD': 'right',
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
            const action = keyActionMap[e.code]
            if (action && carConnected && carIP && !keys.current[e.code]) {
                keys.current[e.code] = true
                carControl(carIP, action, 50).catch(() => {})
            }
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
            const directionKeys = ['ArrowUp', 'KeyW', 'ArrowDown', 'KeyS', 'ArrowLeft', 'KeyA', 'ArrowRight', 'KeyD']
            if (directionKeys.includes(e.code)) {
                keys.current[e.code] = false
                if (carIP && carConnected) {
                    carControl(carIP, 'stop', 50).catch(() => {})
                }
                return
            }
            keys.current[e.code] = false
        }

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)
        }
    }, [carConnected, carIP])

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

    useEffect(() => {
        if (collectTimerRef.current) {
            window.clearInterval(collectTimerRef.current)
            collectTimerRef.current = null
        }

        if (!isRecording) {
            return
        }

        collectTimerRef.current = window.setInterval(() => {
            if (collectInFlightRef.current) return

            const imageData = fpvCameraViewRef.current?.getImageData()
            if (!imageData) return

            const actions = getCurrentActions()
            collectInFlightRef.current = true
            collectImage({
                image: imageData,
                actions,
                car_ip: carIP,
                timestamp: Date.now(),
            })
                .then((data) => {
                    if (data.count !== undefined) {
                        setCollectedCount(data.count)
                    }
                })
                .catch((error: unknown) => {
                    console.error("Collect image failed:", error)
                })
                .finally(() => {
                    collectInFlightRef.current = false
                })
        }, COLLECT_INTERVAL)

        return () => {
            if (collectTimerRef.current) {
                window.clearInterval(collectTimerRef.current)
                collectTimerRef.current = null
            }
        }
    }, [carIP, getCurrentActions, isRecording])

    return (
        <div className="flex flex-col h-screen bg-slate-950 overflow-hidden">
            {/* 顶部标题栏 */}
            <div className="flex items-center justify-between px-6 py-2 bg-slate-900/50 border-b border-slate-800">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-linear-to-br from-red-600 to-orange-600 flex items-center justify-center shadow-lg shadow-red-900/20">
                        <span className="text-white font-bold text-sm">REAL</span>
                    </div>
                    <div>
                        <h2 className="text-base font-bold text-slate-100">AKA ACT 小车训练平台</h2>
                        <p className="text-xs text-slate-500">Action Chunking with Transformers - Physical Robot Edition</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700">
                        <span className={`w-2 h-2 rounded-full ${carConnected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}/>
                        <span className="text-xs text-slate-300">
                            {carConnected ? 'Robot Connected' : 'Connecting...'}
                        </span>
                    </div>
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
                        onResetCar={() => resetCar(realSocket)}
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

                {/* 中间 - 前方摄像头 */}
                <div className="flex-1 min-w-0 flex flex-col">
                    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 h-full">
                        <RealCameraView
                            ref={topdownCameraViewRef}
                            title="前方摄像头 / 俯视视角"
                            description="预留给前置摄像头，作为环境俯视观察。"
                            devices={cameraDevices}
                            selectedDeviceId={topdownCameraId}
                            onDeviceChange={setTopdownCameraId}
                            cameraError={cameraPermissionError}
                            isRecording={isRecording}
                        />
                    </div>
                </div>

                {/* 右侧 - 小车控制 + 日志 */}
                <div className="w-96 flex flex-col min-w-0">
                    <RealRightPanel
                        ref={fpvCameraViewRef}
                        carState={carState}
                        isRecording={isRecording}
                        getCurrentActions={getCurrentActions}
                        carIP={carIP}
                        onCarIPChange={handleCarIPChange}
                        carConnected={carConnected}
                        cameraDevices={cameraDevices}
                        fpvCameraId={fpvCameraId}
                        onFpvCameraChange={setFpvCameraId}
                        fpvCameraError={cameraPermissionError}
                    />
                </div>
            </div>
        </div>
    )
}

export default RealPage;
