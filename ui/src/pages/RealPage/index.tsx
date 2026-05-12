import {useCallback, useEffect, useRef, useState} from "react"
import {
    realSocket,
    resetCar,
    getCarState,
    setEpisode,
    getEpisodes,
    deleteEpisode,
    runInferenceWithSocket,
    startEpisode,
    endEpisode,
    getEpisodeStatus,
    onTrainingProgress,
} from "../../api/socket.ts";
import {startTraining, stopTraining, loadTrainedModel, collectImage, listDatasetDirs, listModels} from "../../api/api";
import {carHeartbeat, motorDirect, carControl, isCarApiSuccess, carStatus} from "../../api/realCar";
import type {CarState} from "../../models/types.ts";
import {TrainingControl} from "../SimPage/TrainingControl.tsx";
import {InferenceControl} from "../SimPage/InferenceControl.tsx";
import {RealCameraView, type CameraDeviceOption, type RealCameraViewRef} from "./RealCameraView.tsx";
import {type MjpegStreamViewRef} from "./MjpegStreamView.tsx";
import {RealRightPanel} from "./RealRightPanel.tsx";
import {showToast} from "../../lib/toast.ts";
import {getDatasetPath, getTrainPath, getModelPath} from "../../lib/constants.ts";
import {KEY_TO_ACTION} from "../SimPage/actionMapping.ts";
import {useSimCarStore} from "../../stores/simCarStore.ts";

const RealPage = () => {
    const keys = useRef<Record<string, boolean>>({})
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
    const [collectionFps, setCollectionFps] = useState(20)
    const [currentEpisode, setCurrentEpisode] = useState(1)
    const [resumeTraining, setResumeTraining] = useState(false)
    const [isModelLoaded, setIsModelLoaded] = useState(false)
    const [inferenceResult, setInferenceResult] = useState<string[]>([])
    const [isInferring, setIsInferring] = useState(false)
    const [autoInference, setAutoInference] = useState(false)
    const autoInferenceRef = useRef(false)
    const autoInferenceSessionRef = useRef(0)
    const inferenceInFlightRef = useRef(false)
    const inferenceTimerRef = useRef<number | null>(null)
    const inferenceLoopTimeoutRef = useRef<number | null>(null)
    const latestAutoCommandAbortRef = useRef<AbortController | null>(null)
    const latestAutoCommandSeqRef = useRef(0)
    const smoothedActionRef = useRef<[number, number]>([0, 0])
    const topdownCameraViewRef = useRef<RealCameraViewRef | null>(null)
const fpvCameraViewRef = useRef<MjpegStreamViewRef | null>(null)
    const collectTimerRef = useRef<number | null>(null)
    const collectInFlightRef = useRef(false)
    const [cameraDevices, setCameraDevices] = useState<CameraDeviceOption[]>([])
    const [topdownCameraId, setTopdownCameraId] = useState("")
    const [fpvCameraId, setFpvCameraId] = useState("")
    const [cameraPermissionError, setCameraPermissionError] = useState("")

    // Episode 管理状态
    const [isRecording, setIsRecording] = useState(false)
    const [episodeTaskName, setEpisodeTaskName] = useState("default")
    const [datasetName, setDatasetName] = useState("default")
    const [datasets, setDatasets] = useState<string[]>([])
    const [models, setModels] = useState<string[]>([])
    const [selectedModel, setSelectedModel] = useState<string>("")
    const userId = useSimCarStore.getState().userId
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
            if (data.error) {
                showToast.error(`保存失败: ${data.error}`)
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

        getEpisodes(realSocket, userId)
        getEpisodeStatus(realSocket, userId)

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
        loadDatasets()

        return () => {
            realSocket.off("connected")
            realSocket.off("car_state_update")
            realSocket.off("collection_count")
            realSocket.off("training_progress")
            realSocket.off("episode_status")
            realSocket.off("episode_started")
            realSocket.off("episode_ended")
            realSocket.off("episode_finalized")
            unsubscribeTrainingProgress()
        }
    }, [])

    // 加载模型列表（当 userId 或 datasetName 变化时重新加载）
    useEffect(() => {
        setIsModelLoaded(false)
        setSelectedModel("")
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

    useEffect(() => {
        if (!navigator.mediaDevices?.enumerateDevices || !navigator.mediaDevices?.getUserMedia) {
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

    const heartbeatIPRef = useRef("")
    const checkCarHeartbeat = async (ip: string) => {
        if (!ip) {
            setCarConnected(false)
            heartbeatIPRef.current = ""
            return
        }
        heartbeatIPRef.current = ip
        const data = await carHeartbeat(ip)
        if (isCarApiSuccess(data) && heartbeatIPRef.current === ip) {
            setCarConnected(true)
            return
        }
    }

    const handleCarIPChange = (ip: string) => {
        setCarIP(ip)
        setCarConnected(false)
        checkCarHeartbeat(ip)
    }

    const sendInferenceActionToCar = async (
        left: number,
        right: number,
        duration: number,
        options?: { signal?: AbortSignal },
    ) => {
        if (!carIP) {
            throw new Error("请先输入小车IP")
        }

        const mapVelocityToMotorCommand = (value: number) => {
            if (Math.abs(value) < 1e-3) {
                return 0
            }
            const sign = value >= 0 ? 1 : -1
            return sign * Math.round(Math.abs(value))
        }

        const leftCommand = mapVelocityToMotorCommand(left)
        const rightCommand = mapVelocityToMotorCommand(right)
        const data = await motorDirect(carIP, leftCommand, rightCommand, duration, {signal: options?.signal})

        if (!isCarApiSuccess(data)) {
            throw new Error(data?.error || data?.detail || data?.message || "发送推理控制到小车失败")
        }

        return {
            leftCommand,
            rightCommand,
        }
    }

    const dispatchLatestAutoInferenceAction = useCallback((left: number, right: number) => {
        const leftCommand = Math.round(left)
        const rightCommand = Math.round(right)
        setInferenceResult([`motor=[${leftCommand}, ${rightCommand}]`])

        motorDirect(carIP, leftCommand, rightCommand, 0)
            .then((data) => {
                setInferenceResult([`motor=[${data.left}, ${data.right}], duration=0s`])
            })
            .catch((error: unknown) => {
                console.error("Auto inference motor command failed:", error)
            })
    }, [carIP])

    const stopInferenceAndCar = useCallback(() => {
        setAutoInference(false)
        autoInferenceRef.current = false
        autoInferenceSessionRef.current += 1
        inferenceInFlightRef.current = false
        latestAutoCommandSeqRef.current += 1
        latestAutoCommandAbortRef.current?.abort()
        latestAutoCommandAbortRef.current = null
        smoothedActionRef.current = [0, 0]

        if (inferenceTimerRef.current) {
            clearInterval(inferenceTimerRef.current)
            inferenceTimerRef.current = null
        }

        if (inferenceLoopTimeoutRef.current) {
            window.clearTimeout(inferenceLoopTimeoutRef.current)
            inferenceLoopTimeoutRef.current = null
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
        setEpisode(realSocket, userId, episodeId)

        if (episodeId < episodeToDelete) {
            deleteEpisode(realSocket, userId, episodeToDelete)
            getEpisodes(realSocket, userId)
        }
        setCurrentEpisode(episodeId)
    }

    const handleStartEpisode = () => {
        startEpisode(realSocket, userId, currentEpisode, episodeTaskName)
    }

    const handleSelectDataset = (name: string) => {
        setDatasetName(name)
        setIsModelLoaded(false)
        setSelectedModel("")
        getEpisodes(realSocket, userId)
    }

    const handleEndEpisode = () => {
        endEpisode(realSocket, userId, currentEpisode)
        setCurrentEpisode(currentEpisode + 1)
        setCollectedCount(0)
        getEpisodes(realSocket, userId)
    }

    const handleStartTraining = async () => {
        try {
            const userId = useSimCarStore.getState().userId
            const result = await startTraining({
                data_dir: getDatasetPath(userId, datasetName),
                output_dir: getTrainPath(userId, datasetName),
                epochs: trainingEpochs,
                batch_size: 8,
                lr: 1e-4,
                resume_from: resumeTraining ? getModelPath(userId, datasetName) : undefined,
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
            await stopTraining()
        } catch {
            showToast.error('停止训练失败')
        }
    }

    const handleLoadModel = async (modelName: string) => {
        try {
            const userId = useSimCarStore.getState().userId
            const dataDir = getDatasetPath(userId, datasetName)
            const modelPath = `output/train/${userId}/${modelName}/model.pt`
            const result = await loadTrainedModel(dataDir, modelPath)
            if (result.success) {
                setIsModelLoaded(true)
                setSelectedModel(modelName)
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

    const doInference = async (sessionId?: number) => {
        if (sessionId !== undefined && sessionId !== autoInferenceSessionRef.current) {
            return
        }

        if (!carIP) {
            throw new Error("请先输入小车IP")
        }

        const allStatus = await carStatus(carIP)
        const velLeft = allStatus?.left_speed
        const velRight = allStatus?.right_speed
        if (typeof velLeft !== "number" || typeof velRight !== "number") {
            throw new Error("小车实时状态返回格式不正确")
        }

        setCarConnected(true)
        setCarState((prev) => ({
            ...prev,
            vel_left: velLeft,
            vel_right: velRight,
        }))

        const imageBase64 = fpvCameraViewRef.current?.getImageData()
        const state: [number, number] = [velLeft, velRight]
        const result = await runInferenceWithSocket(realSocket, state, imageBase64)

        if (sessionId !== undefined && (!autoInferenceRef.current || sessionId !== autoInferenceSessionRef.current)) {
            return
        }

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
                console.error("Invalid velocity values:", {velLeftTarget, velRightTarget})
                return
            }

            if (sessionId !== undefined && (!autoInferenceRef.current || sessionId !== autoInferenceSessionRef.current)) {
                return
            }

            if (sessionId === undefined) {
                const {leftCommand, rightCommand} = await sendInferenceActionToCar(
                    velLeftTarget,
                    velRightTarget,
                    0,
                )
                const velStr = `v=[${velLeftTarget.toFixed(2)}, ${velRightTarget.toFixed(2)}] -> motor=[${leftCommand}, ${rightCommand}], duration=1s`
                setInferenceResult([velStr])
                return
            }

            dispatchLatestAutoInferenceAction(velLeftTarget, velRightTarget)
        } else if (!result.success) {
            throw new Error(result.error || '推理失败1')
        }
    }

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
        }
    }

    useEffect(() => {
        if (!autoInferenceRef.current) {
            return
        }

        const sessionId = autoInferenceSessionRef.current
        const inferenceFps = 5
        const inferenceInterval = 1000 / inferenceFps
        let cancelled = false

        const runLoop = async () => {
            if (cancelled || !autoInferenceRef.current || sessionId !== autoInferenceSessionRef.current) {
                return
            }

            const startedAt = performance.now()
            inferenceInFlightRef.current = true
            try {
                await doInference(sessionId)
            } catch (error) {
                console.error('Auto inference error:', error)
            } finally {
                inferenceInFlightRef.current = false
            }

            const elapsed = performance.now() - startedAt
            const delay = Math.max(inferenceInterval - elapsed, 0)
            inferenceLoopTimeoutRef.current = window.setTimeout(() => {
                void runLoop()
            }, delay)
        }

        void runLoop()

        return () => {
            cancelled = true
            if (inferenceLoopTimeoutRef.current) {
                window.clearTimeout(inferenceLoopTimeoutRef.current)
                inferenceLoopTimeoutRef.current = null
            }
        }
    }, [collectionFps, autoInference])

    const getCurrentActions = useCallback((): string[] => {
        const actions: string[] = []
        for (const [code, action] of Object.entries(KEY_TO_ACTION)) {
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

    useEffect(() => {
        if (collectTimerRef.current) {
            window.clearInterval(collectTimerRef.current)
            collectTimerRef.current = null
        }

        if (!isRecording) {
            return
        }

        const collectInterval = 1000 / Math.max(collectionFps, 1)

        collectTimerRef.current = window.setInterval(async () => {
            if (collectInFlightRef.current) return
            if (!isRecording || !carIP) return

            collectInFlightRef.current = true
            const captureTimestampMs = Date.now()
            try {
                const allStatus = await carStatus(carIP)
                if (typeof allStatus.left_speed !== 'number' || typeof allStatus.right_speed !== 'number') {
                    throw new Error("carStatus 返回格式不正确")
                }

                const imageData = fpvCameraViewRef.current?.getImageData()
                console.log('[Collect] imageData:', imageData ? 'ok' : 'undefined', 'isRecording:', isRecording)
                if (!imageData) {
                    collectInFlightRef.current = false
                    return
                }

                const data = await collectImage({
                    image: imageData,
                    user_id: userId,
                    dataset_name: datasetName,
                    timestamp: captureTimestampMs,
                    state: {
                        vel_left: allStatus.left_speed,
                        vel_right: allStatus.right_speed,
                    },
                    action: [
                        typeof allStatus.left_target === 'number' ? allStatus.left_target : 0,
                        typeof allStatus.right_target === 'number' ? allStatus.right_target : 0,
                    ],
                })
                if (data.count !== undefined) {
                    setCollectedCount(data.count)
                }
            } catch (error: unknown) {
                console.error("Collect image failed:", error)
            } finally {
                collectInFlightRef.current = false
            }
        }, collectInterval)

        return () => {
            if (collectTimerRef.current) {
                window.clearInterval(collectTimerRef.current)
                collectTimerRef.current = null
            }
        }
    }, [carIP, collectionFps, getCurrentActions, isRecording])

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
                        onResetCar={() => resetCar(realSocket)}
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
