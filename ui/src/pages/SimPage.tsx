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
    getTrainingStatus,
    stopTraining,
    loadTrainedModel,
    runInference,
    startEpisode,
    endEpisode,
    finalizeEpisode,
    getEpisodeStatus,
    sendImageData,
} from "../api/socket";
import type {CarState} from "../models/types.ts";

const MAP_W = 800;
const MAP_H = 600;

const FPS = 30
const SEND_INTERVAL = 50 // 发送控制指令间隔(ms)
const frameInterval = 1000 / FPS

const SimPage = () => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const fpvRef = useRef<HTMLCanvasElement | null>(null);
    const keys = useRef<Record<string, boolean>>({})
    const [carState, setCarState] = useState<CarState>({
        x: 400,
        y: 300,
        angle: -Math.PI / 2,
        speed: 0,
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
    const inferenceTimerRef = useRef<number | null>(null)
    const lastInferredActionRef = useRef<string[]>([])

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

        // 获取初始训练状态
        getTrainingStatus().then(data => {
            setIsTraining(data.is_running)
            setTrainingProgress({
                epoch: data.epoch,
                total_epochs: data.total_epochs,
                loss: data.loss,
                progress: data.progress
            })
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

    const sendCommand = (cmd: string) => {
        // 发送动作到后端
        sendActions([cmd])
    }

    const handleSetEpisode = (episodeId: number) => {
        if (episodeId < 1) return
        if (isRecording) {
            if (!confirm('正在录制中，切换轮次将结束当前录制。是否继续?')) {
                return
            }
            handleEndEpisode()
        }

        // 重置帧数
        setCollectedCount(0)

        // 如果是回退到之前的轮次，删除当前轮次的数据
        if (episodeId < currentEpisode) {
            deleteEpisode(currentEpisode)
            setCurrentEpisode(episodeId)
            // 刷新轮次列表
            getEpisodes()
        } else {
            setCurrentEpisode(episodeId)
        }
        setEpisode(episodeId)
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
            alert('加载模型失败')
        }
    }

    const doInference = async () => {
        // 使用当前车辆状态作为输入
        const state = [
            carState.x,
            carState.y,
            carState.angle,
            carState.speed,
        ]

        // 获取第一人称视角的图像
        const fpv = fpvRef.current
        let imageBase64: string | undefined
        if (fpv) {
            imageBase64 = fpv.toDataURL('image/jpeg', 0.8)
        }

        const result = await runInference(state, imageBase64)
        console.log('推理请求: imageBase64 length =', imageBase64?.length)
        if (result.success) {
            // 解析动作 - 用阈值提取多个动作
            const actions: string[] = []
            // result.action 是 [action_chunk_size, action_dim] 的二维数组
            // 取第一个动作
            console.log(result)
            const firstAction = result.action[0]
            console.log('模型输出:', firstAction)
            console.log('  forward:', firstAction[0], 'backward:', firstAction[1], 'left:', firstAction[2], 'right:', firstAction[3], 'stop:', firstAction[4])

            // 用阈值提取多个动作（多标签模式）
            const actionNames = ['forward', 'backward', 'left', 'right', 'stop']
            const threshold = 0.01  // 阈值，超过该值就执行对应动作

            // 找出所有超过阈值的动作
            for (let i = 0; i < firstAction.length; i++) {
                if (firstAction[i] > threshold) {
                    actions.push(actionNames[i])
                    console.log(`  执行 ${actionNames[i]}: ${firstAction[i]} > ${threshold}`)
                }
            }

            // 如果没有超过阈值的动作，选择最大的那个作为默认
            if (actions.length === 0) {
                const maxIdx = firstAction.indexOf(Math.max(...firstAction))
                actions.push(actionNames[maxIdx])
                console.log('无动作超过阈值，使用最大:', actionNames[maxIdx])
            }

            console.log('预测动作:', actions)

            setInferenceResult(actions)
            lastInferredActionRef.current = actions

            // 执行动作
            if (actions.length > 0) {
                console.log(actions)
                sendActions(actions)
            }
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
            // 停止自动推理
            setAutoInference(false)
            if (inferenceTimerRef.current) {
                clearInterval(inferenceTimerRef.current)
                inferenceTimerRef.current = null
            }
        } else {
            // 开始自动推理
            setAutoInference(true)
            // 先执行一次
            await doInference()
            // 然后每50ms执行一次 (与sendActions频率一致)
            inferenceTimerRef.current = window.setInterval(async () => {
                await doInference()
            }, 50)
        }
    }

    const drawGrid = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
        ctx.strokeStyle = '#e0e0e0'
        ctx.lineWidth = 1
        const gridSize = 50

        ctx.beginPath()
        for (let x = 0; x <= w; x += gridSize) {
            ctx.moveTo(x, 0)
            ctx.lineTo(x, h)
        }
        for (let y = 0; y <= h; y += gridSize) {
            ctx.moveTo(0, y)
            ctx.lineTo(w, y)
        }
        ctx.stroke()
    }, [])

    const drawCarBody = useCallback((ctx: CanvasRenderingContext2D, x: number, y: number, angle: number) => {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.fillStyle = 'blue';
        ctx.fillRect(-20, -10, 40, 20);
        ctx.fillStyle = 'yellow';
        ctx.beginPath();
        ctx.arc(15, -6, 3, 0, Math.PI * 2);
        ctx.arc(15, 6, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#2c3e50'
        ctx.fillRect(5, -8, 10, 16)
        ctx.restore();
    }, [])

    const drawTopDown = useCallback((ctx: CanvasRenderingContext2D) => {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

        drawGrid(ctx, ctx.canvas.width, ctx.canvas.height)

        ctx.save()

        // 使用后端的车辆状态
        drawCarBody(ctx, carState.x, carState.y, carState.angle)

        ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        ctx.beginPath();
        ctx.moveTo(carState.x, carState.y);
        ctx.lineTo(carState.x + Math.cos(carState.angle - Math.PI / 6) * 100, carState.y + Math.sin(carState.angle - Math.PI / 6) * 100);
        ctx.moveTo(carState.x, carState.y);
        ctx.lineTo(carState.x + Math.cos(carState.angle + Math.PI / 6) * 100, carState.y + Math.sin(carState.angle + Math.PI / 6) * 100);
        ctx.stroke();

        ctx.restore()
    }, [carState, drawCarBody, drawGrid])


    const getRaySegmentIntersection = (rx: number, ry: number, rdx: number, rdy: number, wall: {
        x1: number,
        y1: number,
        x2: number,
        y2: number
    }) => {
        const {x1, y1, x2, y2} = wall;
        const v1x = x1 - rx;
        const v1y = y1 - ry;
        const v2x = x2 - x1;
        const v2y = y2 - y1;
        const v3x = -rdx; // 射线方向反转
        const v3y = -rdy;

        const cross = v2x * v3y - v2y * v3x;
        if (Math.abs(cross) < 0.0001) return null; // 平行

        const t1 = (v2x * v1y - v2y * v1x) / cross; // 射线距离
        const t2 = (v3x * v1y - v3y * v1x) / cross; // 线段比例 (0~1)

        // t1 > 0 代表射线前方，t2 在 0~1 代表交点在线段上
        if (t1 > 0 && t2 >= 0 && t2 <= 1) {
            return t1;
        }
        return null;
    };

    const castRay = (sx: number, sy: number, angle: number) => {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        let minDist = Infinity;
        let hitColor = null;

        // 将所有障碍物转换为线段进行检测
        const boundaries = [
            {x1: 0, y1: 0, x2: MAP_W, y2: 0, color: '#333'}, // 上墙
            {x1: MAP_W, y1: 0, x2: MAP_W, y2: MAP_H, color: '#333'}, // 右墙
            {x1: MAP_W, y1: MAP_H, x2: 0, y2: MAP_H, color: '#333'}, // 下墙
            {x1: 0, y1: MAP_H, x2: 0, y2: 0, color: '#333'}  // 左墙
        ];

        // 检测射线与每一条线段的交点
        boundaries.forEach(wall => {
            const dist = getRaySegmentIntersection(sx, sy, cos, sin, wall);
            if (dist !== null && dist < minDist) {
                minDist = dist;
                hitColor = wall.color;
            }
        });

        return minDist === Infinity ? null : {distance: minDist, color: hitColor};
    };

    const drawFirstPerson = useCallback((ctx: CanvasRenderingContext2D) => {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const {x, y, angle} = carState;

        // 天空和地面
        ctx.fillStyle = '#87CEEB'; // 天空蓝
        ctx.fillRect(0, 0, w, h / 2);
        ctx.fillStyle = '#7f8c8d'; // 地面灰
        ctx.fillRect(0, h / 2, w, h / 2);

        // 参数
        const fov = Math.PI / 3; // 60度视野
        const rayCount = w / 4;  // 射线数量 (为了性能，每4个像素投射一条，然后画宽一点)
        const rayWidth = w / rayCount;

        // 遍历每一条射线
        for (let i = 0; i < rayCount; i++) {
            // 当前射线角度 = 车角度 - 半个FOV + 增量
            const rayAngle = (angle + Math.PI - fov / 2) + (i / rayCount) * fov;

            // 计算这一条射线碰到了什么，以及距离是多少
            const hit = castRay(x, y, rayAngle);

            if (hit) {
                const correctedDist = hit.distance * Math.cos(rayAngle - angle);

                const wallHeight = (h * 40) / correctedDist;

                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                ctx.fillStyle = hit.color;
                ctx.globalAlpha = Math.max(0.3, 1 - correctedDist / 600);
                ctx.fillRect(i * rayWidth, (h - wallHeight) / 2, rayWidth + 1, wallHeight);
                ctx.globalAlpha = 1.0;
            }
        }
    }, [carState, castRay])


    useEffect(() => {
        const canvas = canvasRef.current
        const fpv = fpvRef.current
        if (canvas == null || fpv == null) return
        const ctxTop = canvas.getContext('2d')
        const ctxFpv = fpv.getContext('2d')

        if (ctxTop == null || ctxFpv == null) return

        ctxFpv.imageSmoothingEnabled = false;

        let animationFrameId: number

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

        // 获取当前按下的动作列表
        const getCurrentActions = (): string[] => {
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
        }

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)

        let lastTime = 0;
        let lastSendTime = 0;
        let lastCollectTime = 0;
        const COLLECT_INTERVAL = 500; // 采集间隔(ms)，2fps

        const renderLoop = (currentTime: number) => {
            animationFrameId = window.requestAnimationFrame(renderLoop)

            const delta = currentTime - lastTime

            if (delta < frameInterval) return

            lastTime = currentTime - (delta % frameInterval)

            // 控制发送频率
            if (currentTime - lastSendTime >= SEND_INTERVAL) {
                // 如果在自动推理模式，使用推断的动作；否则使用键盘输入
                const actions = autoInference ? lastInferredActionRef.current : getCurrentActions()
                sendActions(actions)
                lastSendTime = currentTime
            }

            // 如果正在录制，收集数据
            if (isRecording && currentTime - lastCollectTime >= COLLECT_INTERVAL) {
                const actions = getCurrentActions()
                // 从第一人称Canvas获取图像数据
                const imageData = fpv.toDataURL('image/jpeg', 0.8)
                sendImageData(imageData, actions)
                lastCollectTime = currentTime
            }

            // 渲染
            drawTopDown(ctxTop)
            drawFirstPerson(ctxFpv)
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)

            window.cancelAnimationFrame(animationFrameId)
        }
    }, [drawFirstPerson, drawTopDown, isRecording])

    return (
        <div className="flex flex-col gap-3 p-4 h-screen overflow-hidden">
            <h1 className="text-center font-bold">AKA-Sim 模拟器</h1>
            <div className="flex gap-5 flex-1 items-stretch">
                <div className="w-64 flex flex-col h-full">
                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2">
                        <div className="font-semibold">训练控制</div>

                        {/* 数据收集流程 */}
                        <div className="border border-gray-300 rounded p-2">
                            <div className="text-xs font-semibold mb-2">数据采集</div>

                            {/* 录制状态 */}
                            <div className="flex items-center gap-2 mb-2">
                                <span
                                    className={`text-xs px-2 py-0.5 rounded ${isRecording ? 'bg-red-500 text-black' : 'bg-gray-300'}`}>
                                    {isRecording ? '收集中' : '未录制'}
                                </span>
                                {collectedCount > 0 && (
                                    <span className="text-xs text-gray-600">{collectedCount} 帧</span>
                                )}
                            </div>

                            {/* 操作按钮 */}
                            <div className="flex flex-col gap-2">
                                {isRecording ? (
                                    // 录制中：显示结束按钮
                                    <button
                                        onClick={handleEndEpisode}
                                        className="px-2 py-1.5 text-xs bg-red-500 text-black rounded hover:bg-red-600 font-medium"
                                    >
                                        结束采集 (第{currentEpisode}轮)
                                    </button>
                                ) : (
                                    // 未录制：显示开始和复位按钮
                                    <>
                                        <button
                                            onClick={handleStartEpisode}
                                            className="px-2 py-1.5 text-xs bg-green-500 text-black rounded hover:bg-green-600 font-medium"
                                        >
                                            开始采集 (第{currentEpisode}轮)
                                        </button>
                                        <button
                                            onClick={() => resetCar()}
                                            className="px-2 py-1.5 text-xs bg-yellow-500 text-black rounded hover:bg-yellow-600 font-medium"
                                        >
                                            复位场景
                                        </button>
                                    </>
                                )}
                            </div>

                            {/* 回退按钮 */}
                            {!isRecording && currentEpisode > 1 && (
                                <div className="mt-2 pt-2 border-t border-gray-200">
                                    <button
                                        onClick={() => handleSetEpisode(currentEpisode - 1)}
                                        className="px-2 py-1 text-xs bg-gray-400 text-black rounded hover:bg-gray-500"
                                    >
                                        回退到第 {currentEpisode - 1} 轮
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* 当前轮次 */}
                        {!isRecording && (
                            <div className="text-xs text-gray-600">
                                当前轮次: {currentEpisode}
                            </div>
                        )}

                        {/* 轮次选择 */}
                        <div className="text-xs text-gray-500 mb-1">
                            可修改轮次重新采集
                        </div>
                        <div className="flex items-center gap-2">
                            <label className="text-xs text-gray-600">采集轮次:</label>
                            <input
                                type="number"
                                value={currentEpisode}
                                onChange={(e) => handleSetEpisode(parseInt(e.target.value) || 1)}
                                min={1}
                                disabled={isRecording}
                                className="w-16 px-2 py-1 text-sm border rounded"
                            />
                        </div>

                        {/* 显示各轮次数据量 */}
                        {Object.keys(episodeCounts).length > 0 && (
                            <div className="text-xs text-gray-600 max-h-24 overflow-y-auto border rounded p-1">
                                {Object.entries(episodeCounts).map(([ep, count]) => (
                                    <div key={ep} className="flex justify-between">
                                        <span>轮次 {ep}:</span>
                                        <span>{count} 样本</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        <div className="flex items-center gap-2">
                            <label className="text-xs text-gray-600">训练轮次:</label>
                            <input
                                type="number"
                                value={trainingEpochs}
                                onChange={(e) => setTrainingEpochs(parseInt(e.target.value) || 1)}
                                min={1}
                                max={1000}
                                disabled={isTraining}
                                className="w-20 px-2 py-1 text-sm border rounded"
                            />
                        </div>

                        {/* 增量训练选项 */}
                        <div className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                id="resumeTraining"
                                checked={resumeTraining}
                                onChange={(e) => setResumeTraining(e.target.checked)}
                                disabled={isTraining}
                                className="w-4 h-4"
                            />
                            <label htmlFor="resumeTraining" className="text-xs text-gray-600">
                                从已有模型继续训练
                            </label>
                        </div>
                        {!isTraining ? (
                            <button
                                onClick={handleStartTraining}
                                disabled={collectedCount === 0}
                                className={`px-3 py-1 rounded ${collectedCount > 0 ? 'bg-purple-500 text-black hover:bg-purple-600' : 'bg-gray-300 text-gray-500'}`}
                            >
                                开始训练
                            </button>
                        ) : (
                            <button
                                onClick={handleStopTraining}
                                className="px-3 py-1 bg-red-500 text-black rounded hover:bg-red-600"
                            >
                                停止训练
                            </button>
                        )}
                        {isTraining && (
                            <div className="mt-2">
                                <div className="text-xs text-gray-600">
                                    Epoch: {trainingProgress.epoch}/{trainingProgress.total_epochs}
                                </div>
                                <div className="text-xs text-gray-600">
                                    Loss: {trainingProgress.loss.toFixed(6)}
                                </div>
                                <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-purple-500 h-2 rounded-full transition-all"
                                        style={{width: `${trainingProgress.progress * 100}%`}}
                                    />
                                </div>
                                <div className="text-xs text-gray-600 text-center mt-1">
                                    {Math.round(trainingProgress.progress * 100)}%
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2 mt-4">
                        <div className="font-semibold">推理控制</div>
                        <div className="text-xs text-gray-600">
                            模型状态: {isModelLoaded ? '已加载' : '未加载'}
                        </div>
                        {!isModelLoaded ? (
                            <button
                                onClick={handleLoadModel}
                                className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600"
                            >
                                加载模型
                            </button>
                        ) : (
                            <div className="flex gap-2">
                                <button
                                    onClick={handleInference}
                                    disabled={isInferring || autoInference}
                                    className={`px-3 py-1 rounded ${isInferring || autoInference ? 'bg-gray-300' : 'bg-green-500 text-black hover:bg-green-600'}`}
                                >
                                    {isInferring ? '推理中...' : '单次推理'}
                                </button>
                                <button
                                    onClick={handleAutoInference}
                                    disabled={isInferring}
                                    className={`px-3 py-1 rounded ${autoInference ? 'bg-red-500 text-black' : 'bg-orange-500 text-black hover:bg-orange-600'}`}
                                >
                                    {autoInference ? '停止自动' : '自动推理'}
                                </button>
                            </div>
                        )}
                        {inferenceResult.length > 0 && (
                            <div className="text-xs text-gray-600 mt-2">
                                推理动作: {inferenceResult.join(', ')}
                            </div>
                        )}
                    </div>
                </div>
                <div className="flex-1 flex flex-col h-full">
                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-3 h-full">
                        <div className="font-semibold">俯视地图</div>
                        <div className="relative flex justify-center">
                            <canvas
                                ref={canvasRef}
                                width={800}
                                height={600}
                                className="bg-white block rounded"
                            />
                            <div className="absolute top-2 left-2 bg-white/85 p-1.5 rounded text-xs">
                                使用 WASD 或 方向键 移动<br/>
                                实时同步后端状态
                            </div>
                        </div>
                        <div className="flex gap-2.5 flex-wrap justify-center items-center">
                            <button onClick={() => sendCommand('forward')}
                                    className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 前进
                            </button>
                            <button onClick={() => sendCommand('left')}
                                    className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 左转
                            </button>
                            <button onClick={() => sendCommand('right')}
                                    className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 右转
                            </button>
                            <button onClick={() => sendCommand('backward')}
                                    className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 后退
                            </button>
                            <button onClick={() => resetCar()}
                                    className="px-3 py-1 bg-green-500 text-black rounded hover:bg-green-600">复位
                            </button>
                            <span className="text-xs text-gray-600 ml-2">帧数: {collectedCount}</span>
                        </div>
                    </div>
                </div>
                <div className="w-90 flex flex-col h-full">
                    <div
                        className="border-2 border-gray-800 rounded-lg bg-gray-100 p-2.5 flex flex-col gap-2 h-full text-gray-800 overflow-y-auto min-h-0">
                        <div className="font-semibold">车载摄像头</div>
                        <canvas ref={fpvRef} width={320} height={240}
                                className="bg-black border-2 border-gray-800 rounded self-center"/>
                        <div className="text-xs text-gray-600">
                            说明：右侧画面是根据左侧地图实时计算生成的伪3D视角。<br/>
                            状态来源：后端实时同步
                        </div>
                        <div className="text-xs mt-2">
                            <div className="font-semibold">当前状态:</div>
                            <div>X: {carState.x.toFixed(1)}</div>
                            <div>Y: {carState.y.toFixed(1)}</div>
                            <div>角度: {(carState.angle * 180 / Math.PI).toFixed(1)}°</div>
                            <div>速度: {carState.speed.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default SimPage;
