import {useCallback, useEffect, useRef} from "react"
import {resetCar} from "../api/socket.ts";

const MAP_W = 800;
const MAP_H = 600;

const FPS = 30
const frameInterval = 1000 / FPS

const SimPage = () => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const fpvRef = useRef<HTMLCanvasElement | null>(null);
    const keys = useRef<Record<string, boolean>>({})
    const carState = useRef({
        x: 400,
        y: 300,
        angle: -Math.PI / 2,
        speed: 0,        // 当前速度
        maxSpeed: 5,     // 最大速度
        acceleration: 0.2, // 加速度
        friction: 0.95,  // 摩擦力 (模拟惯性)
        rotationSpeed: 0.05 // 转向灵敏度
    })

    const checkCollision = (x: number, y: number) => {
        if (x < 0 || x > MAP_W || y < 0 || y > MAP_H) return true;
        return false
    };

    const updatePhysics = useCallback(() => {
        const state = carState.current

        // 前进 / 后退
        if (keys.current['ArrowUp'] || keys.current['KeyW']) {
            if (state.speed < state.maxSpeed) state.speed += state.acceleration
        }
        if (keys.current['ArrowDown'] || keys.current['KeyS']) {
            if (state.speed > -state.maxSpeed / 2) state.speed -= state.acceleration
        }

        if (keys.current['ArrowLeft'] || keys.current['KeyA']) {
            state.angle -= state.rotationSpeed
        }
        if (keys.current['ArrowRight'] || keys.current['KeyD']) {
            state.angle += state.rotationSpeed
        }

        if (!keys.current['ArrowUp'] && !keys.current['KeyW'] &&
            !keys.current['ArrowDown'] && !keys.current['KeyS']) {
            state.speed = 0
        }

        // 更新坐标 (核心三角函数：x = v*cos(θ), y = v*sin(θ))
        state.x += Math.cos(state.angle) * state.speed
        state.y += Math.sin(state.angle) * state.speed

        // 简单的边界检测 (碰到墙壁反弹)
        if (checkCollision(state.x, state.y)) {
            state.x -= Math.cos(state.angle) * state.speed * 2
            state.y -= Math.sin(state.angle) * state.speed * 2
            state.speed = 0
        }
    }, [])

    const sendCommand = (cmd: string) => {
        keys.current[cmd] = true
        setTimeout(() => {
            keys.current[cmd] = false
        }, 200)
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

    const drawCarBody = useCallback((ctx: CanvasRenderingContext2D) => {
        const {x, y, angle} = carState.current;
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

        drawCarBody(ctx)

        const {x, y, angle} = carState.current;
        ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(angle - Math.PI / 6) * 100, y + Math.sin(angle - Math.PI / 6) * 100);
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(angle + Math.PI / 6) * 100, y + Math.sin(angle + Math.PI / 6) * 100);
        ctx.stroke();

        ctx.restore()
    }, [drawCarBody, drawGrid])


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
        const {x, y, angle} = carState.current;

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
    }, [])


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

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)

        let lastTime = 0;

        const renderLoop = (currentTime: number) => {
            animationFrameId = window.requestAnimationFrame(renderLoop)

            const delta = currentTime - lastTime

            if (delta < frameInterval) return

            lastTime = currentTime - (delta % frameInterval)

            updatePhysics()
            drawTopDown(ctxTop)
            drawFirstPerson(ctxFpv)
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)

            window.cancelAnimationFrame(animationFrameId)
        }
    }, [drawFirstPerson, drawTopDown, updatePhysics])

    return (
        <div className="flex flex-col gap-3 p-4 h-screen overflow-hidden">
            <h1 className="text-center font-bold">AKA-Sim 模拟器</h1>
            <div className="flex gap-5 flex-1 items-stretch">
                <div className="w-64 flex flex-col h-full">
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
                                使用 QE 键旋转选中的目标物<br/>
                                选中目标物后按 Delete 键删除
                            </div>
                        </div>
                        <div className="flex gap-2.5 flex-wrap justify-center items-center">
                            <button onClick={() => sendCommand('ArrowUp')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 前进</button>
                            <button onClick={() => sendCommand('ArrowLeft')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 左转</button>
                            <button onClick={() => sendCommand('ArrowRight')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 右转</button>
                            <button onClick={() => sendCommand('ArrowDown')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 后退</button>
                            <button onClick={() => resetCar()} className="px-3 py-1 bg-green-500 text-black rounded hover:bg-green-600">复位</button>
                        </div>
                    </div>
                </div>
                <div className="w-90 flex flex-col h-full">
                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-2.5 flex flex-col gap-2 h-full text-gray-800 overflow-y-auto min-h-0">
                        <div className="font-semibold">车载摄像头</div>
                        <canvas ref={fpvRef} width={320} height={240}
                                className="bg-black border-2 border-gray-800 rounded self-center"/>
                        <div className="text-xs text-gray-600">
                            说明：右侧画面是根据左侧地图实时计算生成的伪3D视角。
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default SimPage;