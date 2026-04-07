import {useEffect, useRef, useState} from "react";
import type {CarState, Obstacle} from "../../models/types.ts";
import {useSimCarStore} from "../../stores/simCarStore.ts";

interface TopDownViewProps {
    obstacles: Obstacle[];
    onObstaclesChange: (obstacles: Obstacle[]) => void;
    collectedCount: number;
    resetCar: () => void;
    sendCommand: (action: [number, number]) => void;
}

const MAP_W = 800;
const MAP_H = 600;

const drawGrid = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
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
}

const drawCarBody = (ctx: CanvasRenderingContext2D, x: number, y: number, angle: number) => {
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
}

const drawObstacles = (ctx: CanvasRenderingContext2D, obstacles: Obstacle[]) => {
    obstacles.forEach(obs => {
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(obs.x - obs.width / 2, obs.y - obs.height / 2, obs.width, obs.height);
        ctx.strokeStyle = '#c0392b';
        ctx.lineWidth = 2;
        ctx.strokeRect(obs.x - obs.width / 2, obs.y - obs.height / 2, obs.width, obs.height);
    });
}

const drawTopDown = (ctx: CanvasRenderingContext2D, carState: CarState, obstacles: Obstacle[]) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

    drawGrid(ctx, ctx.canvas.width, ctx.canvas.height)

    // 绘制障碍物
    drawObstacles(ctx, obstacles)

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
}

export const TopDownView = ({
    obstacles,
    onObstaclesChange,
    collectedCount,
    resetCar,
    sendCommand
}: TopDownViewProps) => {
    const carState = useSimCarStore((state) => state.carState)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const [draggingIdx, setDraggingIdx] = useState<number | null>(null)
    const [dragOffset, setDragOffset] = useState<{x: number, y: number}>({x: 0, y: 0})

    // 检查点是否在障碍物内
    const isPointInObstacle = (px: number, py: number, obs: Obstacle): boolean => {
        return px >= obs.x - obs.width / 2 &&
               px <= obs.x + obs.width / 2 &&
               py >= obs.y - obs.height / 2 &&
               py <= obs.y + obs.height / 2
    }

    // 鼠标按下事件
    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current
        if (!canvas) return

        const rect = canvas.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top

        // 检查是否点击了某个障碍物
        for (let i = obstacles.length - 1; i >= 0; i--) {
            if (isPointInObstacle(x, y, obstacles[i])) {
                setDraggingIdx(i)
                setDragOffset({
                    x: x - obstacles[i].x,
                    y: y - obstacles[i].y
                })
                return
            }
        }
    }

    // 鼠标移动事件
    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (draggingIdx === null) return

        const canvas = canvasRef.current
        if (!canvas) return

        const rect = canvas.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top

        // 边界限制
        const obs = obstacles[draggingIdx]
        const newX = Math.max(obs.width / 2, Math.min(MAP_W - obs.width / 2, x - dragOffset.x))
        const newY = Math.max(obs.height / 2, Math.min(MAP_H - obs.height / 2, y - dragOffset.y))

        onObstaclesChange(obstacles.map((o, i) =>
            i === draggingIdx ? {...o, x: newX, y: newY} : o
        ))
    }

    // 鼠标释放事件
    const handleMouseUp = () => {
        setDraggingIdx(null)
    }

    useEffect(() => {
        const canvas = canvasRef.current
        if (canvas == null) return
        const ctx = canvas.getContext('2d')
        if (ctx == null) return

        let animationFrameId: number

        const renderLoop = () => {
            drawTopDown(ctx, carState, obstacles)
            animationFrameId = window.requestAnimationFrame(renderLoop)
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.cancelAnimationFrame(animationFrameId)
        }
    }, [carState, obstacles])

    // 鼠标事件处理
    const handleMouseLeave = () => {
        setDraggingIdx(null)
    }

    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-3 h-[88vh]">
            <div className="font-semibold">俯视地图</div>
            <div className="relative flex justify-center">
                <canvas
                    ref={canvasRef}
                    width={MAP_W}
                    height={MAP_H}
                    className="bg-white block rounded cursor-move"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseLeave}
                />
                <div className="absolute top-2 left-2 bg-white/85 p-1.5 rounded text-xs">
                    使用 WASD 或 方向键 移动<br/>
                    实时同步后端状态
                </div>
            </div>
            <div className="flex gap-2.5 flex-wrap justify-center items-center">
                <button onClick={() => sendCommand([0.2, 0.2])}
                        className="px-3 py-1 bg-blue-500 text-sm text-white rounded hover:bg-blue-600">指令: 前进
                </button>
                <button onClick={() => sendCommand([-0.2, 0.2])}
                        className="px-3 py-1 bg-blue-500 text-sm text-white rounded hover:bg-blue-600">指令: 左转
                </button>
                <button onClick={() => sendCommand([0.2, -0.2])}
                        className="px-3 py-1 bg-blue-500 text-sm text-white rounded hover:bg-blue-600">指令: 右转
                </button>
                <button onClick={() => sendCommand([-0.2, -0.2])}
                        className="px-3 py-1 bg-blue-500 text-sm text-white rounded hover:bg-blue-600">指令: 后退
                </button>
                <button onClick={() => resetCar()}
                        className="px-3 py-1 bg-green-500 text-sm text-white rounded hover:bg-green-600">复位
                </button>
                <span className="text-xs text-gray-600 ml-2">帧数: {collectedCount}</span>
            </div>
        </div>
    )
}
