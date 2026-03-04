import {useEffect, useRef} from "react";
import type {CarState} from "../../models/types.ts";

interface TopDownViewProps {
    carState: CarState;
    collectedCount: number;
    resetCar: () => void;
    sendCommand: (cmd: string) => void;
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

const drawTopDown = (ctx: CanvasRenderingContext2D, carState: CarState) => {
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
}

export const TopDownView = ({
    carState,
    collectedCount,
    resetCar,
    sendCommand
}: TopDownViewProps) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (canvas == null) return
        const ctx = canvas.getContext('2d')
        if (ctx == null) return

        let animationFrameId: number

        const renderLoop = () => {
            drawTopDown(ctx, carState)
            animationFrameId = window.requestAnimationFrame(renderLoop)
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.cancelAnimationFrame(animationFrameId)
        }
    }, [carState])

    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-3 h-full">
            <div className="font-semibold">俯视地图</div>
            <div className="relative flex justify-center">
                <canvas
                    ref={canvasRef}
                    width={MAP_W}
                    height={MAP_H}
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
    )
}
