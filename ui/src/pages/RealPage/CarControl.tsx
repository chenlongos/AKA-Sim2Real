import {useEffect, useRef} from "react";
import type {CarState} from "../../models/types.ts";

interface CarControlProps {
    carState: CarState;
    isRecording: boolean;
    getCurrentActions: () => string[];
    onCollect?: (imageData: string, actions: string[]) => void;
}

export const CarControl = ({
    carState,
    isRecording,
    getCurrentActions,
    onCollect,
}: CarControlProps) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const lastCollectTimeRef = useRef(0)

    // 实时显示摄像头画面（模拟画面，实际真车会替换为真实摄像头）
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        ctx.fillStyle = '#1a1a2e'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        // 显示提示文字
        ctx.fillStyle = '#00ff88'
        ctx.font = '14px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('Real Camera Feed', canvas.width / 2, canvas.height / 2 - 10)
        ctx.fillText(`${carState.vel_left.toFixed(3)} | ${carState.vel_right.toFixed(3)}`, canvas.width / 2, canvas.height / 2 + 10)

        // 采集数据
        if (isRecording && onCollect && getCurrentActions) {
            const now = Date.now()
            if (now - lastCollectTimeRef.current >= 100) {
                const imageData = canvas.toDataURL('image/jpeg', 0.8)
                const actions = getCurrentActions()
                onCollect(imageData, actions)
                lastCollectTimeRef.current = now
            }
        }
    }, [carState, isRecording, onCollect, getCurrentActions])

    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-3 h-full text-gray-800 overflow-y-auto min-h-0">
            <div className="font-semibold">小车控制 & 状态</div>

            {/* 摄像头画面占位 */}
            <canvas
                ref={canvasRef}
                width={320}
                height={240}
                className="bg-black border-2 border-gray-800 rounded self-center"
            />

            {/* 速度显示 */}
            <div className="border border-gray-300 rounded p-2">
                <div className="text-xs font-semibold mb-2">实时速度</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-blue-100 rounded p-1.5 text-center">
                        <div className="text-gray-500">左轮</div>
                        <div className="font-mono font-bold text-blue-700">{carState.vel_left.toFixed(3)}</div>
                        <div className="text-gray-400">m/s</div>
                    </div>
                    <div className="bg-green-100 rounded p-1.5 text-center">
                        <div className="text-gray-500">右轮</div>
                        <div className="font-mono font-bold text-green-700">{carState.vel_right.toFixed(3)}</div>
                        <div className="text-gray-400">m/s</div>
                    </div>
                </div>
            </div>

            {/* 控制说明 */}
            <div className="border border-gray-300 rounded p-2">
                <div className="text-xs font-semibold mb-2">控制说明</div>
                <div className="text-xs text-gray-600 space-y-1">
                    <div>↑ / W - 前进</div>
                    <div>↓ / S - 后退</div>
                    <div>← / A - 左转</div>
                    <div>→ / D - 右转</div>
                    <div className="pt-1 border-t border-gray-200">
                        当前动作: {getCurrentActions().join(', ') || '无'}
                    </div>
                </div>
            </div>

            {/* 录制状态 */}
            <div className="text-xs text-gray-500">
                状态来源: 后端实时同步
            </div>
        </div>
    )
}