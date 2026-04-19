import {forwardRef, useEffect, useImperativeHandle, useRef} from "react";
import type {CarState, Obstacle} from "../../models/types.ts";
import {drawFirstPerson} from "./utils/canvasRenderer.ts";

export interface FirstPersonViewRef {
    getImageData: () => string | undefined;
}

interface FirstPersonViewProps {
    carState: CarState;
    obstacles: Obstacle[];
    isRecording: boolean;
    onCollect?: (imageData: string, actions: string[]) => void;
    getCurrentActions?: () => string[];
}

export const FirstPersonView = forwardRef<FirstPersonViewRef, FirstPersonViewProps>(({
    carState,
    obstacles,
    isRecording,
    onCollect,
    getCurrentActions
}, ref) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            const canvas = canvasRef.current
            if (!canvas) return undefined
            return canvas.toDataURL('image/jpeg', 0.8)
        }
    }), [])

    const lastCollectTimeRef = useRef(0)

    useEffect(() => {
        const canvas = canvasRef.current
        if (canvas == null) return
        const ctx = canvas.getContext('2d')
        if (ctx == null) return

        ctx.imageSmoothingEnabled = false;

        let animationFrameId: number
        const COLLECT_INTERVAL = 1000 / 30; // 采集间隔(ms)，30fps
        const FPS = 30;
        const frameInterval = 1000 / FPS;
        let lastTime = 0;

        const renderLoop = (currentTime: number) => {
            animationFrameId = window.requestAnimationFrame(renderLoop)

            if (currentTime - lastTime < frameInterval) return

            lastTime = currentTime

            // 渲染
            drawFirstPerson(ctx, carState, obstacles)

            // 采集数据
            if (isRecording && onCollect && getCurrentActions && currentTime - lastCollectTimeRef.current >= COLLECT_INTERVAL) {
                const imageData = canvas.toDataURL('image/jpeg', 0.8)
                const actions = getCurrentActions()
                onCollect(imageData, actions)
                lastCollectTimeRef.current = currentTime
            }
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.cancelAnimationFrame(animationFrameId)
        }
    }, [carState, obstacles, isRecording, onCollect, getCurrentActions])

    return (
        <div
            className="border-2 border-gray-800 rounded-lg bg-gray-100 p-2.5 flex flex-col gap-2 h-full text-gray-800 overflow-y-auto min-h-0">
            <div className="font-semibold">车载摄像头</div>
            <canvas ref={canvasRef} width={320} height={240}
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
                <div>左轮: {carState.vel_left.toFixed(3)} m/s</div>
                <div>右轮: {carState.vel_right.toFixed(3)} m/s</div>
            </div>
        </div>
    )
})
