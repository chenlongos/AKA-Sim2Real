import {forwardRef, useEffect, useImperativeHandle, useRef} from "react";
import type {Obstacle} from "../../models/types.ts";
import {LogConsole} from "./LogConsole.tsx";
import {useSimCarStore} from "../../stores/simCarStore.ts";
import {drawFirstPerson} from "./utils/canvasRenderer.ts";

export interface RightPanelRef {
    getImageData: () => string | undefined;
}

interface RightPanelProps {
    obstacles: Obstacle[];
    isRecording: boolean;
    collectionFps: number;
    onCollect?: (imageData: string) => void;
}

const RIGHT_PANEL_COLORS = {
    sky: '#4a90d9',
    ground: '#5d6d7e',
};

export const RightPanel = forwardRef<RightPanelRef, RightPanelProps>(({
    obstacles,
    isRecording,
    collectionFps,
    onCollect
}, ref) => {
    const carState = useSimCarStore((state) => state.carState)
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            const canvas = canvasRef.current;
            if (!canvas) return undefined;
            return canvas.toDataURL('image/jpeg', 0.8);
        }
    }), []);

    const lastCollectTimeRef = useRef(0);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas == null) return;
        const ctx = canvas.getContext('2d');
        if (ctx == null) return;

        ctx.imageSmoothingEnabled = false;

        let animationFrameId: number;
        const collectInterval = 1000 / Math.max(collectionFps, 1);
        const FPS = 30;
        const frameInterval = 1000 / FPS;
        let lastTime = 0;

        const renderLoop = (currentTime: number) => {
            if (currentTime - lastTime >= frameInterval) {
                lastTime = currentTime;

                // 渲染
                drawFirstPerson(ctx, carState, obstacles, {colors: RIGHT_PANEL_COLORS});

                // 采集数据
                if (isRecording && onCollect && currentTime - lastCollectTimeRef.current >= collectInterval) {
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);
                    onCollect(imageData);
                    lastCollectTimeRef.current = currentTime;
                }
            }

            animationFrameId = window.requestAnimationFrame(renderLoop);
        };

        animationFrameId = window.requestAnimationFrame(renderLoop);

        return () => {
            window.cancelAnimationFrame(animationFrameId);
        };
    }, [carState, obstacles, isRecording, collectionFps, onCollect]);

    return (
        <div className="flex flex-col h-full gap-3">
            {/* 第一视角卡片 */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                        <span className="w-2 h-2 bg-blue-500 rounded-full"/>
                        车载摄像头
                    </h3>
                    {isRecording && (
                        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-red-500/20 text-red-400 border border-red-500/30">
                            <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"/>
                            <span className="text-xs font-medium">REC</span>
                        </div>
                    )}
                </div>

                <canvas
                    ref={canvasRef}
                    width={320}
                    height={240}
                    className="w-full bg-black border border-slate-700 rounded-lg shadow-lg"
                />

                {/* 状态信息 */}
                <div className="mt-3 pt-3 border-t border-slate-700/50">
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                        <div className="text-slate-500">位置</div>
                        <div className="text-slate-300 font-mono">
                            X: {carState.x.toFixed(1)}, Y: {carState.y.toFixed(1)}
                        </div>
                        <div className="text-slate-500">角度</div>
                        <div className="text-slate-300 font-mono">
                            {(carState.angle * 180 / Math.PI).toFixed(1)}°
                        </div>
                        <div className="text-slate-500">轮速</div>
                        <div className="text-slate-300 font-mono">
                            L: {carState.vel_left.toFixed(3)}, R: {carState.vel_right.toFixed(3)}
                        </div>
                    </div>
                </div>
            </div>

            {/* 日志控制台 */}
            <div className="flex-1 min-h-0">
                <LogConsole className="h-full"/>
            </div>
        </div>
    );
});
