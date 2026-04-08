import {forwardRef, useEffect, useImperativeHandle, useRef} from "react";
import type {CarState, Obstacle} from "../../models/types.ts";
import {LogConsole} from "./LogConsole.tsx";
import {useSimCarStore} from "../../stores/simCarStore.ts";

export interface RightPanelRef {
    getImageData: () => string | undefined;
}

interface RightPanelProps {
    obstacles: Obstacle[];
    isRecording: boolean;
    collectionFps: number;
    onCollect?: (imageData: string) => void;
}

const MAP_W = 800;
const MAP_H = 600;

// 将障碍物转换为边界线段
const getObstacleBoundaries = (obstacles: Obstacle[]) => {
    return obstacles.flatMap(obs => [
        {x1: obs.x - obs.width / 2, y1: obs.y - obs.height / 2, x2: obs.x + obs.width / 2, y2: obs.y - obs.height / 2, color: '#e74c3c'}, // 上边
        {x1: obs.x + obs.width / 2, y1: obs.y - obs.height / 2, x2: obs.x + obs.width / 2, y2: obs.y + obs.height / 2, color: '#e74c3c'}, // 右边
        {x1: obs.x + obs.width / 2, y1: obs.y + obs.height / 2, x2: obs.x - obs.width / 2, y2: obs.y + obs.height / 2, color: '#e74c3c'}, // 下边
        {x1: obs.x - obs.width / 2, y1: obs.y + obs.height / 2, x2: obs.x - obs.width / 2, y2: obs.y - obs.height / 2, color: '#e74c3c'}, // 左边
    ]);
};

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

const castRay = (sx: number, sy: number, angle: number, obstacles: Obstacle[]) => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    let minDist = Infinity;
    let hitColor = null;

    // 地图边界 + 障碍物边界
    const boundaries = [
        {x1: 0, y1: 0, x2: MAP_W, y2: 0, color: '#333'}, // 上墙
        {x1: MAP_W, y1: 0, x2: MAP_W, y2: MAP_H, color: '#333'}, // 右墙
        {x1: MAP_W, y1: MAP_H, x2: 0, y2: MAP_H, color: '#333'}, // 下墙
        {x1: 0, y1: MAP_H, x2: 0, y2: 0, color: '#333'},  // 左墙
        ...getObstacleBoundaries(obstacles),  // 障碍物
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

const drawFirstPerson = (ctx: CanvasRenderingContext2D, carState: CarState, obstacles: Obstacle[]) => {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    const {x, y, angle} = carState;

    // 天空和地面 - 更亮的色调
    ctx.fillStyle = '#4a90d9'; // 天空 - 蓝色
    ctx.fillRect(0, 0, w, h / 2);
    ctx.fillStyle = '#5d6d7e'; // 地面 - 浅灰
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
        const hit = castRay(x, y, rayAngle, obstacles);

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
                drawFirstPerson(ctx, carState, obstacles);

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
