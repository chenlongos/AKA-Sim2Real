import {useEffect, useRef, useImperativeHandle, forwardRef} from "react";
import type {CarState} from "../../models/types.ts";

export interface FirstPersonViewRef {
    getImageData: () => string | undefined;
}

interface FirstPersonViewProps {
    carState: CarState;
    isRecording: boolean;
    onCollect?: (imageData: string, actions: string[]) => void;
    getCurrentActions?: () => string[];
}

const MAP_W = 800;
const MAP_H = 600;

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

const drawFirstPerson = (ctx: CanvasRenderingContext2D, carState: CarState) => {
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
}

export const FirstPersonView = forwardRef<FirstPersonViewRef, FirstPersonViewProps>(({
    carState,
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

    useEffect(() => {
        const canvas = canvasRef.current
        if (canvas == null) return
        const ctx = canvas.getContext('2d')
        if (ctx == null) return

        ctx.imageSmoothingEnabled = false;

        let animationFrameId: number
        let lastCollectTime = 0;
        const COLLECT_INTERVAL = 500; // 采集间隔(ms)，2fps
        const FPS = 30;
        const frameInterval = 1000 / FPS;
        let lastTime = 0;

        const renderLoop = (currentTime: number) => {
            animationFrameId = window.requestAnimationFrame(renderLoop)

            const delta = currentTime - lastTime
            if (delta < frameInterval) return

            lastTime = currentTime - (delta % frameInterval)

            // 渲染
            drawFirstPerson(ctx, carState)

            // 采集数据
            if (isRecording && onCollect && getCurrentActions && currentTime - lastCollectTime >= COLLECT_INTERVAL) {
                const imageData = canvas.toDataURL('image/jpeg', 0.8)
                const actions = getCurrentActions()
                onCollect(imageData, actions)
                lastCollectTime = currentTime
            }
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.cancelAnimationFrame(animationFrameId)
        }
    }, [carState, isRecording, onCollect, getCurrentActions])

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
                <div>速度: {carState.speed.toFixed(2)}</div>
            </div>
        </div>
    )
})
