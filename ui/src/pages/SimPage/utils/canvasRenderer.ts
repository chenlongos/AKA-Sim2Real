import type {CarState, Obstacle} from "../../../models/types.ts";

export const MAP_WIDTH = 800;
export const MAP_HEIGHT = 600;

export interface WallSegment {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    color: string;
}

export interface RayHit {
    distance: number;
    color: string;
}

export interface FirstPersonColors {
    sky: string;
    ground: string;
}

export const DEFAULT_FIRST_PERSON_COLORS: FirstPersonColors = {
    sky: '#87CEEB',
    ground: '#7f8c8d',
};

export const getObstacleBoundaries = (obstacles: Obstacle[]): WallSegment[] => {
    return obstacles.flatMap(obs => [
        {
            x1: obs.x - obs.width / 2,
            y1: obs.y - obs.height / 2,
            x2: obs.x + obs.width / 2,
            y2: obs.y - obs.height / 2,
            color: '#e74c3c',
        }, // 上边
        {
            x1: obs.x + obs.width / 2,
            y1: obs.y - obs.height / 2,
            x2: obs.x + obs.width / 2,
            y2: obs.y + obs.height / 2,
            color: '#e74c3c',
        }, // 右边
        {
            x1: obs.x + obs.width / 2,
            y1: obs.y + obs.height / 2,
            x2: obs.x - obs.width / 2,
            y2: obs.y + obs.height / 2,
            color: '#e74c3c',
        }, // 下边
        {
            x1: obs.x - obs.width / 2,
            y1: obs.y + obs.height / 2,
            x2: obs.x - obs.width / 2,
            y2: obs.y - obs.height / 2,
            color: '#e74c3c',
        }, // 左边
    ]);
};

export const getRaySegmentIntersection = (
    rx: number,
    ry: number,
    rdx: number,
    rdy: number,
    wall: WallSegment,
): number | null => {
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

export const castRay = (
    sx: number,
    sy: number,
    angle: number,
    obstacles: Obstacle[],
): RayHit | null => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    let minDist = Infinity;
    let hitColor: string | null = null;

    // 地图边界 + 障碍物边界
    const boundaries: WallSegment[] = [
        {x1: 0, y1: 0, x2: MAP_WIDTH, y2: 0, color: '#333'}, // 上墙
        {x1: MAP_WIDTH, y1: 0, x2: MAP_WIDTH, y2: MAP_HEIGHT, color: '#333'}, // 右墙
        {x1: MAP_WIDTH, y1: MAP_HEIGHT, x2: 0, y2: MAP_HEIGHT, color: '#333'}, // 下墙
        {x1: 0, y1: MAP_HEIGHT, x2: 0, y2: 0, color: '#333'}, // 左墙
        ...getObstacleBoundaries(obstacles), // 障碍物
    ];

    // 检测射线与每一条线段的交点
    for (const wall of boundaries) {
        const dist = getRaySegmentIntersection(sx, sy, cos, sin, wall);
        if (dist !== null && dist < minDist) {
            minDist = dist;
            hitColor = wall.color;
        }
    }

    if (minDist === Infinity || hitColor === null) return null;
    return {distance: minDist, color: hitColor};
};

export interface DrawFirstPersonOptions {
    colors?: FirstPersonColors;
}

export const drawFirstPerson = (
    ctx: CanvasRenderingContext2D,
    carState: CarState,
    obstacles: Obstacle[],
    options: DrawFirstPersonOptions = {},
): void => {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    const {x, y, angle} = carState;
    const {sky = DEFAULT_FIRST_PERSON_COLORS.sky, ground = DEFAULT_FIRST_PERSON_COLORS.ground} =
        options;

    // 天空和地面
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h / 2);
    ctx.fillStyle = ground;
    ctx.fillRect(0, h / 2, w, h / 2);

    // 参数
    const fov = Math.PI / 3; // 60度视野
    const rayCount = w / 4; // 射线数量 (为了性能，每4个像素投射一条，然后画宽一点)
    const rayWidth = w / rayCount;

    // 遍历每一条射线
    for (let i = 0; i < rayCount; i++) {
        // 当前射线角度 = 车角度 - 半个FOV + 增量
        const rayAngle = angle + Math.PI - fov / 2 + (i / rayCount) * fov;

        // 计算这一条射线碰到了什么，以及距离是多少
        const hit = castRay(x, y, rayAngle, obstacles);

        if (hit) {
            const correctedDist = hit.distance * Math.cos(rayAngle - angle);

            const wallHeight = (h * 40) / correctedDist;

            ctx.fillStyle = hit.color;
            ctx.globalAlpha = Math.max(0.3, 1 - correctedDist / 600);
            ctx.fillRect(i * rayWidth, (h - wallHeight) / 2, rayWidth + 1, wallHeight);
            ctx.globalAlpha = 1.0;
        }
    }
};
