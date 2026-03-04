// 小车状态类型
export interface CarState {
    x: number;
    y: number;
    angle: number;
    speed: number;
}

// 障碍物类型
export interface Obstacle {
    x: number;
    y: number;
    width: number;
    height: number;
}
