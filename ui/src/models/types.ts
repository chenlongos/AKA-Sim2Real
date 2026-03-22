// 小车状态类型
export interface CarState {
    x: number;
    y: number;
    angle: number;
    vel_left: number;
    vel_right: number;
}

// 障碍物类型
export interface Obstacle {
    x: number;
    y: number;
    width: number;
    height: number;
}
