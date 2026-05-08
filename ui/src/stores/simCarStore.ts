import { create } from "zustand";
import type { CarState } from "../models/types.ts";

// 运动学常量 (vel_left/vel_right 单位是 m/s)
const ANGULAR_SCALE = 0.01;
const FRICTION = 0.98;
const MAP_WIDTH = 800;
const MAP_HEIGHT = 600;

export const initialSimCarState: CarState = {
    x: 400,
    y: 300,
    angle: -Math.PI / 2,
    vel_left: 0,
    vel_right: 0,
};

interface SimCarStore {
    carState: CarState;
    setCarState: (carState: CarState) => void;
    resetCarState: () => void;
    tick: (action: [number, number]) => void;
    applyFriction: () => void;
    getCarState: () => CarState;
}

export const useSimCarStore = create<SimCarStore>((set, get) => ({
    carState: initialSimCarState,
    setCarState: (carState) => set({ carState }),
    resetCarState: () => set({ carState: initialSimCarState }),
    getCarState: () => get().carState,

    tick: (action: [number, number]) => {
        const state = get().carState;

        // vel_left/vel_right 存的是 m/s 单位，和训练数据一致
        // 用于发送给后端推理，以及采集时记录状态
        const velLeftMs = action[0];
        const velRightMs = action[1];

        // 限制速度范围 (m/s)
        const clampedVelLeftMs = Math.max(-0.2, Math.min(0.2, velLeftMs));
        const clampedVelRightMs = Math.max(-0.2, Math.min(0.2, velRightMs));

        // 运动学计算时转换为像素/帧
        const velLeftPx = clampedVelLeftMs * 100;
        const velRightPx = clampedVelRightMs * 100;

        // 差速轮运动学
        const linearVel = (velLeftPx + velRightPx) / 2;
        const angularVel = (velLeftPx - velRightPx) * ANGULAR_SCALE;

        let x = state.x + Math.cos(state.angle) * linearVel;
        let y = state.y + Math.sin(state.angle) * linearVel;
        let angle = state.angle + angularVel;

        // 角度归一化到 [-π, π]
        angle = Math.atan2(Math.sin(angle), Math.cos(angle));

        // 边界检测
        x = Math.max(20, Math.min(MAP_WIDTH - 20, x));
        y = Math.max(20, Math.min(MAP_HEIGHT - 20, y));

        set({
            carState: {
                ...state,
                x,
                y,
                angle,
                vel_left: clampedVelLeftMs,
                vel_right: clampedVelRightMs,
            },
        });
    },

    applyFriction: () => {
        const state = get().carState;

        // vel_left/vel_right 存的是 m/s
        let velLeftMs = state.vel_left * FRICTION;
        let velRightMs = state.vel_right * FRICTION;

        // 停止时清零
        if (Math.abs(velLeftMs) < 0.001 && Math.abs(velRightMs) < 0.001) {
            velLeftMs = 0;
            velRightMs = 0;
        }

        // 运动学计算时转换为像素/帧
        const velLeftPx = velLeftMs * 100;
        const velRightPx = velRightMs * 100;

        // 差速轮运动学
        const linearVel = (velLeftPx + velRightPx) / 2;
        const angularVel = (velLeftPx - velRightPx) * ANGULAR_SCALE;

        let x = state.x + Math.cos(state.angle) * linearVel;
        let y = state.y + Math.sin(state.angle) * linearVel;
        let angle = state.angle + angularVel;

        // 角度归一化到 [-π, π]
        angle = Math.atan2(Math.sin(angle), Math.cos(angle));

        // 边界检测
        x = Math.max(20, Math.min(MAP_WIDTH - 20, x));
        y = Math.max(20, Math.min(MAP_HEIGHT - 20, y));

        set({
            carState: {
                ...state,
                x,
                y,
                angle,
                vel_left: velLeftMs,
                vel_right: velRightMs,
            },
        });
    },
}));
