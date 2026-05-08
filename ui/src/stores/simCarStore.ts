import { create } from "zustand";
import type { CarState } from "../models/types.ts";

// 运动学常量 (vel_left/vel_right 单位是 m/s)
const ANGULAR_SCALE = 0.01;
const FRICTION = 0.98;
const MS_TO_PIXELS = 100;  // m/s 转 像素/帧
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
    setTargetVelocity: (velLeft: number, velRight: number) => void;
    applyPhysics: () => void;
    getCarState: () => CarState;
}

export const useSimCarStore = create<SimCarStore>((set, get) => ({
    carState: initialSimCarState,
    setCarState: (carState) => set({ carState }),
    resetCarState: () => set({ carState: initialSimCarState }),
    getCarState: () => get().carState,

    // 设置目标速度（m/s），用于推理结果
    setTargetVelocity: (velLeft: number, velRight: number) => {
        const clampedVelLeft = Math.max(-0.2, Math.min(0.2, velLeft));
        const clampedVelRight = Math.max(-0.2, Math.min(0.2, velRight));
        set((state) => ({
            carState: {
                ...state.carState,
                vel_left: clampedVelLeft,
                vel_right: clampedVelRight,
            },
        }));
    },

    // 应用物理：一帧的运动学 + 摩擦力
    applyPhysics: () => {
        const state = get().carState;
        const velLeftMs = state.vel_left;
        const velRightMs = state.vel_right;

        // 转换为像素/帧
        const velLeftPx = velLeftMs * MS_TO_PIXELS;
        const velRightPx = velRightMs * MS_TO_PIXELS;

        // 差速轮运动学
        const linearVel = (velLeftPx + velRightPx) / 2;
        const angularVel = (velLeftPx - velRightPx) * ANGULAR_SCALE;

        let x = state.x + Math.cos(state.angle) * linearVel;
        let y = state.y + Math.sin(state.angle) * linearVel;
        let angle = state.angle + angularVel;

        // 应用摩擦力
        let newVelLeftMs = velLeftMs * FRICTION;
        let newVelRightMs = velRightMs * FRICTION;

        // 停止时清零
        if (Math.abs(newVelLeftMs) < 0.001 && Math.abs(newVelRightMs) < 0.001) {
            newVelLeftMs = 0;
            newVelRightMs = 0;
        }

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
                vel_left: newVelLeftMs,
                vel_right: newVelRightMs,
            },
        });
    },
}));
