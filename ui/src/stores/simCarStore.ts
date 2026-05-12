import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CarState } from "../models/types.ts";

// 运动学常量 (vel_left/vel_right 单位是 m/s)
const ANGULAR_SCALE = 0.01;
const MS_TO_PIXELS = 10;  // m/s 转 像素/帧
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
    userId: string;
    currentEpisode: number;
    carState: CarState;
    setCarState: (carState: CarState) => void;
    resetCarState: () => void;
    setTargetVelocity: (velLeft: number, velRight: number) => void;
    applyPhysics: () => void;
    getCarState: () => CarState;
}

const generateUserId = () => {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const useSimCarStore = create<SimCarStore>()(
    persist(
        (set, get) => ({
            userId: generateUserId(),
            currentEpisode: 1,
            carState: initialSimCarState,
            setCarState: (carState) => set({ carState }),
            resetCarState: () => set({ carState: initialSimCarState }),
            getCarState: () => get().carState,

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

                // 摩擦力已禁用，速度保持不变
                const finalVelLeft = velLeftMs;
                const finalVelRight = velRightMs;

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
                        vel_left: finalVelLeft,
                        vel_right: finalVelRight,
                    },
                });
            },
        }),
        {
            name: "sim-car-storage",
            partialize: (state) => ({
                userId: state.userId,
                currentEpisode: state.currentEpisode,
            }),
        }
    )
);
