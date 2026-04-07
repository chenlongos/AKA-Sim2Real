import { create } from "zustand";
import type { CarState } from "../models/types.ts";

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
}

export const useSimCarStore = create<SimCarStore>((set) => ({
    carState: initialSimCarState,
    setCarState: (carState) => set({ carState }),
    resetCarState: () => set({ carState: initialSimCarState }),
}));
