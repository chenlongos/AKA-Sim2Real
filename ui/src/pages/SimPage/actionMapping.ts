export const SIM_ACTION_TO_CONTINUOUS: Record<string, [number, number]> = {
    forward: [0.2, 0.2],
    backward: [-0.2, -0.2],
    left: [-0.2, 0.2],
    right: [0.2, -0.2],
    stop: [0, 0],
};

export const SIM_KEY_TO_ACTION: Record<string, string> = {
    ArrowUp: "forward",
    KeyW: "forward",
    ArrowDown: "backward",
    KeyS: "backward",
    ArrowLeft: "left",
    KeyA: "left",
    ArrowRight: "right",
    KeyD: "right",
};

// Re-export for backward compatibility
export const KEY_TO_ACTION = SIM_KEY_TO_ACTION;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

export const getContinuousActionFromDiscreteActions = (actions: string[]): [number, number] => {
    let left = 0;
    let right = 0;

    for (const action of actions) {
        const mapped = SIM_ACTION_TO_CONTINUOUS[action];
        if (!mapped) continue;
        left += mapped[0];
        right += mapped[1];
    }

    return [clamp(left, -0.2, 0.2), clamp(right, -0.2, 0.2)];
};
