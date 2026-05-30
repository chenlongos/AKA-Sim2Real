import { useState, useCallback } from "react";
import type { MjData } from "@mujoco/mujoco";

interface Props {
  isLoaded: boolean;
  setControl: (name: string, value: number) => void;
  reset: () => void;
  data: React.RefObject<MjData | null>;
  showJointOverlay: boolean;
  setShowJointOverlay: (v: boolean) => void;
}

export function ControlPanel({ isLoaded, setControl, reset, showJointOverlay, setShowJointOverlay }: Props) {
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [roll, setRoll] = useState(0);
  const [leftWheel, setLeftWheel] = useState(0);
  const [rightWheel, setRightWheel] = useState(0);

  const applyArm = useCallback(() => {
    setControl("motor_yaw", yaw * 50);
    setControl("motor_pitch", pitch * 50);
    setControl("motor_roll", roll * 30);
  }, [yaw, pitch, roll, setControl]);

  const applyWheels = useCallback(() => {
    setControl("motor_wheel_fl", leftWheel);
    setControl("motor_wheel_rl", leftWheel);
    setControl("motor_wheel_fr", rightWheel);
    setControl("motor_wheel_rr", rightWheel);
  }, [leftWheel, rightWheel, setControl]);

  if (!isLoaded) return null;

  return (
    <div className="space-y-4">
      <div className="bg-slate-800/50 rounded-lg p-3">
        <h3 className="text-sm font-medium text-slate-300 mb-3">
          Arm Control
        </h3>
        {[
          { label: "Yaw", value: yaw, set: setYaw, min: -1, max: 1, step: 0.1 },
          {
            label: "Pitch",
            value: pitch,
            set: setPitch,
            min: -1,
            max: 1,
            step: 0.1,
          },
          {
            label: "Roll",
            value: roll,
            set: setRoll,
            min: -1,
            max: 1,
            step: 0.1,
          },
        ].map(({ label, value, set, min, max, step }) => (
          <div key={label} className="mb-2">
            <label className="block text-xs text-slate-400 mb-1">
              {label}: {value.toFixed(2)}
            </label>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={value}
              onChange={(e) => set(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
            />
          </div>
        ))}
        <button
          onClick={applyArm}
          className="w-full px-3 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded text-sm transition-colors"
        >
          Apply Arm
        </button>
      </div>

      <div className="bg-slate-800/50 rounded-lg p-3">
        <h3 className="text-sm font-medium text-slate-300 mb-3">
          Car Control
        </h3>
        {[
          {
            label: "Left",
            value: leftWheel,
            set: setLeftWheel,
          },
          {
            label: "Right",
            value: rightWheel,
            set: setRightWheel,
          },
        ].map(({ label, value, set }) => (
          <div key={label} className="mb-2">
            <label className="block text-xs text-slate-400 mb-1">
              {label}: {value.toFixed(1)}
            </label>
            <input
              type="range"
              min={-10}
              max={10}
              step={0.5}
              value={value}
              onChange={(e) => set(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>
        ))}
        <button
          onClick={applyWheels}
          className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm transition-colors"
        >
          Apply Wheels
        </button>
      </div>

      <button
        onClick={reset}
        className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded text-sm transition-colors"
      >
        Reset
      </button>

      <label className="flex items-center gap-2 px-3 py-2 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-700/50 transition-colors">
        <input
          type="checkbox"
          checked={showJointOverlay}
          onChange={(e) => setShowJointOverlay(e.target.checked)}
          className="w-4 h-4 rounded accent-emerald-500 cursor-pointer"
        />
        <span className="text-sm text-slate-300 select-none">Show Joints</span>
      </label>
    </div>
  );
}
