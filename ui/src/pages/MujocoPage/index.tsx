import { useState, useEffect, useRef, useCallback } from "react";
import { useMujoco } from "./useMujoco";
import MujocoRenderer from "./MujocoRenderer";
import { ControlPanel } from "./ControlPanel";

const DRIVE_KEYS = [
  "KeyW", "KeyA", "KeyS", "KeyD",
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "Space",
];

export default function MujocoPage() {
  const { isLoaded, mujoco, model, data, step, setControl, reset } =
    useMujoco();
  const keysRef = useRef<Set<string>>(new Set());
  const [showJointOverlay, setShowJointOverlay] = useState(false);
  const [driveSpeed, setDriveSpeed] = useState(5);
  const [turnSpeed, setTurnSpeed] = useState(3);
  const [fps, setFps] = useState(0);
  const fpsFramesRef = useRef(0);
  const fpsLastTimeRef = useRef(performance.now());

  const applyDrive = useCallback(() => {
    let leftVel = 0;
    let rightVel = 0;

    const k = keysRef.current;

    if (k.has("KeyW") || k.has("ArrowUp")) {
      leftVel += driveSpeed;
      rightVel += driveSpeed;
    }
    if (k.has("KeyS") || k.has("ArrowDown")) {
      leftVel -= driveSpeed;
      rightVel -= driveSpeed;
    }
    if (k.has("KeyA") || k.has("ArrowLeft")) {
      leftVel -= turnSpeed;
      rightVel += turnSpeed;
    }
    if (k.has("KeyD") || k.has("ArrowRight")) {
      leftVel += turnSpeed;
      rightVel -= turnSpeed;
    }
    if (k.has("Space")) {
      leftVel = 0;
      rightVel = 0;
    }

    setControl("motor_wheel_fl", leftVel);
    setControl("motor_wheel_rl", leftVel);
    setControl("motor_wheel_fr", rightVel);
    setControl("motor_wheel_rr", rightVel);
  }, [setControl, driveSpeed, turnSpeed]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (DRIVE_KEYS.includes(e.code)) {
        e.preventDefault();
        e.stopPropagation();
        keysRef.current.add(e.code);
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      keysRef.current.delete(e.code);
    };
    // capture:true ensures we intercept before the browser scrolls
    window.addEventListener("keydown", handleKeyDown, { capture: true });
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown, { capture: true });
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // Apply drive controls every frame before mj_step, with real FPS
  const stepWithDrive = useCallback(() => {
    applyDrive();

    // Real FPS counter
    fpsFramesRef.current++;
    const now = performance.now();
    if (now - fpsLastTimeRef.current >= 1000) {
      setFps(fpsFramesRef.current);
      fpsFramesRef.current = 0;
      fpsLastTimeRef.current = now;
    }

    step();
  }, [applyDrive, step]);

  return (
    <div className="flex flex-col h-screen bg-slate-950 overflow-hidden">
      <div className="flex items-center justify-between px-6 py-2 bg-slate-900/50 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-linear-to-br from-violet-600 to-blue-600 flex items-center justify-center shadow-lg shadow-violet-900/20">
            <span className="text-white font-bold text-sm">MJC</span>
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">
              MuJoCo WASM + Three.js
            </h2>
            <p className="text-xs text-slate-500">Browser-native simulation</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-slate-500">
            FPS: {fps || "--"}
          </span>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700">
            <span
              className={`w-2 h-2 rounded-full ${isLoaded ? "bg-emerald-500" : "bg-yellow-500 animate-pulse"}`}
            />
            <span className="text-xs text-slate-300">
              {isLoaded ? "Running" : "Loading..."}
            </span>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 relative">
          <MujocoRenderer
            mujoco={mujoco}
            model={model}
            data={data}
            isLoaded={isLoaded}
            onStep={stepWithDrive}
            showJointOverlay={showJointOverlay}
          />
        </div>

        <div className="w-72 bg-slate-900/50 border-l border-slate-800 p-3 overflow-y-auto">
          <ControlPanel
            isLoaded={isLoaded}
            reset={reset}
            showJointOverlay={showJointOverlay}
            setShowJointOverlay={setShowJointOverlay}
            driveSpeed={driveSpeed}
            turnSpeed={turnSpeed}
            onDriveSpeedChange={setDriveSpeed}
            onTurnSpeedChange={setTurnSpeed}
          />
          <p className="text-xs text-slate-500 mt-3 text-center">
            WASD / Arrows: drive · Space: brake
          </p>
        </div>
      </div>
    </div>
  );
}
