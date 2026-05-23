import { useMujoco } from "./useMujoco";
import MujocoRenderer from "./MujocoRenderer";
import { ControlPanel } from "./ControlPanel";

export default function MujocoPage() {
  const { isLoaded, mujoco, model, data, step, setControl, reset } =
    useMujoco();

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
            FPS: {isLoaded ? "60" : "--"}
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
            onStep={step}
          />
        </div>

        <div className="w-72 bg-slate-900/50 border-l border-slate-800 p-3 overflow-y-auto">
          <ControlPanel
            isLoaded={isLoaded}
            setControl={setControl}
            reset={reset}
            data={data}
          />
        </div>
      </div>
    </div>
  );
}
