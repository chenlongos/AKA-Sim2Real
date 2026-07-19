interface Props {
  isLoaded: boolean;
  reset: () => void;
  showJointOverlay: boolean;
  setShowJointOverlay: (v: boolean) => void;
  driveSpeed: number;
  turnSpeed: number;
  onDriveSpeedChange: (v: number) => void;
  onTurnSpeedChange: (v: number) => void;
}

const DRIVE_SPEED_MIN = 0.5;
const DRIVE_SPEED_MAX = 5;
const DRIVE_SPEED_STEP = 0.1;

export function ControlPanel({
  isLoaded,
  reset,
  showJointOverlay,
  setShowJointOverlay,
  driveSpeed,
  turnSpeed,
  onDriveSpeedChange,
  onTurnSpeedChange,
}: Props) {
  if (!isLoaded) return null;

  return (
    <div className="space-y-4">
      <div className="bg-slate-800/50 rounded-lg p-3">
        <h3 className="text-sm font-medium text-slate-300 mb-3">Drive Settings</h3>
        <div className="mb-2">
          <label className="block text-xs text-slate-400 mb-1">
            Speed: {driveSpeed.toFixed(1)}
          </label>
          <input
            type="range"
            min={DRIVE_SPEED_MIN}
            max={DRIVE_SPEED_MAX}
            step={DRIVE_SPEED_STEP}
            value={driveSpeed}
            onChange={(e) => onDriveSpeedChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
        </div>
        <div className="mb-2">
          <label className="block text-xs text-slate-400 mb-1">
            Turn: {turnSpeed.toFixed(1)}
          </label>
          <input
            type="range" min={0.5} max={8} step={0.5}
            value={turnSpeed}
            onChange={(e) => onTurnSpeedChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
        </div>
      </div>

      <button
        onClick={reset}
        className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded text-sm transition-colors"
      >
        Reset Car
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
