import { useState, useEffect, useRef, useCallback } from "react";
import { startTraining, stopTraining, getTrainingStatus, loadTrainedModel, listDatasetDirs, listModels, getDatasetInfo, type TrainingStatus } from "../../api/api";
import { runInferenceWithSocket, simSocket } from "../../api/socket";
import { getDatasetPath, getTrainPath } from "../../lib/constants";
import { useSimCarStore } from "../../stores/simCarStore";
import { useMujoco, type CarToBallState, type MjPosition } from "./useMujoco";
import { useDataCollection } from "./useDataCollection";
import MujocoRenderer from "./MujocoRenderer";

const DRIVE_KEYS = [
  "KeyW", "KeyA", "KeyS", "KeyD",
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "Space",
];

type TaskMode = "follow" | "avoid";
type ControlMode = "manual" | "auto" | "model";
type ModelTaskRunState = {
  startedAt: number;
  followCompleteSince: number | null;
  avoidDangerSeen: boolean;
  avoidSafeSince: number | null;
};

const TASK_DATASET_NAMES: Record<TaskMode, string> = {
  follow: "mujoco-follow",
  avoid: "mujoco-avoid",
};
const MANAGED_TASK_DATASET_NAMES = new Set(["mujoco-default", ...Object.values(TASK_DATASET_NAMES)]);

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const GROUND_GRID_HALF_SIZE = 10;
const BALL_GRID_MARGIN = 0.5;
const BALL_GRID_LIMIT = GROUND_GRID_HALF_SIZE - BALL_GRID_MARGIN;
const AVOID_CAR_GRID_LIMIT = GROUND_GRID_HALF_SIZE - 1.1;
const DRIVE_SPEED_MIN = 0.5;
const DRIVE_SPEED_MAX = 5;
const DRIVE_SPEED_STEP = 0.1;
const DEFAULT_DRIVE_SPEED = 0.7;
const TURN_SPEED_MIN = 0.5;
const TURN_SPEED_MAX = 8;
const TURN_SPEED_STEP = 0.5;
const DEFAULT_TURN_SPEED = 4;
const MOTOR_CONTROL_LIMIT = 8;
const TRAINING_BALL_EDGE_BUFFER = 0.2;
const BALL_PLACEMENT_ATTEMPTS = 40;
const AUTO_BALL_RESET_COOLDOWN_MS = 700;
const BALL_APPROACH_DISTANCE = 1.4;
const MODEL_INFERENCE_INTERVAL_MS = 100;
const MODEL_TASK_TIMEOUT_MS = 15000;
const FOLLOW_TASK_COMPLETE_DISTANCE = BALL_APPROACH_DISTANCE;
const FOLLOW_TASK_COMPLETE_ANGLE = 0.65;
const FOLLOW_TASK_COMPLETE_STABLE_MS = 400;
const AVOID_TARGET_DISTANCE_MIN = 5.5;
const AVOID_TARGET_DISTANCE_MAX = 8;
const AVOID_TRIGGER_DISTANCE = 3;
const AVOID_TASK_MIN_DURATION_MS = 700;
const AVOID_TASK_SAFE_DISTANCE = AVOID_TRIGGER_DISTANCE + 0.8;
const AVOID_TASK_SAFE_STABLE_MS = 700;
const AVOID_VISIBLE_FRONTNESS = 0.25;
const AVOID_VISIBLE_ANGLE = 1.2;
const AVOID_RELOCATE_MIN_SETTLE_MS = 1800;
const AVOID_RELOCATE_MAX_SETTLE_MS = 4200;
const AVOID_HEADING_SAMPLE_MS = 180;
const AVOID_HEADING_STABLE_MS = 550;
const AVOID_HEADING_STABLE_RATE = 0.25;
const AVOID_MISPLACED_RETRY_MS = 1600;
const AVOID_EDGE_MARGIN = 1.0;
const AVOID_EDGE_RELEASE_MARGIN = 1.4;
const AVOID_EDGE_LOOKAHEAD_DISTANCE = 2.2;
const AVOID_EDGE_CENTER_ANGLE = 0.45;
const AVOID_EDGE_TURN_GAIN = 1.7;
const AVOID_EDGE_TURN_LIMIT = 1.35;
const AVOID_EDGE_FORWARD_SPEED_SCALE = 0.35;
const BALL_RELOCATION_MIN_DISTANCE = 3.2;
const FOLLOW_VISIBLE_ANGLES = [-0.72, -0.45, -0.22, 0, 0.22, 0.45, 0.72];
const AVOID_VISIBLE_ANGLES = [-0.58, -0.3, 0, 0.3, 0.58];
const AVOID_AHEAD_ANGLES = [0, -0.18, 0.18, -0.35, 0.35, -0.52, 0.52];

function pickNumber(values: number[]) {
  return values[Math.floor(Math.random() * values.length)];
}

function pickNumberInRange(min: number, max: number) {
  return min + Math.random() * (max - min);
}

function roundCoordinate(value: number) {
  return Number(value.toFixed(2));
}

function clampCoordinateToGrid(value: number) {
  return clamp(value, -BALL_GRID_LIMIT, BALL_GRID_LIMIT);
}

function clampBallPositionToGrid(position: MjPosition): MjPosition {
  return {
    x: roundCoordinate(clampCoordinateToGrid(position.x)),
    y: roundCoordinate(clampCoordinateToGrid(position.y)),
    z: position.z,
  };
}

function getPlanarDistance(from: MjPosition, to: MjPosition) {
  return Math.hypot(from.x - to.x, from.y - to.y);
}

function getGridEdgeMargin(position: MjPosition, gridLimit: number) {
  return Math.min(gridLimit - Math.abs(position.x), gridLimit - Math.abs(position.y));
}

function getSmallestAngleDelta(from: number, to: number) {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}

function isAvoidBallInView(state: CarToBallState) {
  return state.frontness > AVOID_VISIBLE_FRONTNESS && Math.abs(state.angleError) < AVOID_VISIBLE_ANGLE;
}

function createRandomBallPosition(label = "随机位置") {
  const position = clampBallPositionToGrid({
    x: roundCoordinate(Math.random() * 8 - 4),
    y: roundCoordinate(Math.random() * 9 - 2),
    z: 0.25,
  });
  return { label, ...position };
}

function getTrainingBallDistanceRange(taskMode: TaskMode) {
  return taskMode === "follow"
    ? { min: 2.3, max: 5.5 }
    : { min: AVOID_TARGET_DISTANCE_MIN, max: AVOID_TARGET_DISTANCE_MAX };
}

function getRayDistanceToGridBounds(origin: MjPosition, angle: number, gridLimit: number) {
  const worldMin = -gridLimit;
  const worldMax = gridLimit;
  const directionX = Math.cos(angle);
  const directionY = Math.sin(angle);
  const distances: number[] = [];

  if (directionX > 0) distances.push((worldMax - origin.x) / directionX);
  if (directionX < 0) distances.push((worldMin - origin.x) / directionX);
  if (directionY > 0) distances.push((worldMax - origin.y) / directionY);
  if (directionY < 0) distances.push((worldMin - origin.y) / directionY);

  const positiveDistances = distances.filter((distance) => Number.isFinite(distance) && distance > 0);
  return positiveDistances.length > 0 ? Math.min(...positiveDistances) : 0;
}

function getRayDistanceToTrainingBounds(origin: MjPosition, angle: number) {
  return getRayDistanceToGridBounds(origin, angle, BALL_GRID_LIMIT);
}

function createBallPositionFromRay(origin: MjPosition, angle: number, distance: number): MjPosition {
  return clampBallPositionToGrid({
    x: roundCoordinate(origin.x + Math.cos(angle) * distance),
    y: roundCoordinate(origin.y + Math.sin(angle) * distance),
    z: 0.25,
  });
}

function createFallbackVisibleBallPosition() {
  return clampBallPositionToGrid({
    x: roundCoordinate(Math.random() * 5 - 2.5),
    y: roundCoordinate(Math.random() * 3 - 1),
    z: 0.25,
  });
}

function createVisibleBallPosition(state: CarToBallState | null, taskMode: TaskMode): MjPosition {
  if (!state) {
    return createFallbackVisibleBallPosition();
  }

  const angles = taskMode === "follow" ? FOLLOW_VISIBLE_ANGLES : AVOID_VISIBLE_ANGLES;
  const distanceRange = getTrainingBallDistanceRange(taskMode);
  let farthestPosition: MjPosition | null = null;
  let farthestDistance = 0;

  for (let attempt = 0; attempt < BALL_PLACEMENT_ATTEMPTS; attempt++) {
    const relativeAngle = pickNumber(angles) + (Math.random() * 0.12 - 0.06);
    const worldAngle = state.headingAngle + relativeAngle;
    const maxDistance = getRayDistanceToTrainingBounds(state.car, worldAngle) - TRAINING_BALL_EDGE_BUFFER;
    if (taskMode === "follow" && maxDistance < distanceRange.min) continue;
    if (taskMode === "avoid" && maxDistance < 1.5) continue;

    const distanceLimit = Math.min(distanceRange.max, maxDistance);
    const distance = taskMode === "avoid"
      ? distanceLimit
      : distanceRange.min + Math.random() * (distanceLimit - distanceRange.min);
    const position = createBallPositionFromRay(state.car, worldAngle, distance);
    const relocationDistance = getPlanarDistance(position, state.ball);
    if (relocationDistance > farthestDistance) {
      farthestPosition = position;
      farthestDistance = relocationDistance;
    }
    if (relocationDistance >= BALL_RELOCATION_MIN_DISTANCE) {
      return position;
    }
  }

  if (farthestPosition) return farthestPosition;

  const fallbackAngle = Math.atan2(-state.car.y, -state.car.x);
  const fallbackDistance = Math.min(distanceRange.min, Math.max(1.5, getRayDistanceToTrainingBounds(state.car, fallbackAngle) - TRAINING_BALL_EDGE_BUFFER));
  return createBallPositionFromRay(state.car, fallbackAngle, fallbackDistance);
}

function createAvoidBallPositionAhead(state: CarToBallState | null): MjPosition {
  if (!state) {
    return createFallbackVisibleBallPosition();
  }

  const desiredDistance = pickNumberInRange(AVOID_TARGET_DISTANCE_MIN, AVOID_TARGET_DISTANCE_MAX);
  const minimumUsefulDistance = Math.max(AVOID_TRIGGER_DISTANCE + 0.8, desiredDistance * 0.7);
  let bestInGridPosition: MjPosition | null = null;
  let bestInGridDistance = 0;

  for (const relativeAngle of AVOID_AHEAD_ANGLES) {
    const worldAngle = state.headingAngle + relativeAngle;
    const maxDistance = getRayDistanceToTrainingBounds(state.car, worldAngle) - TRAINING_BALL_EDGE_BUFFER;
    if (maxDistance < 1.5) continue;

    const distance = Math.min(desiredDistance, maxDistance);
    const position = createBallPositionFromRay(state.car, worldAngle, distance);
    if (distance > bestInGridDistance) {
      bestInGridPosition = position;
      bestInGridDistance = distance;
    }
    if (distance >= minimumUsefulDistance) return position;
  }

  if (bestInGridPosition) return bestInGridPosition;

  const fallbackAngle = Math.atan2(-state.car.y, -state.car.x);
  const fallbackDistance = Math.min(desiredDistance, Math.max(1.5, getRayDistanceToTrainingBounds(state.car, fallbackAngle) - TRAINING_BALL_EDGE_BUFFER));
  return createBallPositionFromRay(state.car, fallbackAngle, fallbackDistance);
}

function getAvoidBoundaryReturnDrive(state: CarToBallState, maxSpeed: number, turnSpeed: number) {
  const edgeMargin = getGridEdgeMargin(state.car, AVOID_CAR_GRID_LIMIT);
  const distanceToEdgeAhead = getRayDistanceToGridBounds(state.car, state.headingAngle, AVOID_CAR_GRID_LIMIT);
  const shouldReturnToGrid = edgeMargin < AVOID_EDGE_MARGIN || distanceToEdgeAhead < AVOID_EDGE_LOOKAHEAD_DISTANCE;
  if (!shouldReturnToGrid) return null;

  const centerAngle = Math.atan2(-state.car.y, -state.car.x);
  const centerAngleError = getSmallestAngleDelta(state.headingAngle, centerAngle);
  const turnCommand = clamp(
    centerAngleError * turnSpeed * AVOID_EDGE_TURN_GAIN,
    -turnSpeed * AVOID_EDGE_TURN_LIMIT,
    turnSpeed * AVOID_EDGE_TURN_LIMIT,
  );
  const forwardSpeed = Math.abs(centerAngleError) < AVOID_EDGE_CENTER_ANGLE
    ? maxSpeed * AVOID_EDGE_FORWARD_SPEED_SCALE
    : 0;

  return {
    leftVel: clampMotorCommand(forwardSpeed - turnCommand),
    rightVel: clampMotorCommand(forwardSpeed + turnCommand),
  };
}

function clampMotorCommand(value: number) {
  return clamp(value, -MOTOR_CONTROL_LIMIT, MOTOR_CONTROL_LIMIT);
}

const defaultTrainingStatus: TrainingStatus = {
  is_running: false,
  epoch: 0,
  total_epochs: 50,
  loss: 0,
  progress: 0,
  error: null,
  message: "",
};

function timestamp() {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

export default function MujocoPage() {
  const userId = useSimCarStore((state) => state.userId);
  const {
    isLoaded,
    mujoco,
    model,
    data,
    step,
    setControl,
    reset,
    getBodyPosition,
    getCarToBallState,
    getWheelVelocityState,
    setTargetBallPosition,
  } = useMujoco();
  const keysRef = useRef<Set<string>>(new Set());
  const firstPersonCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const firstPersonPreviewRef = useRef<HTMLDivElement | null>(null);
  const lastDriveCommandRef = useRef({ left: Number.NaN, right: Number.NaN });
  const modelDriveRef = useRef({ left: 0, right: 0 });
  const inferenceTimerRef = useRef<number | null>(null);
  const modelTaskRunRef = useRef<ModelTaskRunState | null>(null);
  const modelTaskInferenceInFlightRef = useRef(false);
  const lastAutoBallResetRef = useRef(0);
  const avoidBallWasVisibleRef = useRef(false);
  const avoidBallLostViewAtRef = useRef<number | null>(null);
  const avoidAllowMisplacedRetryRef = useRef(false);
  const avoidHeadingSettleRef = useRef<{
    heading: number;
    sampledAt: number;
    stableSince: number | null;
  } | null>(null);
  const [showJointOverlay, setShowJointOverlay] = useState(false);
  const [driveSpeed, setDriveSpeed] = useState(DEFAULT_DRIVE_SPEED);
  const [turnSpeed, setTurnSpeed] = useState(DEFAULT_TURN_SPEED);
  const [fps, setFps] = useState(0);
  const [collectionFps, setCollectionFps] = useState(30);
  const [datasetName, setDatasetName] = useState(TASK_DATASET_NAMES.follow);
  const [savedDatasetFrameCount, setSavedDatasetFrameCount] = useState(0);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [trainingDatasetName, setTrainingDatasetName] = useState("");
  const [episodeId, setEpisodeId] = useState(1);
  const [taskMode, setTaskMode] = useState<TaskMode>("follow");
  const [controlMode, setControlMode] = useState<ControlMode>("manual");
  const [trainingEpochs, setTrainingEpochs] = useState(50);
  const [trainingStatus, setTrainingStatus] = useState(defaultTrainingStatus);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [autoInference, setAutoInference] = useState(false);
  const [isTaskInferenceRunning, setIsTaskInferenceRunning] = useState(false);
  const [inferenceResult, setInferenceResult] = useState("模型未加载");
  const [wheelState, setWheelState] = useState({ left: 0, right: 0 });
  const [carPosition, setCarPosition] = useState<MjPosition | null>(null);
  const [logs, setLogs] = useState<string[]>([
    `[${timestamp()}] MuJoCo 页面就绪，等待仿真初始化...`,
  ]);
  const fpsFramesRef = useRef(0);
  const fpsLastTimeRef = useRef(performance.now());

  const addLog = useCallback((message: string) => {
    setLogs((items) => [`[${timestamp()}] ${message}`, ...items].slice(0, 120));
  }, []);

  const refreshDatasets = useCallback(async () => {
    try {
      const result = await listDatasetDirs(userId);
      const nextDatasets = result.datasets || [];
      setDatasets(nextDatasets);
      setTrainingDatasetName((current) => {
        if (current && nextDatasets.includes(current)) return current;
        return nextDatasets[0] || "";
      });
    } catch (error) {
      setDatasets([]);
      addLog(`数据集列表刷新失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog, userId]);

  const refreshDatasetFrameCount = useCallback(async (targetDatasetName = datasetName) => {
    const normalizedDatasetName = targetDatasetName.trim();
    if (!normalizedDatasetName) {
      setSavedDatasetFrameCount(0);
      return;
    }

    try {
      const result = await getDatasetInfo(userId, normalizedDatasetName);
      setSavedDatasetFrameCount(result.total_frames || 0);
    } catch (error) {
      setSavedDatasetFrameCount(0);
      addLog(`数据集帧数刷新失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog, datasetName, userId]);

  const collection = useDataCollection({
    userId,
    datasetName,
    episodeId,
    fps: collectionFps,
    taskName: taskMode === "follow" ? "mujoco_follow_ball" : "mujoco_avoid_ball",
    modelRef: model,
    dataRef: data,
    fpCanvasRef: firstPersonCanvasRef,
    onLog: addLog,
    onEpisodeEnded: (payload) => {
      if (!payload.error) {
        setSavedDatasetFrameCount((value) => value + Math.max(0, payload.frame_count ?? 0));
        setEpisodeId((value) => value + 1);
        refreshDatasets();
        refreshDatasetFrameCount(datasetName);
      }
    },
  });
  const {
    isRecording,
    frameCount,
    socketStatus,
    disconnect,
    startRecording,
    stopRecording,
    setAction,
    captureFrame,
  } = collection;
  const totalCollectedFrameCount = savedDatasetFrameCount + (isRecording ? frameCount : 0);

  const computeDrive = useCallback(() => {
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

    return {
      leftVel: clampMotorCommand(leftVel),
      rightVel: clampMotorCommand(rightVel),
    };
  }, [driveSpeed, turnSpeed]);

  const computeAutoDrive = useCallback(() => {
    const state = getCarToBallState();
    if (!state) return { leftVel: 0, rightVel: 0 };

    const maxSpeed = clamp(driveSpeed, DRIVE_SPEED_MIN, DRIVE_SPEED_MAX);
    const turnCommand = clamp(state.angleError * turnSpeed * 1.6, -turnSpeed * 1.4, turnSpeed * 1.4);

    if (taskMode === "follow") {
      const forwardSpeed = state.distance > 1.1 ? maxSpeed * 0.65 : 0;
      return {
        leftVel: clampMotorCommand(forwardSpeed - turnCommand),
        rightVel: clampMotorCommand(forwardSpeed + turnCommand),
      };
    }

    const boundaryReturnDrive = getAvoidBoundaryReturnDrive(state, maxSpeed, turnSpeed);
    if (boundaryReturnDrive) return boundaryReturnDrive;

    const ballIsAhead = isAvoidBallInView(state);
    const ballIsClose = state.distance < AVOID_TRIGGER_DISTANCE;
    if (ballIsAhead && ballIsClose) {
      const awayTurn = (state.angleError >= 0 ? -1 : 1) * turnSpeed * 1.25;
      const forwardSpeed = maxSpeed * 0.35;
      return {
        leftVel: clampMotorCommand(forwardSpeed - awayTurn),
        rightVel: clampMotorCommand(forwardSpeed + awayTurn),
      };
    }

    return {
      leftVel: clampMotorCommand(maxSpeed * 0.45),
      rightVel: clampMotorCommand(maxSpeed * 0.45),
    };
  }, [driveSpeed, getCarToBallState, taskMode, turnSpeed]);

  const computeActiveDrive = useCallback(() => {
    if (controlMode === "model") {
      return {
        leftVel: modelDriveRef.current.left,
        rightVel: modelDriveRef.current.right,
      };
    }
    if (controlMode === "auto") return computeAutoDrive();
    return computeDrive();
  }, [computeAutoDrive, computeDrive, controlMode]);

  const refreshTelemetry = useCallback(() => {
    const { velLeft, velRight } = getWheelVelocityState();
    setWheelState({ left: velLeft, right: velRight });
    setCarPosition(getBodyPosition("car"));
  }, [getBodyPosition, getWheelVelocityState]);

  const applyDrive = useCallback(() => {
    const { leftVel, rightVel } = computeActiveDrive();
    setAction(leftVel, rightVel);

    setControl("motor_wheel_fl", leftVel);
    setControl("motor_wheel_rl", leftVel);
    setControl("motor_wheel_fr", rightVel);
    setControl("motor_wheel_rr", rightVel);

    const last = lastDriveCommandRef.current;
    if (Math.abs(last.left - leftVel) > 0.05 || Math.abs(last.right - rightVel) > 0.05) {
      lastDriveCommandRef.current = { left: leftVel, right: rightVel };
    }
  }, [setAction, setControl, computeActiveDrive]);

  useEffect(() => {
    refreshTelemetry();
  }, [refreshTelemetry]);

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
    window.addEventListener("keydown", handleKeyDown, { capture: true });
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown, { capture: true });
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  const placeVisibleTrainingBall = useCallback((reason: string) => {
    const state = getCarToBallState();
    const position = taskMode === "avoid"
      ? createAvoidBallPositionAhead(state)
      : createVisibleBallPosition(state, taskMode);
    if (setTargetBallPosition(position)) {
      lastAutoBallResetRef.current = performance.now();
      avoidBallWasVisibleRef.current = taskMode !== "avoid";
      avoidBallLostViewAtRef.current = null;
      avoidAllowMisplacedRetryRef.current = taskMode === "avoid";
      avoidHeadingSettleRef.current = null;
      addLog(`${reason}: (${position.x.toFixed(1)}, ${position.y.toFixed(1)})`);
    }
  }, [addLog, getCarToBallState, setTargetBallPosition, taskMode]);

  const avoidHeadingIsSettled = useCallback((state: CarToBallState, now: number) => {
    const sample = avoidHeadingSettleRef.current;
    if (!sample) {
      avoidHeadingSettleRef.current = {
        heading: state.headingAngle,
        sampledAt: now,
        stableSince: null,
      };
      return false;
    }

    const elapsedSeconds = (now - sample.sampledAt) / 1000;
    if (elapsedSeconds * 1000 < AVOID_HEADING_SAMPLE_MS) return false;

    const headingDelta = Math.abs(getSmallestAngleDelta(sample.heading, state.headingAngle));
    const headingRate = headingDelta / Math.max(elapsedSeconds, 0.001);
    const stableSince = headingRate <= AVOID_HEADING_STABLE_RATE
      ? sample.stableSince ?? now
      : null;

    avoidHeadingSettleRef.current = {
      heading: state.headingAngle,
      sampledAt: now,
      stableSince,
    };

    return stableSince !== null && now - stableSince >= AVOID_HEADING_STABLE_MS;
  }, []);

  const maybeRandomizeBallForAutoCollection = useCallback(() => {
    if (!isRecording || controlMode !== "auto") return;

    const state = getCarToBallState();
    if (!state) return;

    const now = performance.now();
    const timeSinceLastReset = now - lastAutoBallResetRef.current;
    if (timeSinceLastReset < AUTO_BALL_RESET_COOLDOWN_MS) return;

    if (taskMode === "follow") {
      if (state.distance > BALL_APPROACH_DISTANCE) return;
      placeVisibleTrainingBall("自动采集换目标");
      return;
    }

    const ballIsInAvoidView = isAvoidBallInView(state);

    if (ballIsInAvoidView) {
      avoidBallWasVisibleRef.current = true;
      avoidBallLostViewAtRef.current = null;
      avoidHeadingSettleRef.current = null;
      return;
    }

    if (!avoidBallWasVisibleRef.current) {
      if (!avoidAllowMisplacedRetryRef.current) return;
      if (timeSinceLastReset < AVOID_MISPLACED_RETRY_MS) return;
      placeVisibleTrainingBall("自动采集重置前方目标");
      return;
    }

    if (avoidBallLostViewAtRef.current === null) {
      avoidBallLostViewAtRef.current = now;
      avoidHeadingSettleRef.current = null;
      return;
    }

    const lostViewDuration = now - avoidBallLostViewAtRef.current;
    const reachedMinimumWait = lostViewDuration >= AVOID_RELOCATE_MIN_SETTLE_MS;
    const reachedMaximumWait = lostViewDuration >= AVOID_RELOCATE_MAX_SETTLE_MS;
    if (!reachedMinimumWait) return;
    if (getGridEdgeMargin(state.car, AVOID_CAR_GRID_LIMIT) < AVOID_EDGE_RELEASE_MARGIN) return;
    if (!reachedMaximumWait && !avoidHeadingIsSettled(state, now)) return;

    placeVisibleTrainingBall("自动采集换目标");
  }, [avoidHeadingIsSettled, controlMode, getCarToBallState, isRecording, placeVisibleTrainingBall, taskMode]);

  const stepWithDrive = useCallback(() => {
    maybeRandomizeBallForAutoCollection();
    applyDrive();

    fpsFramesRef.current++;
    const now = performance.now();
    if (now - fpsLastTimeRef.current >= 1000) {
      setFps(fpsFramesRef.current);
      fpsFramesRef.current = 0;
      fpsLastTimeRef.current = now;
    }

    step();
    refreshTelemetry();
    captureFrame();
  }, [applyDrive, captureFrame, maybeRandomizeBallForAutoCollection, refreshTelemetry, step]);

  const handleReset = useCallback(() => {
    reset();
    keysRef.current.clear();
    lastDriveCommandRef.current = { left: Number.NaN, right: Number.NaN };
    avoidBallWasVisibleRef.current = false;
    avoidBallLostViewAtRef.current = null;
    avoidAllowMisplacedRetryRef.current = false;
    avoidHeadingSettleRef.current = null;
    setWheelState({ left: 0, right: 0 });
    setCarPosition(getBodyPosition("car"));
    addLog("场景已复位");
  }, [addLog, getBodyPosition, reset]);

  const handleSetBallPosition = useCallback((position: { label: string; x: number; y: number; z: number }) => {
    const boundedPosition = clampBallPositionToGrid(position);
    if (!setTargetBallPosition(boundedPosition)) {
      addLog("目标球位置设置失败：MuJoCo 尚未加载完成");
      return;
    }
    addLog(`目标球已移动到${position.label}: (${boundedPosition.x.toFixed(1)}, ${boundedPosition.y.toFixed(1)})`);
  }, [addLog, setTargetBallPosition]);

  const handleRandomizeBall = useCallback(() => {
    handleSetBallPosition(createRandomBallPosition());
  }, [handleSetBallPosition]);

  const handlePlaceBallOnGround = useCallback((position: { x: number; y: number }) => {
    const clampedX = clamp(position.x, -8, 8);
    const clampedY = clamp(position.y, -8, 8);
    handleSetBallPosition({
      label: "鼠标位置",
      x: clampedX,
      y: clampedY,
      z: 0.25,
    });
  }, [handleSetBallPosition]);

  const handleFirstPersonCanvasReady = useCallback((canvas: HTMLCanvasElement | null) => {
    firstPersonCanvasRef.current = canvas;
  }, []);

  const handleToggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      if (!isLoaded) {
        addLog("仿真尚未加载完成，不能开始采集");
        return;
      }
      startRecording();
    }
  }, [addLog, isLoaded, isRecording, startRecording, stopRecording]);

  const handleStartAutoCollection = useCallback(() => {
    if (!isLoaded) {
      addLog("仿真尚未加载完成，不能开始自动采集");
      return;
    }

    setControlMode("auto");
    setAutoInference(false);
    if (taskMode === "follow") {
      placeVisibleTrainingBall("自动采集起始目标");
    } else {
      const state = getCarToBallState();
      avoidBallWasVisibleRef.current = state ? isAvoidBallInView(state) : false;
      avoidBallLostViewAtRef.current = null;
      avoidAllowMisplacedRetryRef.current = false;
      avoidHeadingSettleRef.current = null;
      lastAutoBallResetRef.current = performance.now();
      addLog("避开球自动采集起始：保留当前目标球位置");
    }
    if (!isRecording) {
      startRecording();
    }
    addLog(`自动采集已启用: ${taskMode === "follow" ? "跟踪球" : "避开球"}`);
  }, [addLog, getCarToBallState, isLoaded, isRecording, placeVisibleTrainingBall, startRecording, taskMode]);

  const refreshModels = useCallback(async () => {
    try {
      const modelDatasetName = trainingDatasetName || datasetName;
      const result = await listModels(userId, modelDatasetName);
      const nextModels = result.models || [];
      setModels(nextModels);
      setSelectedModel((current) => {
        if (current && nextModels.includes(current)) return current;
        return nextModels.length === 1 ? nextModels[0] : "";
      });
    } catch (error) {
      setModels([]);
      setSelectedModel("");
      addLog(`模型列表刷新失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog, datasetName, trainingDatasetName, userId]);

  useEffect(() => {
    setIsModelLoaded(false);
    setSelectedModel("");
    setAutoInference(false);
    setIsTaskInferenceRunning(false);
    modelDriveRef.current = { left: 0, right: 0 };
    modelTaskRunRef.current = null;
    modelTaskInferenceInFlightRef.current = false;
    refreshModels();
  }, [refreshModels]);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  useEffect(() => {
    refreshDatasetFrameCount();
  }, [refreshDatasetFrameCount]);

  const stopModelInference = useCallback(() => {
    setAutoInference(false);
    setIsTaskInferenceRunning(false);
    setControlMode("manual");
    if (inferenceTimerRef.current) {
      window.clearInterval(inferenceTimerRef.current);
      inferenceTimerRef.current = null;
    }
    modelTaskRunRef.current = null;
    modelTaskInferenceInFlightRef.current = false;
    modelDriveRef.current = { left: 0, right: 0 };
    setControl("motor_wheel_fl", 0);
    setControl("motor_wheel_rl", 0);
    setControl("motor_wheel_fr", 0);
    setControl("motor_wheel_rr", 0);
  }, [setControl]);

  const handleSetTaskMode = useCallback((nextTaskMode: TaskMode) => {
    setTaskMode(nextTaskMode);
    setIsModelLoaded(false);
    setSelectedModel("");
    stopModelInference();

    const nextDatasetName = TASK_DATASET_NAMES[nextTaskMode];
    if (!isRecording && MANAGED_TASK_DATASET_NAMES.has(datasetName)) {
      setDatasetName(nextDatasetName);
    }
    addLog(`任务切换为: ${nextTaskMode === "follow" ? "跟踪球" : "避开球"}`);
  }, [addLog, datasetName, isRecording, stopModelInference]);

  const handleLoadModel = useCallback(async () => {
    if (!selectedModel) {
      addLog("请先选择一个训练模型");
      return;
    }

    if (!trainingDatasetName) {
      addLog("请先选择训练数据集，再加载模型");
      return;
    }

    const modelDatasetName = trainingDatasetName;
    const dataDir = getDatasetPath(userId, modelDatasetName);
    const modelPath = `output/train/${userId}/${selectedModel}/model.pt`;
    addLog(`加载推理模型: ${modelPath}`);

    try {
      const result = await loadTrainedModel(userId, dataDir, modelPath);
      if (!result.success) {
        addLog(`模型加载失败: ${result.message || result.detail || "unknown error"}`);
        return;
      }
      setIsModelLoaded(true);
      setInferenceResult("模型已加载");
      addLog("模型加载成功，可以单次推理或自动推理控制小车");
    } catch (error) {
      addLog(`模型加载请求失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog, selectedModel, trainingDatasetName, userId]);

  const runModelInference = useCallback(async () => {
    if (!isModelLoaded) {
      addLog("请先加载模型");
      return;
    }

    const canvas = firstPersonCanvasRef.current;
    if (!canvas) {
      addLog("第一视角画面还没准备好，不能推理");
      return;
    }

    const { velLeft, velRight } = getWheelVelocityState();
    const image = canvas.toDataURL("image/jpeg", 0.75);
    const result = await runInferenceWithSocket(simSocket, [velLeft, velRight], image, userId);

    if (!result.success || !Array.isArray(result.action)) {
      throw new Error(result.error || "推理失败");
    }

    const [leftRaw, rightRaw] = result.action;
    if (typeof leftRaw !== "number" || typeof rightRaw !== "number") {
      throw new Error(`模型输出格式不正确: ${JSON.stringify(result.action)}`);
    }

    const left = clampMotorCommand(leftRaw);
    const right = clampMotorCommand(rightRaw);
    modelDriveRef.current = { left, right };
    setControlMode("model");
    setInferenceResult(`L:${left.toFixed(2)}, R:${right.toFixed(2)}`);
    addLog(`推理动作: L=${left.toFixed(2)}, R=${right.toFixed(2)}`);
  }, [addLog, getWheelVelocityState, isModelLoaded, userId]);

  const getModelTaskCompletionMessage = useCallback(() => {
    const taskRun = modelTaskRunRef.current;
    if (!taskRun) return null;

    const state = getCarToBallState();
    if (!state) return null;

    const now = performance.now();
    if (now - taskRun.startedAt >= MODEL_TASK_TIMEOUT_MS) {
      return "单次推理超时，已自动停车";
    }

    if (taskMode === "follow") {
      const reachedTarget = state.distance <= FOLLOW_TASK_COMPLETE_DISTANCE
        && Math.abs(state.angleError) <= FOLLOW_TASK_COMPLETE_ANGLE;
      if (!reachedTarget) {
        taskRun.followCompleteSince = null;
        return null;
      }

      taskRun.followCompleteSince = taskRun.followCompleteSince ?? now;
      if (now - taskRun.followCompleteSince >= FOLLOW_TASK_COMPLETE_STABLE_MS) {
        return "追踪任务完成，已到达目标球附近";
      }
      return null;
    }

    const ballIsInDangerZone = isAvoidBallInView(state) && state.distance < AVOID_TRIGGER_DISTANCE;
    if (ballIsInDangerZone) {
      taskRun.avoidDangerSeen = true;
      taskRun.avoidSafeSince = null;
      return null;
    }

    if (!taskRun.avoidDangerSeen) return null;

    const ballIsClear = !isAvoidBallInView(state) || state.distance >= AVOID_TASK_SAFE_DISTANCE;
    if (!ballIsClear) {
      taskRun.avoidSafeSince = null;
      return null;
    }
    if (now - taskRun.startedAt < AVOID_TASK_MIN_DURATION_MS) return null;

    taskRun.avoidSafeSince = taskRun.avoidSafeSince ?? now;
    if (now - taskRun.avoidSafeSince >= AVOID_TASK_SAFE_STABLE_MS) {
      return "避让任务完成，目标球已离开前方危险区";
    }
    return null;
  }, [getCarToBallState, taskMode]);

  const finishModelTask = useCallback((message: string) => {
    stopModelInference();
    setInferenceResult(message);
    addLog(message);
  }, [addLog, stopModelInference]);

  const runModelTaskTick = useCallback(async () => {
    if (modelTaskInferenceInFlightRef.current) return;

    const beforeInferenceMessage = getModelTaskCompletionMessage();
    if (beforeInferenceMessage) {
      finishModelTask(beforeInferenceMessage);
      return;
    }

    modelTaskInferenceInFlightRef.current = true;
    try {
      await runModelInference();
    } catch (error) {
      addLog(`单次推理失败: ${error instanceof Error ? error.message : String(error)}`);
      stopModelInference();
      return;
    } finally {
      modelTaskInferenceInFlightRef.current = false;
    }

    const afterInferenceMessage = getModelTaskCompletionMessage();
    if (afterInferenceMessage) {
      finishModelTask(afterInferenceMessage);
    }
  }, [addLog, finishModelTask, getModelTaskCompletionMessage, runModelInference, stopModelInference]);

  const handleSingleInference = useCallback(async () => {
    if (isTaskInferenceRunning) {
      stopModelInference();
      setInferenceResult("单次推理已停止");
      addLog("已停止单次推理");
      return;
    }

    if (!isModelLoaded) {
      addLog("请先加载模型");
      return;
    }
    if (autoInference) return;
    if (!firstPersonCanvasRef.current) {
      addLog("第一视角画面还没准备好，不能执行任务");
      return;
    }

    const state = getCarToBallState();
    if (!state) {
      addLog("无法读取车辆与目标球状态");
      return;
    }

    modelTaskRunRef.current = {
      startedAt: performance.now(),
      followCompleteSince: null,
      avoidDangerSeen: taskMode === "avoid" && isAvoidBallInView(state) && state.distance < AVOID_TRIGGER_DISTANCE,
      avoidSafeSince: null,
    };
    modelTaskInferenceInFlightRef.current = false;
    setControlMode("model");
    setAutoInference(false);
    setIsTaskInferenceRunning(true);
    addLog(`单次推理已启动: ${taskMode === "follow" ? "追踪球" : "避开球"}`);

    await runModelTaskTick();
    if (!modelTaskRunRef.current) return;

    if (inferenceTimerRef.current) {
      window.clearInterval(inferenceTimerRef.current);
    }
    inferenceTimerRef.current = window.setInterval(() => {
      runModelTaskTick();
    }, MODEL_INFERENCE_INTERVAL_MS);
  }, [
    addLog,
    autoInference,
    getCarToBallState,
    isModelLoaded,
    isTaskInferenceRunning,
    runModelTaskTick,
    stopModelInference,
    taskMode,
  ]);

  const handleToggleAutoInference = useCallback(async () => {
    if (!isModelLoaded) {
      addLog("请先加载模型");
      return;
    }
    if (isTaskInferenceRunning) {
      addLog("单次推理执行中，请先停止");
      return;
    }

    if (autoInference) {
      stopModelInference();
      setInferenceResult("自动推理已停止");
      addLog("已停止模型自动推理");
      return;
    }

    setControlMode("model");
    setAutoInference(true);
    addLog("模型自动推理已启动");
    try {
      await runModelInference();
    } catch (error) {
      addLog(`推理失败: ${error instanceof Error ? error.message : String(error)}`);
    }
    inferenceTimerRef.current = window.setInterval(() => {
      runModelInference().catch((error) => {
        addLog(`自动推理失败: ${error instanceof Error ? error.message : String(error)}`);
        stopModelInference();
      });
    }, MODEL_INFERENCE_INTERVAL_MS);
  }, [addLog, autoInference, isModelLoaded, isTaskInferenceRunning, runModelInference, stopModelInference]);

  const handleStartTraining = useCallback(async () => {
    if (isRecording) {
      addLog("请先结束采集，再开始训练");
      return;
    }

    if (!trainingDatasetName) {
      addLog("请先选择训练数据集");
      return;
    }

    const dataDir = getDatasetPath(userId, trainingDatasetName);
    const outputDir = getTrainPath(userId, trainingDatasetName);
    addLog(`启动训练: ${dataDir}`);

    try {
      const result = await startTraining(userId, {
        data_dir: dataDir,
        output_dir: outputDir,
        epochs: trainingEpochs,
        batch_size: 8,
        lr: 1e-4,
      });

      if (!result.success) {
        addLog(`训练启动失败: ${result.message || "unknown error"}`);
        return;
      }

      setTrainingStatus({
        is_running: true,
        epoch: 0,
        total_epochs: trainingEpochs,
        loss: 0,
        progress: 0,
        error: null,
        message: "训练中",
      });
      addLog("训练任务已提交");
    } catch (error) {
      addLog(`训练请求失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [addLog, isRecording, trainingDatasetName, trainingEpochs, userId]);

  const handleStopTraining = useCallback(async () => {
    await stopTraining(userId);
    setTrainingStatus((status) => ({ ...status, is_running: false, message: "训练已停止" }));
    addLog("已发送停止训练请求");
  }, [addLog, userId]);

  useEffect(() => {
    if (!trainingStatus.is_running) return;
    const timer = window.setInterval(async () => {
      try {
        const status = await getTrainingStatus(userId);
        setTrainingStatus(status);
        if (!status.is_running && status.error) {
          addLog(`训练失败: ${status.error}`);
          return;
        }
        if (!status.is_running && status.progress >= 1) {
          addLog(`训练完成: output/train/${userId}/${trainingDatasetName}/model.pt`);
          setSelectedModel(trainingDatasetName);
          refreshModels();
        }
      } catch (error) {
        addLog(`训练状态获取失败: ${error instanceof Error ? error.message : String(error)}`);
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [addLog, refreshModels, trainingDatasetName, trainingStatus.is_running, userId]);

  useEffect(() => {
    return () => {
      if (inferenceTimerRef.current) {
        window.clearInterval(inferenceTimerRef.current);
      }
      disconnect();
    };
  }, [disconnect]);

  return (
    <div className="flex flex-col h-screen bg-slate-950 overflow-hidden text-slate-100">
      <header className="flex items-center justify-between px-6 py-2 bg-slate-950 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-600 to-blue-600 flex items-center justify-center shadow-lg shadow-violet-900/20">
            <span className="text-white font-bold text-sm">MJC</span>
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">AKA MuJoCo 自动采集模拟器</h2>
            <p className="text-xs text-slate-500">Browser-native MuJoCo WASM + Three.js</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-700">
            <span className={`w-2 h-2 rounded-full ${isLoaded ? "bg-emerald-500" : "bg-yellow-500 animate-pulse"}`} />
            <span className="text-xs text-slate-300">{isLoaded ? "Simulation Active" : "Loading"}</span>
          </div>
          <span className="text-xs text-slate-500">{fps || "--"} FPS</span>
        </div>
      </header>

      <main className="grid flex-1 min-h-0 grid-cols-[304px_minmax(0,1fr)_384px] gap-4 p-4 overflow-hidden">
        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1">
          <section className="rounded-lg border border-slate-700 bg-slate-900/80 p-3">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                数据采集
              </h3>
              <span className={`rounded px-2 py-0.5 text-xs ${isRecording ? "bg-red-500/20 text-red-300" : "bg-slate-800 text-slate-400"}`}>
                {isRecording ? "采集中" : "未录制"}
              </span>
            </div>

            <label className="mb-1 block text-xs text-slate-400">数据集名称</label>
            <input
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value || "mujoco-default")}
              disabled={isRecording}
              className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-mono text-slate-100 outline-none focus:border-emerald-500"
            />

            <div className="mb-3 flex items-center justify-between text-xs text-slate-400">
              <span>采样频率</span>
              <span className="font-mono text-slate-200">{collectionFps} FPS</span>
            </div>
            <div className="mb-3 grid grid-cols-3 gap-2">
              {[10, 20, 30].map((value) => (
                <button
                  key={value}
                  onClick={() => setCollectionFps(value)}
                  disabled={isRecording}
                  className={`rounded border px-2 py-1 text-xs font-mono transition ${collectionFps === value ? "border-emerald-500/50 bg-emerald-500/20 text-emerald-200" : "border-slate-700 bg-slate-950 text-slate-300 hover:bg-slate-800"}`}
                >
                  {value} FPS
                </button>
              ))}
            </div>

            <div className="mb-3 grid grid-cols-2 gap-2 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-xs">
              <div>
                <div className="text-slate-500">本轮帧数</div>
                <div className="mt-1 font-mono text-slate-100">{frameCount}</div>
              </div>
              <div>
                <div className="text-slate-500">总帧数</div>
                <div className="mt-1 font-mono text-emerald-200">{totalCollectedFrameCount}</div>
              </div>
            </div>

            <button
              onClick={handleToggleRecording}
              className={`mb-2 w-full rounded-lg py-2.5 text-sm font-medium text-white transition ${isRecording ? "bg-red-600 hover:bg-red-500" : "bg-emerald-600 hover:bg-emerald-500"}`}
            >
              {isRecording ? `结束采集 (第${episodeId}轮)` : `开始采集 (第${episodeId}轮)`}
            </button>
            <button
              onClick={handleReset}
              className="w-full rounded-lg bg-emerald-700 py-2.5 text-sm font-medium text-white transition hover:bg-emerald-600"
            >
              复位场景
            </button>

            <div className="mt-3 border-t border-slate-700/70 pt-3">
              <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
                <span>采集轮次</span>
                <span className="font-mono text-slate-200">Ep. {episodeId}</span>
              </div>
              <input
                type="number"
                min={1}
                value={episodeId}
                disabled={isRecording}
                onChange={(e) => setEpisodeId(Number(e.target.value) || 1)}
                className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-mono text-slate-100 outline-none focus:border-emerald-500"
              />
            </div>
          </section>

          <section className="rounded-lg border border-slate-700 bg-slate-900/80 p-3">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
              <span className="h-2 w-2 rounded-full bg-violet-400" />
              自动任务
            </h3>
            <div className="mb-3 grid grid-cols-3 gap-2">
              {[
                ["manual", "手动驾驶"],
                ["auto", "规则自动"],
                ["model", "模型推理"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  onClick={() => {
                    setControlMode(value as ControlMode);
                    if (value !== "model") {
                      stopModelInference();
                    }
                  }}
                  className={`rounded-lg border px-3 py-2 text-sm transition ${controlMode === value ? "border-cyan-500/60 bg-cyan-500/20 text-cyan-100" : "border-slate-700 bg-slate-950 text-slate-300 hover:bg-slate-800"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[
                ["follow", "跟踪球"],
                ["avoid", "避开球"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  onClick={() => handleSetTaskMode(value as TaskMode)}
                  disabled={isRecording}
                  className={`rounded-lg border px-3 py-2 text-sm transition ${taskMode === value ? "border-violet-500/60 bg-violet-500/20 text-violet-100" : "border-slate-700 bg-slate-950 text-slate-300 hover:bg-slate-800"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              onClick={handleRandomizeBall}
              className="mt-3 w-full rounded-lg border border-orange-500/40 bg-orange-500/10 py-2 text-sm font-medium text-orange-100 transition hover:bg-orange-500/20"
            >
              随机目标球位置
            </button>
            <button
              onClick={handleStartAutoCollection}
              className="mt-2 w-full rounded-lg bg-cyan-700 py-2 text-sm font-medium text-white transition hover:bg-cyan-600"
            >
              自动驾驶并采集
            </button>
            <p className="mt-3 text-xs leading-5 text-slate-500">
              当前目标：{taskMode === "follow" ? "追踪红色目标球并保持距离" : "当目标球进入前方区域时自动左右避让"}
            </p>
          </section>

          <section className="rounded-lg border border-slate-700 bg-slate-900/80 p-3">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
              <span className="h-2 w-2 rounded-full bg-orange-400" />
              驱动设置
            </h3>
            <label className="mb-1 block text-xs text-slate-400">速度：{driveSpeed.toFixed(1)}</label>
            <input
              type="range"
              min={DRIVE_SPEED_MIN}
              max={DRIVE_SPEED_MAX}
              step={DRIVE_SPEED_STEP}
              value={driveSpeed}
              onChange={(e) => setDriveSpeed(Number(e.target.value))}
              className="mb-3 w-full accent-emerald-500"
            />
            <label className="mb-1 block text-xs text-slate-400">转向：{turnSpeed}</label>
            <input
              type="range"
              min={TURN_SPEED_MIN}
              max={TURN_SPEED_MAX}
              step={TURN_SPEED_STEP}
              value={turnSpeed}
              onChange={(e) => setTurnSpeed(Number(e.target.value))}
              className="mb-3 w-full accent-orange-400"
            />
            <label className="flex cursor-pointer items-center gap-2 rounded-lg bg-slate-950 px-3 py-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={showJointOverlay}
                onChange={(e) => setShowJointOverlay(e.target.checked)}
                className="h-4 w-4 accent-emerald-500"
              />
              显示关节
            </label>
          </section>

          <section className="rounded-lg border border-slate-700 bg-slate-900/80 p-3">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              模型训练
            </h3>
            <div className="mb-3 flex items-center justify-between">
              <label className="text-xs text-slate-400">训练数据集</label>
              <button
                onClick={refreshDatasets}
                disabled={trainingStatus.is_running}
                className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:text-slate-600"
              >
                刷新
              </button>
            </div>
            <select
              value={trainingDatasetName}
              onChange={(event) => {
                setTrainingDatasetName(event.target.value);
                setIsModelLoaded(false);
                setSelectedModel("");
              }}
              disabled={trainingStatus.is_running}
              className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-mono text-slate-100 outline-none focus:border-violet-500 disabled:cursor-not-allowed disabled:text-slate-500"
            >
              <option value="">{datasets.length ? "请选择数据集" : "暂无已保存数据集"}</option>
              {datasets.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <label className="mb-1 block text-xs text-slate-400">训练轮次</label>
            <input
              type="number"
              min={1}
              max={1000}
              value={trainingEpochs}
              disabled={trainingStatus.is_running}
              onChange={(e) => setTrainingEpochs(Number(e.target.value) || 1)}
              className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-mono text-slate-100 outline-none focus:border-violet-500"
            />
            {trainingStatus.is_running ? (
              <button onClick={handleStopTraining} className="w-full rounded-lg bg-red-700 py-2.5 text-sm font-medium text-white transition hover:bg-red-600">
                停止训练
              </button>
            ) : (
              <button onClick={handleStartTraining} className="w-full rounded-lg bg-violet-700 py-2.5 text-sm font-medium text-white transition hover:bg-violet-600">
                开始训练
              </button>
            )}
            {(trainingStatus.is_running || trainingStatus.epoch > 0 || trainingStatus.error) && (
              <div className="mt-3 border-t border-slate-700/70 pt-3">
                <div className="mb-1 flex justify-between text-xs">
                  <span className="text-slate-400">Epoch {trainingStatus.epoch}/{trainingStatus.total_epochs}</span>
                  <span className={`font-mono ${trainingStatus.error ? "text-red-300" : "text-violet-300"}`}>{Math.round(trainingStatus.progress * 100)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-950">
                  <div className={`h-full rounded-full ${trainingStatus.error ? "bg-red-500" : "bg-violet-500"}`} style={{ width: `${Math.min(100, trainingStatus.progress * 100)}%` }} />
                </div>
                <div className="mt-1 text-xs font-mono text-slate-400">Loss: {trainingStatus.loss.toFixed(6)}</div>
                {trainingStatus.error && (
                  <div className="mt-2 rounded border border-red-500/30 bg-red-500/10 px-2 py-1.5 text-xs leading-5 text-red-200">
                    {trainingStatus.error}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="rounded-lg border border-slate-700 bg-slate-900/80 p-3">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                推理控制
              </h3>
              <button
                onClick={refreshModels}
                className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300 transition hover:bg-slate-800"
              >
                刷新
              </button>
            </div>

            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="text-slate-500">模型状态</span>
              <span className={`rounded px-2 py-0.5 ${isModelLoaded ? "bg-emerald-500/20 text-emerald-300" : "bg-slate-800 text-slate-400"}`}>
                {isModelLoaded ? "已加载" : "未加载"}
              </span>
            </div>
            <div className="mb-3 flex items-center justify-between text-xs">
              <span className="text-slate-500">推理动作</span>
              <span className="max-w-40 truncate rounded bg-slate-800 px-2 py-0.5 font-mono text-slate-300" title={inferenceResult}>
                {inferenceResult}
              </span>
            </div>

            <label className="mb-1 block text-xs text-slate-400">选择训练模型</label>
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setIsModelLoaded(false);
                stopModelInference();
              }}
              className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-mono text-slate-100 outline-none focus:border-cyan-500"
            >
              <option value="">{models.length ? "请选择模型" : "暂无可用模型"}</option>
              {models.map((modelName) => (
                <option key={modelName} value={modelName}>{modelName}</option>
              ))}
            </select>

            <button
              onClick={handleLoadModel}
              disabled={!selectedModel}
              className={`mb-2 w-full rounded-lg py-2.5 text-sm font-medium text-white transition ${selectedModel ? "bg-cyan-700 hover:bg-cyan-600" : "bg-slate-700 text-slate-500 cursor-not-allowed"}`}
            >
              加载模型
            </button>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={handleSingleInference}
                disabled={!isModelLoaded || autoInference}
                className={`rounded-lg py-2 text-sm font-medium transition ${isTaskInferenceRunning ? "bg-red-700 text-white hover:bg-red-600" : isModelLoaded && !autoInference ? "bg-emerald-700 text-white hover:bg-emerald-600" : "bg-slate-700 text-slate-500 cursor-not-allowed"}`}
              >
                {isTaskInferenceRunning ? "停止推理" : "单次推理"}
              </button>
              <button
                onClick={handleToggleAutoInference}
                disabled={!isModelLoaded || isTaskInferenceRunning}
                className={`rounded-lg py-2 text-sm font-medium transition ${autoInference ? "bg-red-700 text-white hover:bg-red-600" : isModelLoaded && !isTaskInferenceRunning ? "bg-orange-700 text-white hover:bg-orange-600" : "bg-slate-700 text-slate-500 cursor-not-allowed"}`}
              >
                {autoInference ? "停止自动" : "自动推理"}
              </button>
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              单次推理会执行一次完整任务，完成追踪或避让后自动停车。
            </p>
          </section>
        </aside>

        <section className="relative min-h-0 overflow-hidden rounded-lg border border-slate-700 bg-slate-100">
          <div className="absolute left-4 top-4 z-20 rounded bg-white/90 px-3 py-2 text-xs leading-5 text-slate-800 shadow">
            <div>WASD / 方向键移动</div>
            <div>拖动红球移动目标；拖动空地旋转视角</div>
          </div>
          <MujocoRenderer
            mujoco={mujoco}
            model={model}
            data={data}
            isLoaded={isLoaded}
            onStep={stepWithDrive}
            showJointOverlay={showJointOverlay}
            onPlaceBall={handlePlaceBallOnGround}
            onFirstPersonCanvasReady={handleFirstPersonCanvasReady}
            firstPersonContainerRef={firstPersonPreviewRef}
          />
          <div className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2">
            {[
              ["指令: 前进", "KeyW"],
              ["指令: 左转", "KeyA"],
              ["指令: 右转", "KeyD"],
              ["指令: 后退", "KeyS"],
            ].map(([label, key]) => (
              <button
                key={key}
                onMouseDown={() => {
                  keysRef.current.add(key);
                }}
                onMouseUp={() => {
                  keysRef.current.delete(key);
                }}
                onMouseLeave={() => {
                  keysRef.current.delete(key);
                }}
                className="rounded bg-blue-600 px-4 py-2 text-xs font-medium text-white shadow hover:bg-blue-500"
              >
                {label}
              </button>
            ))}
            <button onClick={handleReset} className="rounded bg-emerald-600 px-4 py-2 text-xs font-medium text-white shadow hover:bg-emerald-500">
              复位
            </button>
            <span className="ml-2 text-xs text-slate-700">本轮帧数: {frameCount}</span>
          </div>
        </section>

        <aside className="flex min-h-0 flex-col gap-3 overflow-hidden">
          <section className="rounded-lg border border-slate-700 bg-slate-900/80 p-4">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
              <span className="h-2 w-2 rounded-full bg-blue-400" />
              MuJoCo 视角
            </h3>
            <div
              ref={firstPersonPreviewRef}
              className="mb-3 h-48 overflow-hidden rounded-lg border border-violet-500/50 bg-slate-950 shadow-lg shadow-violet-950/30"
            />
            <div className="space-y-2 border-t border-slate-700/70 pt-3 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">Socket</span>
                <span className={socketStatus === "connected" ? "text-emerald-300" : "text-slate-300"}>{socketStatus}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">小车位置</span>
                <span className="font-mono text-slate-200">
                  {carPosition ? `X: ${carPosition.x.toFixed(2)}, Y: ${carPosition.y.toFixed(2)}` : "X: --, Y: --"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">轮速</span>
                <span className="font-mono text-slate-200">L: {wheelState.left.toFixed(1)}, R: {wheelState.right.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">数据目录</span>
                <span className="max-w-52 truncate font-mono text-slate-200" title={getDatasetPath(userId, datasetName)}>{datasetName}</span>
              </div>
            </div>
          </section>

          <section className="min-h-0 flex-1 rounded-lg border border-slate-700 bg-slate-900/80">
            <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
              <h3 className="text-sm font-semibold text-slate-100">后台日志</h3>
              <button onClick={() => setLogs([])} className="text-xs text-slate-300 hover:text-white">清空</button>
            </div>
            <div className="h-full overflow-y-auto p-4 font-mono text-xs leading-6 text-slate-400">
              {logs.map((item, index) => (
                <div key={`${item}-${index}`}>{item}</div>
              ))}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
