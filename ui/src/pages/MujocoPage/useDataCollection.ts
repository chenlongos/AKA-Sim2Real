import { useRef, useCallback, useState } from "react";
import { io, Socket } from "socket.io-client";
import type { MjModel, MjData } from "@mujoco/mujoco";

interface UseDataCollectionOptions {
  userId: string;
  datasetName: string;
  episodeId: number;
  fps: number;
  taskName: string;
  modelRef: React.RefObject<MjModel | null>;
  dataRef: React.RefObject<MjData | null>;
  fpCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  onLog?: (message: string) => void;
  onEpisodeEnded?: (payload: { output_path?: string; frame_count?: number; error?: string }) => void;
}

const LEFT_WHEEL_JOINTS = ["wheel_fl_joint", "wheel_rl_joint"];
const RIGHT_WHEEL_JOINTS = ["wheel_fr_joint", "wheel_rr_joint"];

function toUint8Array(buf: unknown): Uint8Array | null {
  if (typeof buf === "string") {
    return new TextEncoder().encode(buf);
  }
  if (buf instanceof ArrayBuffer) {
    return new Uint8Array(buf);
  }
  if (ArrayBuffer.isView(buf)) {
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  return null;
}

function resolveName(names: unknown, addr: number): string {
  const buf = toUint8Array(names);
  if (!buf || addr < 0) return "";
  let end = addr;
  while (end < buf.length && buf[end] !== 0) end++;
  return new TextDecoder().decode(buf.slice(addr, end));
}

function getJointQvel(model: MjModel, data: MjData, jointName: string) {
  for (let jointId = 0; jointId < model.njnt; jointId++) {
    if (resolveName(model.names, model.name_jntadr[jointId]) === jointName) {
      return Number(data.qvel[model.jnt_dofadr[jointId]] ?? 0);
    }
  }
  return 0;
}

function averageJointVelocity(model: MjModel, data: MjData, jointNames: string[]) {
  const values = jointNames.map((jointName) => getJointQvel(model, data, jointName));
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function readWheelState(model: MjModel | null, data: MjData | null) {
  if (!data) return { velLeft: 0, velRight: 0 };
  if (model) {
    return {
      velLeft: averageJointVelocity(model, data, LEFT_WHEEL_JOINTS),
      velRight: averageJointVelocity(model, data, RIGHT_WHEEL_JOINTS),
    };
  }

  const qvel = data.qvel;
  const get = (index: number) => (qvel.length > index ? Number(qvel[index]) : 0);
  const velLeft = (get(0) + get(2)) / 2;
  const velRight = (get(1) + get(3)) / 2;
  return { velLeft, velRight };
}

export function useDataCollection({
  userId,
  datasetName,
  episodeId,
  fps,
  taskName,
  modelRef,
  dataRef,
  fpCanvasRef,
  onLog,
  onEpisodeEnded,
}: UseDataCollectionOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [socketStatus, setSocketStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");

  const socketRef = useRef<Socket | null>(null);
  const recordingRef = useRef(false);
  const lastCaptureRef = useRef(0);
  const leftTargetRef = useRef(0);
  const rightTargetRef = useRef(0);
  const localFrameCountRef = useRef(0);

  const connect = useCallback(() => {
    if (socketRef.current?.connected) return socketRef.current;

    setSocketStatus("connecting");
    const socket = io("/sim", {
      path: "/socket.io",
      transports: ["websocket", "polling"],
      auth: { clientId: `mujoco_${userId}` },
    });

    socket.on("connect", () => {
      setSocketStatus("connected");
      onLog?.("Socket connected to /sim");
    });
    socket.on("disconnect", () => {
      setSocketStatus("disconnected");
      onLog?.("Socket disconnected");
    });
    socket.on("connect_error", () => {
      setSocketStatus("error");
      onLog?.("Socket connection failed");
    });
    socket.on("collection_count", (payload: { count: number }) => {
      setFrameCount(payload.count);
      localFrameCountRef.current = payload.count;
    });
    socket.on("episode_started", (payload: { episode_id: number }) => {
      onLog?.(`Episode ${payload.episode_id} started`);
    });
    socket.on("episode_ended", (payload: { output_path?: string; frame_count?: number; error?: string }) => {
      if (payload.error) {
        onLog?.(`Export failed: ${payload.error}`);
      } else {
        onLog?.(`Dataset exported: ${payload.output_path || "unknown path"}`);
      }
      onEpisodeEnded?.(payload);
    });

    socketRef.current = socket;
    return socket;
  }, [onEpisodeEnded, onLog, userId]);

  const disconnect = useCallback(() => {
    socketRef.current?.disconnect();
    socketRef.current = null;
    setSocketStatus("disconnected");
  }, []);

  const startRecording = useCallback(() => {
    const socket = connect();
    recordingRef.current = true;
    setIsRecording(true);
    setFrameCount(0);
    localFrameCountRef.current = 0;
    lastCaptureRef.current = 0;

    socket.emit("start_episode", {
      user_id: userId,
      episode_id: episodeId,
      task_name: taskName,
    });
    onLog?.(`Start collection: ${datasetName}, episode ${episodeId}`);
  }, [connect, datasetName, episodeId, onLog, taskName, userId]);

  const stopRecording = useCallback(() => {
    recordingRef.current = false;
    setIsRecording(false);

    socketRef.current?.emit("end_episode", {
      user_id: userId,
      episode_id: episodeId,
    });
    onLog?.(`Stop collection: ${localFrameCountRef.current} frames`);
  }, [episodeId, onLog, userId]);

  const setAction = useCallback((left: number, right: number) => {
    leftTargetRef.current = left;
    rightTargetRef.current = right;
  }, []);

  const captureFrame = useCallback(() => {
    if (!recordingRef.current) return;

    const now = performance.now();
    const intervalMs = 1000 / Math.max(1, fps);
    if (now - lastCaptureRef.current < intervalMs) return;
    lastCaptureRef.current = now;

    const canvas = fpCanvasRef.current;
    if (!canvas) {
      onLog?.("Skip frame: first-person canvas is not ready");
      return;
    }

    const image = canvas.toDataURL("image/jpeg", 0.75);
    const { velLeft, velRight } = readWheelState(modelRef.current, dataRef.current);

    socketRef.current?.emit("collect_data", {
      user_id: userId,
      dataset_name: datasetName,
      image,
      timestamp: Date.now(),
      state: {
        vel_left: velLeft,
        vel_right: velRight,
      },
      action: [
        leftTargetRef.current,
        rightTargetRef.current,
        0,
      ],
    });

    localFrameCountRef.current += 1;
    setFrameCount(localFrameCountRef.current);
  }, [dataRef, datasetName, fpCanvasRef, fps, modelRef, onLog, userId]);

  return {
    isRecording,
    frameCount,
    socketStatus,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    setAction,
    captureFrame,
  };
}
