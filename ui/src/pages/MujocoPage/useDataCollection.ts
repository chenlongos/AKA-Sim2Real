import { useRef, useCallback, useState } from "react";
import { io, Socket } from "socket.io-client";
import type { MjData } from "@mujoco/mujoco";

const CAPTURE_INTERVAL_MS = 100; // 10 FPS
const USER_ID = "sim_user";

interface Frame {
  timestamp: number;
  image: string; // base64 JPEG
  action: [number, number, number]; // [left, right, gripper]
  state: { vel_left: number; vel_right: number };
}

export function useDataCollection(
  dataRef: React.RefObject<MjData | null>,
  fpCanvasRef: React.RefObject<HTMLCanvasElement | null>,
) {
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [episodeId, setEpisodeId] = useState(1);
  const [socketStatus, setSocketStatus] = useState<string>("disconnected");

  const socketRef = useRef<Socket | null>(null);
  const recordingRef = useRef(false);
  const lastCaptureRef = useRef(0);
  const episodeIdRef = useRef(1);
  const leftVelRef = useRef(0);
  const rightVelRef = useRef(0);

  // Connect to backend
  const connect = useCallback(() => {
    if (socketRef.current?.connected) return;
    const s = io("http://localhost:8000", {
      transports: ["websocket"],
      namespace: "/sim",
    });
    s.on("connect", () => setSocketStatus("connected"));
    s.on("disconnect", () => setSocketStatus("disconnected"));
    s.on("connect_error", () => setSocketStatus("error"));
    s.on("episode_started", (data: { episode_id: number }) => {
      setEpisodeId(data.episode_id);
      episodeIdRef.current = data.episode_id;
    });
    s.on("collection_count", (data: { count: number }) => {
      setFrameCount(data.count);
    });
    socketRef.current = s;
    return s;
  }, []);

  const disconnect = useCallback(() => {
    socketRef.current?.disconnect();
    socketRef.current = null;
    setSocketStatus("disconnected");
  }, []);

  // Start recording
  const startRecording = useCallback(() => {
    const s = connect();
    recordingRef.current = true;
    setIsRecording(true);
    setFrameCount(0);
    lastCaptureRef.current = 0;

    s.emit("start_episode", {
      user_id: USER_ID,
      episode_id: episodeIdRef.current,
      task_name: "maze_driving",
    });
  }, [connect]);

  // Stop recording
  const stopRecording = useCallback(() => {
    recordingRef.current = false;
    setIsRecording(false);
    const s = socketRef.current;
    if (s) {
      s.emit("end_episode", {
        user_id: USER_ID,
        episode_id: episodeIdRef.current,
      });
      episodeIdRef.current += 1;
    }
  }, []);

  // Set current action values (called from drive loop)
  const setAction = useCallback((left: number, right: number) => {
    leftVelRef.current = left;
    rightVelRef.current = right;
  }, []);

  // Capture frame + send data (called each physics step)
  const captureFrame = useCallback(() => {
    if (!recordingRef.current) return;

    const now = performance.now();
    if (now - lastCaptureRef.current < CAPTURE_INTERVAL_MS) return;
    lastCaptureRef.current = now;

    // Capture FPV canvas
    const fpCanvas = fpCanvasRef.current;
    if (!fpCanvas) return;
    const image = fpCanvas.toDataURL("image/jpeg", 0.7);

    // Read wheel velocities from MuJoCo data
    const d = dataRef.current;
    let velLeft = 0;
    let velRight = 0;
    if (d) {
      // Wheel joints are at indices 0-3 (fl, fr, rl, rr)
      // Joint velocity = d.qvel[joint_dof_adr]
      const getJointVel = (name: string) => {
        // Simplified: use the first 4 hinge joints
        // wheel_fl=0, wheel_fr=1, wheel_rl=2, wheel_rr=3
        const idx = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].indexOf(name);
        if (idx >= 0 && d.qvel.length > idx) return d.qvel[idx];
        return 0;
      };
      velLeft = (getJointVel("wheel_fl") + getJointVel("wheel_rl")) / 2;
      velRight = (getJointVel("wheel_fr") + getJointVel("wheel_rr")) / 2;
    }

    const action: [number, number, number] = [
      leftVelRef.current,
      rightVelRef.current,
      0, // gripper (not used for car)
    ];

    socketRef.current?.emit("collect_data", {
      user_id: USER_ID,
      image: image.replace(/^data:image\/jpeg;base64,/, ""),
      dataset_name: "maze",
      timestamp: now,
      state: { vel_left: velLeft, vel_right: velRight },
      action,
    });
  }, [dataRef, fpCanvasRef]);

  return {
    isRecording,
    frameCount,
    episodeId,
    socketStatus,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    setAction,
    captureFrame,
  };
}
