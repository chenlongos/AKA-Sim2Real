import { useEffect, useRef, useState, useCallback } from "react";
import type { MainModule, MjModel, MjData } from "@mujoco/mujoco";
import wasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import { CAR_ARM_XML } from "./carArmXml";

export interface MjPosition {
  x: number;
  y: number;
  z: number;
}

export interface CarToBallState {
  car: MjPosition;
  ball: MjPosition;
  distance: number;
  angleError: number;
  frontness: number;
  headingAngle: number;
  forward: {
    x: number;
    y: number;
  };
}

const TARGET_BALL_JOINT = "target_ball_free";
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

function findNamedIndex(
  names: unknown,
  nameAddresses: ArrayLike<number> | undefined,
  count: number,
  targetName: string,
) {
  if (!nameAddresses) return -1;

  for (let index = 0; index < count; index++) {
    if (resolveName(names, nameAddresses[index]) === targetName) return index;
  }
  return -1;
}

function readBodyPosition(model: MjModel, data: MjData, bodyName: string): MjPosition | null {
  const bodyId = findNamedIndex(model.names, model.name_bodyadr, model.nbody, bodyName);
  if (bodyId < 0) return null;

  return {
    x: Number(data.xpos[bodyId * 3]),
    y: Number(data.xpos[bodyId * 3 + 1]),
    z: Number(data.xpos[bodyId * 3 + 2]),
  };
}

function readBodyForward2D(model: MjModel, data: MjData, bodyName: string) {
  const bodyId = findNamedIndex(model.names, model.name_bodyadr, model.nbody, bodyName);
  if (bodyId < 0) return null;

  const quatIndex = bodyId * 4;
  const w = Number(data.xquat[quatIndex]);
  const x = Number(data.xquat[quatIndex + 1]);
  const y = Number(data.xquat[quatIndex + 2]);
  const z = Number(data.xquat[quatIndex + 3]);

  return {
    x: 1 - 2 * (y * y + z * z),
    y: 2 * (x * y + w * z),
  };
}

function readJointVelocity(model: MjModel, data: MjData, jointName: string) {
  const jointId = findNamedIndex(model.names, model.name_jntadr, model.njnt, jointName);
  if (jointId < 0) return 0;
  return Number(data.qvel[model.jnt_dofadr[jointId]] ?? 0);
}

function readAverageJointVelocity(model: MjModel, data: MjData, jointNames: string[]) {
  const total = jointNames.reduce((sum, jointName) => sum + readJointVelocity(model, data, jointName), 0);
  return total / jointNames.length;
}

export function useMujoco() {
  const [isLoaded, setIsLoaded] = useState(false);
  const mujocoRef = useRef<MainModule | null>(null);
  const modelRef = useRef<MjModel | null>(null);
  const dataRef = useRef<MjData | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      const { default: loadMujoco } = await import("@mujoco/mujoco");
      const mujoco = await loadMujoco({
        locateFile: (path: string) => {
          if (path.endsWith(".wasm")) return wasmUrl;
          return path;
        },
      });
      if (cancelled) return;

      mujocoRef.current = mujoco;

      const model = (mujoco.MjModel as unknown as { from_xml_string: (xml: string) => MjModel }).from_xml_string(CAR_ARM_XML);
      const data = new mujoco.MjData(model);

      modelRef.current = model;
      dataRef.current = data;
      setIsLoaded(true);
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  const step = useCallback(() => {
    const m = mujocoRef.current;
    const model = modelRef.current;
    const data = dataRef.current;
    if (!m || !model || !data) return;
    m.mj_step(model, data);
  }, []);

  const setControl = useCallback((name: string, value: number) => {
    const model = modelRef.current;
    const data = dataRef.current;
    if (!model || !data) return;
    const names = model.names;
    let buf: Uint8Array;
    if (typeof names === "string") {
      buf = new TextEncoder().encode(names);
    } else if (names instanceof ArrayBuffer) {
      buf = new Uint8Array(names);
    } else if (ArrayBuffer.isView(names)) {
      buf = new Uint8Array(names.buffer, names.byteOffset, names.byteLength);
    } else {
      return;
    }
    const decoder = new TextDecoder();
    for (let i = 0; i < model.nu; i++) {
      const addr = model.name_actuatoradr[i];
      let end = addr;
      while (end < buf.length && buf[end] !== 0) end++;
      if (decoder.decode(buf.slice(addr, end)) === name) {
        data.ctrl[i] = value;
        return;
      }
    }
  }, []);

  const reset = useCallback(() => {
    const m = mujocoRef.current;
    const model = modelRef.current;
    const data = dataRef.current;
    if (!m || !model || !data) return;
    m.mj_resetData(model, data);
  }, []);

  const getBodyPosition = useCallback((bodyName: string) => {
    const model = modelRef.current;
    const data = dataRef.current;
    if (!model || !data) return null;
    return readBodyPosition(model, data, bodyName);
  }, []);

  const getCarToBallState = useCallback((): CarToBallState | null => {
    const model = modelRef.current;
    const data = dataRef.current;
    if (!model || !data) return null;

    const car = readBodyPosition(model, data, "car");
    const ball = readBodyPosition(model, data, "target_ball");
    const forward = readBodyForward2D(model, data, "car");
    if (!car || !ball || !forward) return null;

    const dx = ball.x - car.x;
    const dy = ball.y - car.y;
    const distance = Math.hypot(dx, dy);
    const targetAngle = Math.atan2(dy, dx);
    const headingAngle = Math.atan2(forward.y, forward.x);
    const angleError = Math.atan2(Math.sin(targetAngle - headingAngle), Math.cos(targetAngle - headingAngle));
    const frontness = distance > 0 ? (forward.x * dx + forward.y * dy) / distance : 0;

    return { car, ball, distance, angleError, frontness, headingAngle, forward };
  }, []);

  const getWheelVelocityState = useCallback(() => {
    const model = modelRef.current;
    const data = dataRef.current;
    if (!model || !data) return { velLeft: 0, velRight: 0 };

    return {
      velLeft: readAverageJointVelocity(model, data, LEFT_WHEEL_JOINTS),
      velRight: readAverageJointVelocity(model, data, RIGHT_WHEEL_JOINTS),
    };
  }, []);

  const setTargetBallPosition = useCallback((position: MjPosition) => {
    const m = mujocoRef.current;
    const model = modelRef.current;
    const data = dataRef.current;
    if (!m || !model || !data) return false;

    const jointId = findNamedIndex(model.names, model.name_jntadr, model.njnt, TARGET_BALL_JOINT);
    if (jointId < 0) return false;

    const qposAddress = model.jnt_qposadr[jointId];
    const qvelAddress = model.jnt_dofadr[jointId];
    data.qpos[qposAddress] = position.x;
    data.qpos[qposAddress + 1] = position.y;
    data.qpos[qposAddress + 2] = position.z;
    data.qpos[qposAddress + 3] = 1;
    data.qpos[qposAddress + 4] = 0;
    data.qpos[qposAddress + 5] = 0;
    data.qpos[qposAddress + 6] = 0;

    for (let offset = 0; offset < 6; offset++) {
      data.qvel[qvelAddress + offset] = 0;
    }

    m.mj_forward(model, data);
    return true;
  }, []);

  return {
    isLoaded,
    mujoco: mujocoRef,
    model: modelRef,
    data: dataRef,
    step,
    setControl,
    reset,
    getBodyPosition,
    getCarToBallState,
    getWheelVelocityState,
    setTargetBallPosition,
  };
}
