import { useEffect, useRef, useState, useCallback } from "react";
import type { MainModule, MjModel, MjData } from "@mujoco/mujoco";
import wasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import { CAR_ARM_XML } from "./carArmXml";

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
    const m = mujocoRef.current;
    const model = modelRef.current;
    const data = dataRef.current;
    if (!m || !model || !data) return;
    const id = m.mj_name2id(model, m.mjtObj.mjOBJ_ACTUATOR, name);
    if (id >= 0) {
      data.ctrl[id] = value;
    }
  }, []);

  const reset = useCallback(() => {
    const m = mujocoRef.current;
    const model = modelRef.current;
    const data = dataRef.current;
    if (!m || !model || !data) return;
    m.mj_resetData(model, data);
  }, []);

  return {
    isLoaded,
    mujoco: mujocoRef,
    model: modelRef,
    data: dataRef,
    step,
    setControl,
    reset,
  };
}
