import { useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import type { MainModule, MjModel, MjData } from "@mujoco/mujoco";

const MJ_GEOM_PLANE = 0;
const MJ_GEOM_SPHERE = 2;
const MJ_GEOM_CYLINDER = 5;
const MJ_GEOM_BOX = 6;

const SUBSTEPS = 16;

const T = new THREE.Matrix4().set(
  1, 0,  0, 0,
  0, 0,  1, 0,
  0, -1, 0, 0,
  0, 0,  0, 1,
);
const Tinv = new THREE.Matrix4().set(
  1, 0,  0, 0,
  0, 0, -1, 0,
  0, 1,  0, 0,
  0, 0,  0, 1,
);

function mjToThree(pos: Float64Array, mat: Float64Array) {
  const position = new THREE.Vector3(pos[0], pos[2], -pos[1]);

  const Rmj = new THREE.Matrix4().set(
    mat[0], mat[3], mat[6], 0,
    mat[1], mat[4], mat[7], 0,
    mat[2], mat[5], mat[8], 0,
    0, 0, 0, 1,
  );

  const Rthree = T.clone().multiply(Rmj).multiply(Tinv);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(Rthree);

  return { position, quaternion };
}

function createGeomMesh(
  type: number,
  size: Float64Array,
  rgba: Float64Array,
): THREE.Object3D {
  let mesh: THREE.Object3D;

  const alpha = rgba.length >= 4 ? rgba[3] : 1;
  const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  const transparent = alpha < 0.99;

  switch (type) {
    case MJ_GEOM_PLANE: {
      const geometry = new THREE.PlaneGeometry(20, 20);
      geometry.rotateX(-Math.PI / 2);
      const material = new THREE.MeshStandardMaterial({
        color,
        side: THREE.DoubleSide,
        roughness: 0.9,
        transparent,
        opacity: alpha,
      });
      mesh = new THREE.Mesh(geometry, material);
      break;
    }
    case MJ_GEOM_SPHERE: {
      const geometry = new THREE.SphereGeometry(size[0], 32, 32);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.4,
        transparent,
        opacity: alpha,
      });
      mesh = new THREE.Mesh(geometry, material);
      break;
    }
    case MJ_GEOM_CYLINDER: {
      const radius = size[0];
      const halfHeight = size[1];
      const wheelGroup = new THREE.Group();

      const cylGeom = new THREE.CylinderGeometry(radius, radius, halfHeight * 2, 32);
      const cylMat = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.5,
        metalness: 0.3,
        transparent,
        opacity: alpha,
      });
      const cylMesh = new THREE.Mesh(cylGeom, cylMat);
      cylMesh.castShadow = true;
      cylMesh.receiveShadow = true;
      wheelGroup.add(cylMesh);

      // Cross-spokes to make rotation visible (cylinder rotating around own axis is invisible)
      const spokeHalfLen = radius * 0.9;
      const spokeThick = halfHeight * 0.3;
      const spokeGeomX = new THREE.BoxGeometry(spokeHalfLen * 2, spokeThick, spokeThick);
      const spokeGeomZ = new THREE.BoxGeometry(spokeThick, spokeThick, spokeHalfLen * 2);
      const spokeMat = new THREE.MeshStandardMaterial({
        color: 0x333333,
        roughness: 0.5,
        metalness: 0.4,
      });
      const spokeX = new THREE.Mesh(spokeGeomX, spokeMat);
      const spokeZ = new THREE.Mesh(spokeGeomZ, spokeMat);
      spokeX.castShadow = true;
      spokeZ.castShadow = true;
      wheelGroup.add(spokeX);
      wheelGroup.add(spokeZ);

      mesh = wheelGroup;
      break;
    }
    case MJ_GEOM_BOX: {
      const geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.5,
        metalness: 0.2,
        transparent,
        opacity: alpha,
      });
      mesh = new THREE.Mesh(geometry, material);
      break;
    }
    default: {
      const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
      const material = new THREE.MeshStandardMaterial({ color: 0xff00ff });
      mesh = new THREE.Mesh(geometry, material);
    }
  }

  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

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
  if (!buf) return "";
  let end = addr;
  while (end < buf.length && buf[end] !== 0) end++;
  return new TextDecoder().decode(buf.slice(addr, end));
}

function makeLabelSprite(text: string, color: string, scale: number = 0.4) {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = color;
  ctx.font = "bold 48px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 32, 32);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(scale, scale, 1);
  return sprite;
}

function resolveCameraIndices(model: MjModel) {
  let fpIdx = -1;
  let tdIdx = -1;
  const names = model.names;
  for (let i = 0; i < model.ncam; i++) {
    const name = resolveName(names, model.name_camadr[i]);
    if (name === "firstperson") fpIdx = i;
    else if (name === "topdown") tdIdx = i;
  }
  return { fpIdx, tdIdx };
}

interface Props {
  mujoco: React.RefObject<MainModule | null>;
  model: React.RefObject<MjModel | null>;
  data: React.RefObject<MjData | null>;
  isLoaded: boolean;
  onStep: () => void;
}

export default function MujocoRenderer({
  mujoco,
  model,
  data,
  isLoaded,
  onStep,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fpContainerRef = useRef<HTMLDivElement>(null);
  const axesContainerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const orbitCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const fpCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const axesCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const axesSceneRef = useRef<THREE.Scene | null>(null);
  const axesGroupRef = useRef<THREE.Group | null>(null);
  const mainRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const fpRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const axesRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const meshesRef = useRef<THREE.Object3D[]>([]);
  const animRef = useRef(0);
  const draggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });
  const orbitStateRef = useRef({
    azimuth: 0.5,
    elevation: 0.4,
    distance: 6,
    target: new THREE.Vector3(0, 0.3, 0),
  });
  const fpCamIdxRef = useRef(-1);

  const setupScene = useCallback(() => {
    const container = containerRef.current;
    const fpContainer = fpContainerRef.current;
    if (!container || !fpContainer) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x2a2a4e);

    const w = container.clientWidth;
    const h = container.clientHeight;
    const orbitCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);

    const fpW = fpContainer.clientWidth || 320;
    const fpH = fpContainer.clientHeight || 210;
    const fpCamera = new THREE.PerspectiveCamera(110, fpW / fpH, 0.1, 100);

    const mainRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    mainRenderer.setClearColor(0x2a2a4e, 1);
    mainRenderer.setSize(w, h);
    mainRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mainRenderer.shadowMap.enabled = true;
    mainRenderer.domElement.style.position = "absolute";
    mainRenderer.domElement.style.top = "0";
    mainRenderer.domElement.style.left = "0";
    container.appendChild(mainRenderer.domElement);

    const fpRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    fpRenderer.setClearColor(0x2a2a4e, 1);
    fpRenderer.setSize(fpW, fpH);
    fpRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    fpRenderer.shadowMap.enabled = true;
    fpContainer.appendChild(fpRenderer.domElement);

    // Axes gizmo (bottom-left)
    const axesContainer = axesContainerRef.current;
    let axesScene: THREE.Scene | null = null;
    let axesCamera: THREE.PerspectiveCamera | null = null;
    let axesRenderer: THREE.WebGLRenderer | null = null;
    if (axesContainer) {
      axesScene = new THREE.Scene();
      const axesGroup = new THREE.Group();
      axesGroup.add(new THREE.AxesHelper(1.0));
      axesGroup.add(makeLabelSprite("X", "#ff4444").translateX(1.15));
      axesGroup.add(makeLabelSprite("Y", "#44ff44").translateY(1.15));
      axesGroup.add(makeLabelSprite("Z", "#4444ff").translateZ(1.15));
      axesScene.add(axesGroup);
      axesGroupRef.current = axesGroup;
      const size = 120;
      const halfSize = 1.3;
      axesCamera = new THREE.OrthographicCamera(-halfSize, halfSize, halfSize, -halfSize, 0.1, 10);
      axesCamera.position.set(0, 0, 3);
      axesRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      axesRenderer.setClearColor(0x1a1a2e, 1);
      axesRenderer.setSize(size, size);
      axesRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      axesContainer.appendChild(axesRenderer.domElement);
    }
    axesCameraRef.current = axesCamera;
    axesSceneRef.current = axesScene;
    axesRendererRef.current = axesRenderer;

    const ambient = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(ambient);

    const dir = new THREE.DirectionalLight(0xffffff, 1.5);
    dir.position.set(5, 8, 3);
    dir.castShadow = true;
    dir.shadow.mapSize.set(1024, 1024);
    scene.add(dir);

    const hemi = new THREE.HemisphereLight(0x87ceeb, 0x3a3a3a, 0.7);
    scene.add(hemi);

    const grid = new THREE.GridHelper(10, 20, 0x444466, 0x222244);
    scene.add(grid);

    const axes = new THREE.AxesHelper(2);
    scene.add(axes);

    sceneRef.current = scene;
    orbitCameraRef.current = orbitCamera;
    fpCameraRef.current = fpCamera;
    mainRendererRef.current = mainRenderer;
    fpRendererRef.current = fpRenderer;
  }, []);

  const updateOrbitCamera = useCallback(() => {
    const camera = orbitCameraRef.current;
    const cs = orbitStateRef.current;
    if (!camera) return;

    const az = cs.azimuth;
    const el = cs.elevation;
    const d = cs.distance;
    const t = cs.target;

    camera.position.set(
      t.x + d * Math.cos(el) * Math.sin(az),
      t.y + d * Math.sin(el),
      t.z + d * Math.cos(el) * Math.cos(az),
    );
    camera.lookAt(t);
  }, []);

  useEffect(() => {
    setupScene();
    const handleResize = () => {
      const container = containerRef.current;
      const fpContainer = fpContainerRef.current;
      const mainRenderer = mainRendererRef.current;
      const fpRenderer = fpRendererRef.current;
      const orbitCamera = orbitCameraRef.current;
      const fpCamera = fpCameraRef.current;
      if (!container || !mainRenderer || !orbitCamera) return;

      const w = container.clientWidth;
      const h = container.clientHeight;
      mainRenderer.setSize(w, h);
      orbitCamera.aspect = w / h;
      orbitCamera.updateProjectionMatrix();

      if (fpContainer && fpRenderer && fpCamera) {
        const fpW = fpContainer.clientWidth;
        const fpH = fpContainer.clientHeight;
        fpRenderer.setSize(fpW, fpH);
        fpCamera.aspect = fpW / fpH;
        fpCamera.updateProjectionMatrix();
      }
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animRef.current);
      mainRendererRef.current?.domElement?.remove();
      mainRendererRef.current?.dispose();
      fpRendererRef.current?.domElement?.remove();
      fpRendererRef.current?.dispose();
      axesRendererRef.current?.domElement?.remove();
      axesRendererRef.current?.dispose();
    };
  }, [setupScene]);

  const prevLoadedRef = useRef(false);
  useEffect(() => {
    if (!isLoaded || !model.current || prevLoadedRef.current) return;
    prevLoadedRef.current = true;

    const m = model.current!;
    const scene = sceneRef.current;
    if (!scene) return;

    meshesRef.current.forEach((mesh) => scene.remove(mesh));
    meshesRef.current = [];

    for (let i = 0; i < m.ngeom; i++) {
      const type = m.geom_type[i];
      const size = m.geom_size.slice(i * 3, i * 3 + 3) as Float64Array;
      const rgba = m.geom_rgba.slice(i * 4, i * 4 + 4) as Float64Array;

      const mesh = createGeomMesh(type, size, rgba);
      scene.add(mesh);
      meshesRef.current.push(mesh);
    }

    const { fpIdx, tdIdx } = resolveCameraIndices(m);
    fpCamIdxRef.current = fpIdx;

    const orbitCam = orbitCameraRef.current;
    const fpCam = fpCameraRef.current;
    if (tdIdx >= 0 && orbitCam) {
      orbitCam.fov = m.cam_fovy[tdIdx];
      orbitCam.updateProjectionMatrix();
    }
    if (fpIdx >= 0 && fpCam) {
      fpCam.fov = m.cam_fovy[fpIdx];
      fpCam.updateProjectionMatrix();
    }

    updateOrbitCamera();
  }, [isLoaded, model, updateOrbitCamera]);

  useEffect(() => {
    let running = true;

    function loop() {
      if (!running) return;

      const m = mujoco.current;
      const d = data.current;
      const scene = sceneRef.current;
      const mainRenderer = mainRendererRef.current;
      const fpRenderer = fpRendererRef.current;
      const orbitCamera = orbitCameraRef.current;
      const fpCamera = fpCameraRef.current;

      if (m && d && scene) {
        onStep();

        for (let s = 1; s < SUBSTEPS; s++) {
          m.mj_step(model.current!, data.current!);
        }

        // Track car body position for orbit camera target
        const carBodyId = 1;
        const carXpos = d.xpos.slice(carBodyId * 3, carBodyId * 3 + 3) as Float64Array;
        orbitStateRef.current.target.set(carXpos[0], carXpos[2] + 0.3, -carXpos[1]);
        updateOrbitCamera();

        for (let i = 0; i < meshesRef.current.length; i++) {
          const g = d.geom(i);
          const pos = g.xpos as Float64Array;
          const mat = g.xmat as Float64Array;

          const { position, quaternion } = mjToThree(pos, mat);
          meshesRef.current[i].position.copy(position);
          meshesRef.current[i].quaternion.copy(quaternion);
        }

        // First-person camera: use MuJoCo native camera world pose
        const fpIdx = fpCamIdxRef.current;
        if (fpCamera && fpIdx >= 0) {
          const camPos = d.cam_xpos.slice(fpIdx * 3, fpIdx * 3 + 3) as Float64Array;
          const camMat = d.cam_xmat.slice(fpIdx * 9, fpIdx * 9 + 9) as Float64Array;

          const pos3 = new THREE.Vector3(camPos[0], camPos[2], -camPos[1]);

          // MuJoCo camera looks along local -Z, up is local +Y
          // cam_xmat columns are the camera's local axes in MuJoCo world coords
          const lookMj = new THREE.Vector3(-camMat[2], -camMat[5], -camMat[8]);
          const upMj = new THREE.Vector3(camMat[1], camMat[4], camMat[7]);
          const look3 = new THREE.Vector3(lookMj.x, lookMj.z, -lookMj.y);
          const up3 = new THREE.Vector3(upMj.x, upMj.z, -upMj.y);

          fpCamera.position.copy(pos3);
          fpCamera.up.copy(up3);
          fpCamera.lookAt(pos3.clone().add(look3));
        }
      }

      if (mainRenderer && orbitCamera && scene) {
        mainRenderer.render(scene, orbitCamera);
      }
      if (fpRenderer && fpCamera && scene) {
        fpRenderer.render(scene, fpCamera);
      }

      // Axes gizmo: rotate the axes group to match orbit camera view
      const axesCamera = axesCameraRef.current;
      const axesScene = axesSceneRef.current;
      const axesRenderer = axesRendererRef.current;
      const axesGroup = axesGroupRef.current;
      if (axesCamera && axesScene && axesRenderer && axesGroup && orbitCamera) {
        axesGroup.quaternion.copy(orbitCamera.quaternion).invert();
        axesRenderer.render(axesScene, axesCamera);
      }

      animRef.current = requestAnimationFrame(loop);
    }

    if (isLoaded) {
      loop();
    }

    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
    };
  }, [isLoaded, mujoco, data, onStep]);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    draggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!draggingRef.current) return;
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      lastMouseRef.current = { x: e.clientX, y: e.clientY };

      orbitStateRef.current.azimuth -= dx * 0.005;
      orbitStateRef.current.elevation += dy * 0.005;
      orbitStateRef.current.elevation = Math.max(
        -1.5,
        Math.min(1.5, orbitStateRef.current.elevation),
      );
      updateOrbitCamera();
    },
    [updateOrbitCamera],
  );

  const handlePointerUp = useCallback(() => {
    draggingRef.current = false;
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      orbitStateRef.current.distance += e.deltaY * 0.01;
      orbitStateRef.current.distance = Math.max(
        1,
        Math.min(30, orbitStateRef.current.distance),
      );
      updateOrbitCamera();
    },
    [updateOrbitCamera],
  );

  return (
    <div
      ref={containerRef}
      className="absolute inset-0"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
      style={{ touchAction: "none" }}
    >
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950 z-10">
          <p className="text-slate-400">Loading MuJoCo WASM...</p>
        </div>
      )}
      <div
        ref={axesContainerRef}
        className="absolute bottom-3 left-3 w-[120px] h-[120px] z-10 rounded-md overflow-hidden border border-slate-600/50"
      />
      <div
        ref={fpContainerRef}
        className="absolute bottom-3 right-3 w-[300px] h-[200px] border-2 border-violet-500/60 rounded-md overflow-hidden shadow-lg shadow-violet-900/30 z-10"
        style={{ pointerEvents: "none" }}
      />
    </div>
  );
}
