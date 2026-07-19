import { useEffect, useRef, useCallback, type RefObject } from "react";
import * as THREE from "three";
import type { MainModule, MjModel, MjData } from "@mujoco/mujoco";

const MJ_GEOM_PLANE = 0;
const MJ_GEOM_SPHERE = 2;
const MJ_GEOM_CYLINDER = 5;
const MJ_GEOM_BOX = 6;

const SUBSTEPS = 5;

// Coordinate conversion: MuJoCo (x,y,z) → Three.js (x, z, -y)
function mjPosToThree(buf: Float64Array, index: number, target: THREE.Vector3): THREE.Vector3 {
  return target.set(buf[index * 3 + 0], buf[index * 3 + 2], -buf[index * 3 + 1]);
}

// Quaternion conversion: MuJoCo [w,x,y,z] → Three.js (-x, -z, y, -w)
function mjQuatToThree(buf: Float64Array, index: number, target: THREE.Quaternion): THREE.Quaternion {
  return target.set(-buf[index * 4 + 1], -buf[index * 4 + 3], buf[index * 4 + 2], -buf[index * 4 + 0]);
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
      const geometry = new THREE.PlaneGeometry(size[0] * 2, size[1] * 2);
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

      const group = new THREE.Group();
      group.add(cylMesh);

      // Cross-spokes only for wheel-like cylinders (radius > halfHeight)
      if (radius >= halfHeight) {
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
        group.add(spokeX);
        group.add(spokeZ);
      }

      mesh = group;
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

interface JointOverlay {
  cylinder: THREE.Mesh;
  sprite: THREE.Sprite;
  jointIdx: number;
  name: string;
  qposAdr: number;
}

interface Props {
  mujoco: React.RefObject<MainModule | null>;
  model: React.RefObject<MjModel | null>;
  data: React.RefObject<MjData | null>;
  isLoaded: boolean;
  onStep: () => void;
  showJointOverlay: boolean;
  onPlaceBall?: (position: { x: number; y: number }) => void;
  onFirstPersonCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
  firstPersonContainerRef?: RefObject<HTMLDivElement | null>;
}

export default function MujocoRenderer({
  mujoco,
  model,
  data,
  isLoaded,
  onStep,
  showJointOverlay,
  onPlaceBall,
  onFirstPersonCanvasReady,
  firstPersonContainerRef,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fallbackFpContainerRef = useRef<HTMLDivElement>(null);
  const axesContainerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const orbitCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const fpCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const axesCameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const axesSceneRef = useRef<THREE.Scene | null>(null);
  const axesGroupRef = useRef<THREE.Group | null>(null);
  const mainRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const fpRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const axesRendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const bodyGroupsRef = useRef<THREE.Group[]>([]);
  const animRef = useRef(0);
  const draggingRef = useRef(false);
  const draggingBallRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });
  const orbitStateRef = useRef({
    azimuth: 0.5,
    elevation: 0.4,
    distance: 6,
    target: new THREE.Vector3(0, 0.3, 0),
  });
  const fpCamIdxRef = useRef(-1);
  const fpCamBodyIdRef = useRef(-1);
  const carBodyIdRef = useRef(-1);
  const jointGroupRef = useRef<THREE.Group | null>(null);
  const jointOverlaysRef = useRef<JointOverlay[]>([]);
  const jointFrameCountRef = useRef(0);
  const prevLoadedRef = useRef(false);
  const showJointOverlayRef = useRef(showJointOverlay);

  useEffect(() => {
    showJointOverlayRef.current = showJointOverlay;
  }, [showJointOverlay]);

  const getRaycasterFromPointer = useCallback((e: React.PointerEvent) => {
    const container = containerRef.current;
    const camera = orbitCameraRef.current;
    if (!container || !camera) return null;

    const rect = container.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, camera);
    return raycaster;
  }, []);

  const findTargetBallGroup = useCallback(() => {
    return bodyGroupsRef.current.find((group) => group.name === "target_ball") ?? null;
  }, []);

  const isPointerOnTargetBall = useCallback((e: React.PointerEvent) => {
    const ballGroup = findTargetBallGroup();
    const raycaster = getRaycasterFromPointer(e);
    if (!ballGroup || !raycaster) return false;

    return raycaster.intersectObject(ballGroup, true).length > 0;
  }, [findTargetBallGroup, getRaycasterFromPointer]);

  const placeBallFromPointer = useCallback((e: React.PointerEvent) => {
    const raycaster = getRaycasterFromPointer(e);
    if (!raycaster || !onPlaceBall) return;

    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const hitPoint = new THREE.Vector3();
    if (!raycaster.ray.intersectPlane(groundPlane, hitPoint)) return;

    onPlaceBall({
      x: hitPoint.x,
      y: -hitPoint.z,
    });
  }, [getRaycasterFromPointer, onPlaceBall]);

  const setupScene = useCallback(() => {
    const container = containerRef.current;
    const fpContainer = firstPersonContainerRef?.current ?? fallbackFpContainerRef.current;
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

    const fpRenderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    fpRenderer.setClearColor(0x2a2a4e, 1);
    fpRenderer.setSize(fpW, fpH);
    fpRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    fpRenderer.shadowMap.enabled = true;
    fpContainer.appendChild(fpRenderer.domElement);
    onFirstPersonCanvasReady?.(fpRenderer.domElement);

    // Axes gizmo (bottom-left)
    const axesContainer = axesContainerRef.current;
    let axesScene: THREE.Scene | null = null;
    let axesCamera: THREE.OrthographicCamera | null = null;
    let axesRenderer: THREE.WebGLRenderer | null = null;
    if (axesContainer) {
      axesScene = new THREE.Scene();
      const axesGroup = new THREE.Group();
      axesGroup.add(new THREE.AxesHelper(1.0));
      axesGroup.add(makeLabelSprite("X", "#ff4444").translateX(1.15));
      axesGroup.add(makeLabelSprite("Z", "#4444ff").translateY(1.15));
      axesGroup.add(makeLabelSprite("Y", "#44ff44").translateZ(1.15));
      axesScene.add(axesGroup);
      axesGroupRef.current = axesGroup;
      const size = 120;
      const halfSize = 1.3;
      const orthographicCamera = new THREE.OrthographicCamera(-halfSize, halfSize, halfSize, -halfSize, 0.1, 10);
      orthographicCamera.position.set(0, 0, 3);
      axesCamera = orthographicCamera;
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

    const grid = new THREE.GridHelper(20, 40, 0x444466, 0x222244);
    scene.add(grid);

    const axes = new THREE.AxesHelper(2);
    scene.add(axes);

    sceneRef.current = scene;
    orbitCameraRef.current = orbitCamera;
    fpCameraRef.current = fpCamera;
    mainRendererRef.current = mainRenderer;
    fpRendererRef.current = fpRenderer;
  }, [firstPersonContainerRef, onFirstPersonCanvasReady]);

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
      const fpContainer = firstPersonContainerRef?.current ?? fallbackFpContainerRef.current;
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
      onFirstPersonCanvasReady?.(null);
      prevLoadedRef.current = false;
    };
  }, [firstPersonContainerRef, setupScene, onFirstPersonCanvasReady]);

  useEffect(() => {
    if (!isLoaded || !model.current || prevLoadedRef.current) return;
    prevLoadedRef.current = true;

    const m = model.current!;
    const scene = sceneRef.current;
    if (!scene) return;

    // Remove old body groups
    bodyGroupsRef.current.forEach((g) => scene.remove(g));
    bodyGroupsRef.current = [];

    // Create body groups, all as direct children of world (body 0)
    // We use world positions from d.xpos, so no parent-child body hierarchy needed
    const bodyGroups: THREE.Group[] = [];
    for (let b = 0; b < m.nbody; b++) {
      const group = new THREE.Group();
      const nameAddr = m.name_bodyadr?.[b] ?? -1;
      group.name = nameAddr >= 0 ? resolveName(m.names, nameAddr) : `body_${b}`;
      bodyGroups.push(group);
      if (b === 0) {
        scene.add(group);
      } else {
        bodyGroups[0].add(group);
      }
    }

    // Create geom meshes and attach to body groups with LOCAL pos/quat
    for (let i = 0; i < m.ngeom; i++) {
      const type = m.geom_type[i];
      const size = m.geom_size.slice(i * 3, i * 3 + 3) as Float64Array;
      const rgba = m.geom_rgba.slice(i * 4, i * 4 + 4) as Float64Array;
      const bodyId = m.geom_bodyid[i];

      const mesh = createGeomMesh(type, size, rgba);
      mjPosToThree(m.geom_pos, i, mesh.position);
      if (type !== MJ_GEOM_PLANE) {
        mjQuatToThree(m.geom_quat, i, mesh.quaternion);
      }
      bodyGroups[bodyId].add(mesh);
    }

    bodyGroupsRef.current = bodyGroups;

    const { fpIdx, tdIdx } = resolveCameraIndices(m);
    fpCamIdxRef.current = fpIdx;
    fpCamBodyIdRef.current = fpIdx >= 0 ? (m.cam_bodyid?.[fpIdx] ?? -1) : -1;
    carBodyIdRef.current = bodyGroups.findIndex((group) => group.name === "car");

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

    // Joint overlay group
    const jointGroup = new THREE.Group();
    jointGroup.visible = showJointOverlay;
    scene.add(jointGroup);
    jointGroupRef.current = jointGroup;
    const overlays: JointOverlay[] = [];

    for (let i = 0; i < m.njnt; i++) {
      const jtype = m.jnt_type[i];
      if (jtype !== 3) continue; // only hinge joints for now

      const name = resolveName(m.names, m.name_jntadr[i]);

      const qposAdr = m.jnt_qposadr[i];

      // axis cylinder - use MeshBasicMaterial for guaranteed visibility
      const cylGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.5, 8);
      const cylMat = new THREE.MeshBasicMaterial({
        color: 0xff4444,
        depthTest: false,
        depthWrite: false,
      });
      const cylinder = new THREE.Mesh(cylGeo, cylMat);
      cylinder.renderOrder = 999;

      // label sprite
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 64;
      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.scale.set(0.6, 0.15, 1);

      jointGroup.add(cylinder);
      jointGroup.add(sprite);
      overlays.push({ cylinder, sprite, jointIdx: i, name, qposAdr });
    }
    jointOverlaysRef.current = overlays;

    updateOrbitCamera();
  }, [isLoaded, model, showJointOverlay, updateOrbitCamera]);

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

        // Update body group transforms from MuJoCo world data
        const bodyGroups = bodyGroupsRef.current;
        for (let b = 0; b < bodyGroups.length; b++) {
          mjPosToThree(d.xpos, b, bodyGroups[b].position);
          mjQuatToThree(d.xquat, b, bodyGroups[b].quaternion);
        }

        // Track car body position for orbit camera target
        const carBodyId = carBodyIdRef.current;
        const carPos3 = new THREE.Vector3();
        if (carBodyId >= 0) {
          mjPosToThree(d.xpos, carBodyId, carPos3);
          orbitStateRef.current.target.set(carPos3.x, carPos3.y + 0.3, carPos3.z);
          updateOrbitCamera();
        }

        // First-person camera: reuse body group transform (same proven quaternion path)
        const fpIdx = fpCamIdxRef.current;
        const camBodyId = fpCamBodyIdRef.current;
        if (fpCamera && fpIdx >= 0 && camBodyId >= 0 && bodyGroups[camBodyId]) {
          fpCamera.position.copy(bodyGroups[camBodyId].position);
          fpCamera.quaternion.copy(bodyGroups[camBodyId].quaternion);
        }

        // Joint overlay update
        const jointGroup = jointGroupRef.current;
        const overlays = jointOverlaysRef.current;
        if (jointGroup) {
          const show = showJointOverlayRef.current;
          jointGroup.visible = show;
          if (show && overlays.length > 0) {
            jointFrameCountRef.current++;
            const updateLabels = jointFrameCountRef.current % 30 === 0;
            for (const ov of overlays) {
              const ji = ov.jointIdx;
              const jnt = d.jnt(ji);

              const anchorRaw = jnt.xanchor;
              const axisRaw = jnt.xaxis;
              const anchor = (anchorRaw instanceof Float64Array || ArrayBuffer.isView(anchorRaw))
                ? new Float64Array(anchorRaw.buffer, anchorRaw.byteOffset, 3)
                : (Array.isArray(anchorRaw) ? anchorRaw : [0, 0, 0]) as unknown as Float64Array;
              const axis = (axisRaw instanceof Float64Array || ArrayBuffer.isView(axisRaw))
                ? new Float64Array(axisRaw.buffer, axisRaw.byteOffset, 3)
                : (Array.isArray(axisRaw) ? axisRaw : [0, 1, 0]) as unknown as Float64Array;

              const posT = new THREE.Vector3(anchor[0], anchor[2], -anchor[1]);
              const axT = new THREE.Vector3(axis[0], axis[2], -axis[1]).normalize();

              ov.cylinder.position.copy(posT);
              ov.cylinder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axT);

              ov.sprite.position.copy(posT).add(new THREE.Vector3(0, 0.25, 0));

              if (updateLabels) {
                const qposVal = d.qpos[ov.qposAdr]?.toFixed(3) ?? "?";
                const canvas = (ov.sprite.material as THREE.SpriteMaterial).map?.image as HTMLCanvasElement;
                if (canvas) {
                  const labelCtx = canvas.getContext("2d");
                  if (labelCtx) {
                    labelCtx.clearRect(0, 0, canvas.width, canvas.height);
                    labelCtx.fillStyle = "rgba(0,0,0,0.75)";
                    labelCtx.fillRect(0, 0, canvas.width, canvas.height);
                    labelCtx.fillStyle = "#ffffff";
                    labelCtx.font = "20px monospace";
                    labelCtx.textAlign = "center";
                    labelCtx.textBaseline = "middle";
                    labelCtx.fillText(`${ov.name}: ${qposVal}`, 128, 32);
                    (ov.sprite.material as THREE.SpriteMaterial).map!.needsUpdate = true;
                  }
                }
              }
            }
          }
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
  }, [isLoaded, mujoco, data, model, onStep, updateOrbitCamera]);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (isPointerOnTargetBall(e)) {
      draggingBallRef.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      placeBallFromPointer(e);
      return;
    }

    draggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [isPointerOnTargetBall, placeBallFromPointer]);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (draggingBallRef.current) {
        placeBallFromPointer(e);
        return;
      }

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
    [placeBallFromPointer, updateOrbitCamera],
  );

  const handlePointerUp = useCallback(() => {
    if (draggingBallRef.current) {
      draggingBallRef.current = false;
      return;
    }

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
      {!firstPersonContainerRef && (
        <div
          ref={fallbackFpContainerRef}
          className="absolute bottom-3 right-3 w-[300px] h-[200px] border-2 border-violet-500/60 rounded-md overflow-hidden shadow-lg shadow-violet-900/30 z-10"
          style={{ pointerEvents: "none" }}
        />
      )}
    </div>
  );
}
