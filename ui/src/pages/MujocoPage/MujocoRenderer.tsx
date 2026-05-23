import { useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import type { MainModule, MjModel, MjData } from "@mujoco/mujoco";

const MJ_GEOM_PLANE = 0;
const MJ_GEOM_SPHERE = 2;
const MJ_GEOM_CYLINDER = 5;
const MJ_GEOM_BOX = 6;

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
): THREE.Mesh {
  let geometry: THREE.BufferGeometry;
  let material: THREE.MeshStandardMaterial;

  const alpha = rgba.length >= 4 ? rgba[3] : 1;
  const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  const transparent = alpha < 0.99;

  switch (type) {
    case MJ_GEOM_PLANE: {
      geometry = new THREE.PlaneGeometry(20, 20);
      geometry.rotateX(-Math.PI / 2);
      material = new THREE.MeshStandardMaterial({
        color,
        side: THREE.DoubleSide,
        roughness: 0.9,
        transparent,
        opacity: alpha,
      });
      break;
    }
    case MJ_GEOM_SPHERE:
      geometry = new THREE.SphereGeometry(size[0], 32, 32);
      material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.4,
        transparent,
        opacity: alpha,
      });
      break;
    case MJ_GEOM_CYLINDER:
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 32);
      material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.5,
        metalness: 0.3,
        transparent,
        opacity: alpha,
      });
      break;
    case MJ_GEOM_BOX:
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
      material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.5,
        metalness: 0.2,
        transparent,
        opacity: alpha,
      });
      break;
    default:
      geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
      material = new THREE.MeshStandardMaterial({ color: 0xff00ff });
  }

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
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
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const meshesRef = useRef<THREE.Mesh[]>([]);
  const frameCountRef = useRef(0);
  const animRef = useRef(0);
  const draggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });
  const camStateRef = useRef({
    azimuth: 0.5,
    elevation: 0.4,
    distance: 6,
    target: new THREE.Vector3(0, 0.3, 0),
  });

  const setupScene = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x2a2a4e);

    const w = container.clientWidth;
    const h = container.clientHeight;
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    camera.position.set(5, 4, 6);
    camera.lookAt(0, 0.3, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x2a2a4e, 1);
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.domElement.style.position = "absolute";
    renderer.domElement.style.top = "0";
    renderer.domElement.style.left = "0";
    container.appendChild(renderer.domElement);

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
    cameraRef.current = camera;
    rendererRef.current = renderer;
  }, []);

  const updateCamera = useCallback(() => {
    const camera = cameraRef.current;
    const cs = camStateRef.current;
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
      const renderer = rendererRef.current;
      const camera = cameraRef.current;
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animRef.current);
      rendererRef.current?.domElement?.remove();
      rendererRef.current?.dispose();
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

    console.log(`[MujocoRenderer] Creating ${m.ngeom} geoms`);
    for (let i = 0; i < m.ngeom; i++) {
      const type = m.geom_type[i];
      const size = m.geom_size.slice(i * 3, i * 3 + 3) as Float64Array;
      const rgba = m.geom_rgba.slice(i * 4, i * 4 + 4) as Float64Array;

      const mesh = createGeomMesh(type, size, rgba);
      scene.add(mesh);
      meshesRef.current.push(mesh);

      if (i === 0) {
        console.log(
          `[MujocoRenderer] geom[0] type=${type} size=[${size[0].toFixed(2)},${size[1].toFixed(2)},${size[2].toFixed(2)}] rgba=[${rgba[0].toFixed(2)},${rgba[1].toFixed(2)},${rgba[2].toFixed(2)},${rgba[3].toFixed(2)}]`,
        );
      }
    }
    console.log(`[MujocoRenderer] Created ${meshesRef.current.length} meshes`);

    updateCamera();
  }, [isLoaded, model, updateCamera]);

  useEffect(() => {
    let running = true;

    function loop() {
      if (!running) return;

      const m = mujoco.current;
      const d = data.current;
      const scene = sceneRef.current;

      if (m && d && scene) {
        onStep();

        if (frameCountRef.current === 0) {
          const carGeomIdx = 2; // car chassis
          const carPos = d.geom(carGeomIdx).xpos as Float64Array;
          console.log(
            `[MujocoRenderer] frame0 car chassis pos: [${carPos[0].toFixed(3)}, ${carPos[1].toFixed(3)}, ${carPos[2].toFixed(3)}]`,
          );
          frameCountRef.current = 1;
        }

        for (let i = 0; i < meshesRef.current.length; i++) {
          const g = d.geom(i);
          const pos = g.xpos as Float64Array;
          const mat = g.xmat as Float64Array;

          const { position, quaternion } = mjToThree(pos, mat);
          meshesRef.current[i].position.copy(position);
          meshesRef.current[i].quaternion.copy(quaternion);
        }
      }

      const renderer = rendererRef.current;
      const camera = cameraRef.current;
      if (renderer && camera && scene) {
        renderer.render(scene, camera);
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

      camStateRef.current.azimuth -= dx * 0.005;
      camStateRef.current.elevation += dy * 0.005;
      camStateRef.current.elevation = Math.max(
        -1.5,
        Math.min(1.5, camStateRef.current.elevation),
      );
      updateCamera();
    },
    [updateCamera],
  );

  const handlePointerUp = useCallback(() => {
    draggingRef.current = false;
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      camStateRef.current.distance += e.deltaY * 0.01;
      camStateRef.current.distance = Math.max(
        1,
        Math.min(30, camStateRef.current.distance),
      );
      updateCamera();
    },
    [updateCamera],
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
    </div>
  );
}
