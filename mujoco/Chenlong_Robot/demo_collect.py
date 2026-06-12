#!/usr/bin/env python3
"""
demo_collect.py — Mobile Manipulator: All-in-One Demo + Data Collection
========================================================================
A single, self-contained file that covers MuJoCo concepts No.1 through
No.13 and performs automatic pick-and-place with imitation-learning data
collection.

Concepts exercised (one file, 12 lessons):
  No.1  : Model loading + basic simulation loop (MjModel, MjData, mj_step)
  No.2  : GLFW rendering pipeline (camera, scene, context, mouse interaction)
  No.3  : Position servos — PD control on each arm joint via <position> actuators
  No.4  : Dynamics extraction — mj_fullM + qfrc_bias for mass-matrix diagnostics
  No.5  : Finite state machine + cubic polynomial trajectory interpolation
  No.6  : Jacobian-based inverse kinematics (mj_jac → pseudo-inverse)
  No.7  : State-feedback control — linearize arm targets around current pose
  No.8  : Equality-constraint management — activate/deactivate grasp_weld
  No.9  : Contact / position-triggered FSM transitions
  No.11 : Numerical IK via damped Jacobian pseudo-inverse iteration
  No.12 : Separate FK model instance + matplotlib post-simulation summary
  No.13 : Quaternion-based state estimation + camera tracking

Task FSM (8 active states):
  DRIVE_TO_TARGET → REACH → LOWER → GRASP → LIFT → DRIVE_BACK → PLACE → RELEASE → DONE

Data collection:
  Saves all episodes as a single timestamped .npz file under ./episodes/
  Keys: joint_states, ee_position, target_position, actions, fsm_state, timestamps
  Sampling: 10 Hz

Run:
    python3 demo_collect.py             # GUI window + data collection (macOS)
    mjpython demo_collect.py            # Linux / non-macOS
    python3 demo_collect.py --headless  # no window, collect data only

Controls:
    Mouse drag          rotate / pan / zoom
    Backspace           reset simulation (restarts task)
    S                   save accumulated data immediately
    Q / Esc             quit
"""

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys
import time as _time

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive — renders to file, avoids macOS Tk crash
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ═══════════════════════════════════════════════════════════════════════════════
# Constants & geometry  (must match car.xml)
# ═══════════════════════════════════════════════════════════════════════════════
HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "car.xml")
OUT_DIR = os.path.join(HERE, "episodes")
os.makedirs(OUT_DIR, exist_ok=True)

# World positions
TARGET_POS = np.array([1.5, 0.0, 0.04])   # box on ground at (1.5, 0, 0.04)
PLACE_POS  = np.array([0.25, 0.0, 0.55])  # where to put it (in front of origin)

# Arm link lengths (meters) — must match car.xml (2× scale)
L_UPPER   = 0.50   # upper_arm length
L_FOREARM = 0.40   # forearm length
L_WRIST   = 0.16   # wrist length
L_GRIPPER = 0.08   # gripper half-extent
L2_EFF    = L_FOREARM + L_WRIST + L_GRIPPER  # 0.64 — effective forearm+wrist+gripper

# Control gains
DRIVE_SPEED   = 0.5    # wheel ctrl while driving
BRAKE_TORQUE  = 3.0    # braking torque in non-driving states
REACH_TOL     = 0.18   # EE position tolerance (m)
TRAJ_DURATION = 0.6    # cubic trajectory duration (s)
DRIVE_TARGET_X = 1.05  # car x position that triggers arm reach
COLLECT_HZ    = 10     # data sampling rate
NUM_EPISODES  = 5      # number of task repetitions (use --episodes N to override)

# Home / stowed arm pose (joint angles, radians)
STOWED = np.array([0.0, 0.5, -1.0, 0.0, 0.04, 0.04])  # [pan, lift, elbow, wrist, finger_l, finger_r]

# Joint limits (radians / meters for gripper slide) — must match car.xml joint ranges
Q_MIN = np.array([-6.28, -6.28, -6.28, -6.28])  # ±360° (matches car.xml)
Q_MAX = np.array([ 6.28,  6.28,  6.28,  6.28])

# ═══════════════════════════════════════════════════════════════════════════════
# FSM state machine  (No.5: FSM pattern    No.9: position-triggered transitions)
# ═══════════════════════════════════════════════════════════════════════════════
(FSM_DRIVE, FSM_REACH, FSM_LOWER, FSM_GRASP, FSM_LIFT,
 FSM_DRIVE_BACK, FSM_PLACE, FSM_RELEASE, FSM_DONE) = range(9)

FSM_NAMES = ["DRIVE", "REACH", "LOWER", "GRASP", "LIFT",
             "DRIVE_BACK", "PLACE", "RELEASE", "DONE"]

# Safety timeouts — prevent indefinite hangs
FSM_TIMEOUT = {FSM_DRIVE: 15, FSM_REACH: 8, FSM_LOWER: 8, FSM_GRASP: 3,
               FSM_LIFT: 8, FSM_DRIVE_BACK: 15, FSM_PLACE: 5, FSM_RELEASE: 3,
               FSM_DONE: float("inf")}

# ═══════════════════════════════════════════════════════════════════════════════
# Cached model IDs (populated once in main)
# ═══════════════════════════════════════════════════════════════════════════════
_ids = {}  # string → id: car_body, arm_base_body, target_body, ee_site, gripper_body
_arm_qpos_adr = []   # 5 ints: qpos addresses for pan/lift/elbow/wrist/gripper
_arm_dof_adr  = []   # 5 ints: dof  addresses for the same joints

def _cache_ids(m):
    """One-time lookup of all body/site/joint IDs.  (No.1: model introspection)"""
    for name in ("car", "arm_base", "target_box"):
        _ids[name] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
    _ids["ee"]    = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "end_effector")
    _ids["gripper"] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper_palm")

    arm_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch",
                 "finger_l_j", "finger_r_j"]
    global _arm_qpos_adr, _arm_dof_adr
    _arm_qpos_adr = [m.jnt_qposadr[m.joint(n).id] for n in arm_names]
    _arm_dof_adr  = [m.jnt_dofadr[m.joint(n).id]  for n in arm_names]

# ═══════════════════════════════════════════════════════════════════════════════
# State accessors  (No.13: position-based state estimation)
# ═══════════════════════════════════════════════════════════════════════════════
def _car_pos(d):     return d.xpos[_ids["car"]]
def _arm_base_pos(d): return d.xpos[_ids["arm_base"]]
def _target_pos(d):  return d.xpos[_ids["target_box"]]
def _ee_pos(d):      return d.site_xpos[_ids["ee"]]
def _arm_qpos(d):    return np.array([d.qpos[a] for a in _arm_qpos_adr])
def _car_quat(d):    return d.xquat[_ids["car"]].copy()  # (w, x, y, z)  (No.13: quaternion state)

# ═══════════════════════════════════════════════════════════════════════════════
# No.6 : Analytical 2D IK (deterministic, always converges when reachable)
#   Used as seed for the numerical IK to guarantee fast convergence.
# ═══════════════════════════════════════════════════════════════════════════════
def _analytical_ik_2d(x, z, L1=L_UPPER, L2=L2_EFF):
    """2-link IK in the (x, z) plane. Returns (lift, elbow, wrist)."""
    D = np.hypot(x, z)
    max_r = L1 + L2
    if D > max_r * 0.95:
        s = max_r * 0.95 / D
        x, z = x * s, z * s
    D2 = x*x + z*z
    cos_q2 = np.clip((D2 - L1*L1 - L2*L2) / (2*L1*L2), -1.0, 1.0)
    q2_int = np.arccos(cos_q2)
    alpha = np.arctan2(z, x)
    beta  = np.arctan2(L2 * np.sin(q2_int), L1 + L2 * np.cos(q2_int))
    q1 = -(alpha - beta)
    q2 = -(np.pi - q2_int)
    q3 = -(q1 + q2)
    return q1, q2, q3

# ═══════════════════════════════════════════════════════════════════════════════
# No.6 + No.11 : Jacobian-based numerical IK
#   Uses a separate MjData for side-effect-free FK queries (No.12 pattern).
#   Seeded by analytical 2D IK for fast, reliable convergence.
# ═══════════════════════════════════════════════════════════════════════════════
_fk_data = None   # separate MjData for IK queries

def numerical_ik(target_world, arm_base_pos, q_guess=None, max_iter=40,
                 tol=0.015, alpha=0.5, gripper_cmd=0.0):
    """4-DOF arm IK: analytical 2D seed + Jacobian refinement (No.6 + No.11).

    Returns [pan, lift, elbow, wrist, gripper_cmd].
    Falls back gracefully to analytical IK if numerical refinement is unavailable.
    """
    m, d = model, _fk_data
    local = target_world - arm_base_pos
    r_xy = np.hypot(local[0], local[1])
    pan0 = float(np.clip(np.arctan2(local[1], local[0]), -1.57, 1.57))
    lift0, elbow0, wrist0 = _analytical_ik_2d(r_xy, local[2])
    q = np.clip([pan0, lift0, elbow0, wrist0], Q_MIN, Q_MAX)

    if q_guess is not None:
        q = np.clip(0.3 * np.array(q_guess[:4]) + 0.7 * q, Q_MIN, Q_MAX)

    # If FK model unavailable, return analytical result directly
    if m is None or d is None:
        return np.array([q[0], q[1], q[2], q[3], gripper_cmd, gripper_cmd])

    # Jacobian refinement (No.11: numerical optimisation)
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = data.qvel[:]
    mj.mj_fwdPosition(m, d)

    for _ in range(max_iter):
        for adr, val in zip(_arm_qpos_adr, [q[0], q[1], q[2], q[3], gripper_cmd, gripper_cmd]):
            d.qpos[adr] = val
        mj.mj_fwdPosition(m, d)
        ee = d.site_xpos[_ids["ee"]]
        err = target_world - ee[:3]
        if np.linalg.norm(err) < tol:
            break

        jacp = np.zeros((3, m.nv))
        mj.mj_jac(m, d, jacp, None, ee[:3], _ids["gripper"])
        J = np.zeros((3, 4))
        for i, adr in enumerate(_arm_dof_adr[:4]):
            J[:, i] = jacp[:, adr]

        lam = 0.05
        dq = np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ err) * alpha
        q = np.clip(q + dq, Q_MIN, Q_MAX)
        q[3] = float(np.clip(-(q[1] + q[2]), -1.57, 1.57))

    q[3] = float(np.clip(-(q[1] + q[2]), -1.57, 1.57))
    return np.array([q[0], q[1], q[2], q[3], gripper_cmd, gripper_cmd])

# ═══════════════════════════════════════════════════════════════════════════════
# No.5 : Cubic polynomial trajectory generation
#   q(t) = a0 + a1·t + a2·t² + a3·t³   with  dq(0)=0, dq(T)=0
# ═══════════════════════════════════════════════════════════════════════════════
def _cubic_coeffs(q0, qf, T):
    """Return [a0, a1, a2, a3] for one joint."""
    return np.array([q0, 0.0, 3*(qf - q0)/(T*T), -2*(qf - q0)/(T*T*T)])

def _eval_cubic(c, t):
    """Evaluate cubic at time t; returns (position, velocity)."""
    pos = c[0] + c[1]*t + c[2]*t*t + c[3]*t*t*t
    vel = c[1] + 2*c[2]*t + 3*c[3]*t*t
    return pos, vel

# ═══════════════════════════════════════════════════════════════════════════════
# No.4 : Dynamics extraction (diagnostic only)
#   Extracts the 15×15 mass matrix and bias forces once per phase change.
# ═══════════════════════════════════════════════════════════════════════════════
def _log_mass_matrix_diag():
    """Log the diagonal of the joint-space inertia matrix.  (No.4 concept)"""
    M = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, M, data.qM)
    diag = np.diag(M)
    print(f"[No.4 Dynamics] mass-matrix diagonal (nv={model.nv}): "
          f"{np.array2string(diag, precision=2, max_line_width=120)}")

# ═══════════════════════════════════════════════════════════════════════════════
# No.8 : Equality-constraint grasp management
#   Activates / deactivates the grasp_weld defined in car.xml.
# ═══════════════════════════════════════════════════════════════════════════════
def _grasp(activate):
    """Toggle the weld equality between gripper and target_box."""
    eq = model.eq("grasp_weld")
    if eq is None:
        print("[GRASP] ERROR: 'grasp_weld' not found in model")
        return
    eq.active0[0] = 1 if activate else 0
    print(f"[No.8 Constraint] grasp_weld {'ACTIVATED' if activate else 'RELEASED'}")

# ═══════════════════════════════════════════════════════════════════════════════
# DataCollector  —  records images + states + actions at 10 Hz
# ═══════════════════════════════════════════════════════════════════════════════
class DataCollector:
    """Buffers simulation data and saves as .npz on completion."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.joint_states   = []   # list of (13,) float
        self.ee_positions   = []   # list of (3,) float
        self.target_positions = [] # list of (3,) float
        self.actions        = []   # list of (10,) float
        self.sensordata     = []   # list of (nsensordata,) float — raw sensor readings
        self.fsm_states     = []   # list of int
        self.timestamps     = []   # list of float
        self._last_coll_t   = -0.2

    def maybe_record(self, t, fsm, ee, target, last_action):
        """Record a frame if the collection interval has elapsed."""
        if t - self._last_coll_t < 1.0 / COLLECT_HZ - 1e-6:
            return
        self._last_coll_t = t
        # Store car pose (3D pos + 4D quat) + arm qpos (5D) = 12D
        car_pose = np.concatenate([_car_pos(data), _car_quat(data)])
        self.joint_states.append(np.concatenate([car_pose, _arm_qpos(data)]))
        self.ee_positions.append(ee.copy())
        self.target_positions.append(target.copy())
        self.actions.append(last_action.copy())
        self.sensordata.append(data.sensordata.copy())
        self.fsm_states.append(fsm)
        self.timestamps.append(t)

    @property
    def frame_count(self):
        return len(self.timestamps)

    def save(self):
        """Write accumulated data to a timestamped .npz file."""
        if self.frame_count == 0:
            print("[COLLECT] No data to save.")
            return None
        ts = _time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUT_DIR, f"ep_{ts}.npz")
        np.savez_compressed(
            path,
            joint_states=np.array(self.joint_states, dtype=np.float32),
            ee_position=np.array(self.ee_positions, dtype=np.float32),
            target_position=np.array(self.target_positions, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
            sensordata=np.array(self.sensordata, dtype=np.float32),
            fsm_state=np.array(self.fsm_states, dtype=np.int8),
            timestamps=np.array(self.timestamps, dtype=np.float32),
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"\n[COLLECT] Saved {self.frame_count} frames ({size_mb:.1f} MB) → {path}")
        return path

# ═══════════════════════════════════════════════════════════════════════════════
# SimState  —  mutable state shared between controller and main loop
# ═══════════════════════════════════════════════════════════════════════════════
class SimState:
    def __init__(self):
        self.fsm = FSM_DRIVE
        self.fsm_enter = 0.0
        self.last_print = -1.0
        self.frozen_arm = None       # cached arm pose during DRIVE_BACK
        self.last_action = np.zeros(10, dtype=float)
        # Cubic trajectory state  (No.5)
        self._traj_coeffs = None     # (5, 4) array or None
        self._traj_t0 = 0.0
        self._traj_T  = TRAJ_DURATION
        self._cached_target = STOWED.copy()  # cached IK result — recompute on FSM change only
        self._last_ik_time = -1.0             # last IK recompute time (for periodic refresh)

# Global singletons (set up in main)
model = data = None
state = SimState()
collector = DataCollector()
_main_cam_handle = None
_main_opt = _main_scene = _main_context = None

# ═══════════════════════════════════════════════════════════════════════════════
# No.7 : Arm-target computation (state-feedback)
#   Computes desired arm joint angles for the current FSM phase.
#   Called ONLY on FSM transitions; result is cached and smoothed by cubic traj.
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_arm_target(fsm, st):
    """Compute and cache the IK arm target for this FSM state (expensive — call once)."""
    arm_base = _arm_base_pos(data)
    cur_q = _arm_qpos(data)

    if fsm == FSM_DRIVE:
        st._cached_target = STOWED.copy()
        return

    if fsm == FSM_REACH:
        goal = TARGET_POS.copy(); goal[2] += 0.15
        st._cached_target = numerical_ik(goal, arm_base, cur_q, gripper_cmd=0.04)  # open fingers
        return

    if fsm in (FSM_LOWER, FSM_GRASP):
        goal = TARGET_POS.copy(); goal[2] = max(TARGET_POS[2], _target_pos(data)[2] + 0.10)
        st._cached_target = numerical_ik(goal, arm_base, cur_q, gripper_cmd=0.0)   # close fingers
        return

    if fsm == FSM_LIFT:
        goal = TARGET_POS.copy(); goal[2] += 0.15
        st._cached_target = numerical_ik(goal, arm_base, cur_q, gripper_cmd=0.0)   # hold
        return

    if fsm == FSM_DRIVE_BACK:
        if st.frozen_arm is None:
            goal = TARGET_POS.copy(); goal[2] += 0.15
            st.frozen_arm = numerical_ik(goal, arm_base, cur_q, gripper_cmd=0.0).copy()
        st._cached_target = st.frozen_arm
        return

    if fsm == FSM_PLACE:
        st._cached_target = numerical_ik(PLACE_POS, arm_base, cur_q, gripper_cmd=0.0)  # hold
        return

    if fsm == FSM_RELEASE:
        st._cached_target = numerical_ik(PLACE_POS, arm_base, cur_q, gripper_cmd=0.04)  # open
        return

    st._cached_target = STOWED.copy()

# ═══════════════════════════════════════════════════════════════════════════════
# FSM transition evaluation  (No.9 + No.13)
# ═══════════════════════════════════════════════════════════════════════════════
def _check_transitions(st):
    """Evaluate FSM transition conditions; return next state (may be same)."""
    car_x  = _car_pos(data)[0]
    ee     = _ee_pos(data)
    elap   = data.time - st.fsm_enter
    fsm    = st.fsm

    # Timeout fallback (forces progress)
    if elap > FSM_TIMEOUT.get(fsm, float("inf")):
        print(f"[TIMEOUT] {FSM_NAMES[fsm]} exceeded {FSM_TIMEOUT[fsm]:.0f}s — advancing")
        order = {FSM_DRIVE: FSM_REACH, FSM_REACH: FSM_LOWER, FSM_LOWER: FSM_GRASP,
                 FSM_GRASP: FSM_LIFT, FSM_LIFT: FSM_DRIVE_BACK,
                 FSM_DRIVE_BACK: FSM_PLACE, FSM_PLACE: FSM_RELEASE, FSM_RELEASE: FSM_DONE}
        return order.get(fsm, fsm + 1)

    if fsm == FSM_DRIVE:
        if car_x > DRIVE_TARGET_X:
            return FSM_REACH

    elif fsm == FSM_REACH:
        goal = TARGET_POS.copy(); goal[2] += 0.15
        if np.linalg.norm(ee - goal) < REACH_TOL:
            return FSM_LOWER

    elif fsm == FSM_LOWER:
        goal = TARGET_POS.copy(); goal[2] = max(TARGET_POS[2], _target_pos(data)[2] + 0.10)
        if np.linalg.norm(ee - goal) < REACH_TOL:
            return FSM_GRASP

    elif fsm == FSM_GRASP:
        if elap > 1.0:
            return FSM_LIFT

    elif fsm == FSM_LIFT:
        goal = TARGET_POS.copy(); goal[2] += 0.15
        if np.linalg.norm(ee - goal) < REACH_TOL:
            st.frozen_arm = None
            return FSM_DRIVE_BACK

    elif fsm == FSM_DRIVE_BACK:
        # Stop before overshooting origin
        if car_x < 0.30:
            return FSM_PLACE

    elif fsm == FSM_PLACE:
        if np.linalg.norm(ee - PLACE_POS) < REACH_TOL:
            return FSM_RELEASE

    elif fsm == FSM_RELEASE:
        if elap > 0.5:
            return FSM_DONE

    return fsm

# ═══════════════════════════════════════════════════════════════════════════════
# Controller callback — called every mj_step  (No.2 + No.3 + No.5 + No.7)
#   Heavy IK computation only happens on FSM transitions (cached).
#   Cubic trajectory interpolation provides smooth motion between waypoints.
# ═══════════════════════════════════════════════════════════════════════════════
def controller(m, d):
    global state
    fsm = state.fsm

    # ── Car: proportional drive toward target, brake in non-driving states ──
    car_x = _car_pos(data)[0]
    if fsm == FSM_DRIVE:
        # Proportional speed: slow down as we approach the target
        dist = max(0, DRIVE_TARGET_X - car_x)
        speed = min(DRIVE_SPEED, 0.8 * dist + 0.15)
        d.ctrl[0:4] = max(0.1, speed)  # minimum drive to overcome friction
    elif fsm == FSM_DRIVE_BACK:
        dist = max(0, car_x - 0.3)  # distance to origin
        speed = min(DRIVE_SPEED, 0.8 * dist + 0.15)
        d.ctrl[0:4] = -max(0.1, speed)
    else:
        # Active braking via velocity feedback  (No.7: state-feedback pattern)
        wheel_dof_start = _arm_dof_adr[-1] + 1  # 6 (car free) + 5 (arm) = 11
        for i in range(4):
            wv = d.qvel[wheel_dof_start + i]
            d.ctrl[i] = -np.sign(wv) * BRAKE_TORQUE if abs(wv) > 0.01 else 0.0

    # ── Arm: track cached target via cubic trajectory  (No.5) ─────────────
    # Periodically recompute IK during arm-movement states so the arm stays
    # on-target even as the car drifts after braking  (No.7: state-feedback)
    if fsm in (FSM_REACH, FSM_LOWER, FSM_LIFT, FSM_PLACE):
        if data.time - state._last_ik_time > 0.5:
            old_target = state._cached_target.copy()
            _compute_arm_target(fsm, state)
            state._last_ik_time = data.time
            # If target changed noticeably, restart cubic trajectory
            if np.linalg.norm(state._cached_target - old_target) > 0.03:
                cur_q = _arm_qpos(data)
                state._traj_coeffs = np.array([_cubic_coeffs(
                    cur_q[j], state._cached_target[j], TRAJ_DURATION) for j in range(6)])
                state._traj_t0 = data.time
                state._traj_T  = TRAJ_DURATION

    cur_q = _arm_qpos(data)

    # Evaluate trajectory or hold at target
    if (state._traj_coeffs is not None and
        data.time - state._traj_t0 < state._traj_T):
        t = data.time - state._traj_t0
        arm_cmd = np.array([_eval_cubic(state._traj_coeffs[j], t)[0] for j in range(6)])
    else:
        arm_cmd = state._cached_target
        state._traj_coeffs = None

    # Write arm position-servo targets  (No.3: PD via <position> actuators)
    for i in range(6):
        d.ctrl[4 + i] = arm_cmd[i]

    # Store action for data collection
    state.last_action = np.concatenate([d.ctrl[:4], arm_cmd])

    # ── Logging ───────────────────────────────────────────────────────────
    if data.time - state.last_print > 1.0:
        state.last_print = data.time
        car_p = _car_pos(data)
        ee_p  = _ee_pos(data)
        tgt_p = _target_pos(data)
        traj  = " [traj]" if (state._traj_coeffs is not None and
                              data.time - state._traj_t0 < state._traj_T) else ""
        print(f"[{data.time:5.2f}s] {FSM_NAMES[fsm]:<12} "
              f"car_x={car_p[0]:+.2f}  ee=({ee_p[0]:+.2f},{ee_p[1]:+.2f},{ee_p[2]:+.2f})  "
              f"box_z={tgt_p[2]:+.2f}{traj}")

    # ── Data collection ───────────────────────────────────────────────────
    collector.maybe_record(data.time, fsm, _ee_pos(data), _target_pos(data),
                           state.last_action)

    # ── FSM transition ────────────────────────────────────────────────────
    nxt = _check_transitions(state)
    if nxt != fsm:
        print(f"\n>>> [{data.time:5.2f}s] {FSM_NAMES[fsm]} → {FSM_NAMES[nxt]}\n")
        if nxt == FSM_GRASP:
            _grasp(True)
        if nxt == FSM_RELEASE:
            _grasp(False)
        state.fsm = nxt
        state.fsm_enter = data.time
        # Compute new IK target for this phase (expensive — once per phase)
        _compute_arm_target(nxt, state)
        # Start cubic trajectory from current qpos toward new target  (No.5)
        cur_q = _arm_qpos(data)
        state._traj_coeffs = np.array([_cubic_coeffs(cur_q[j],
                                       state._cached_target[j], TRAJ_DURATION)
                                       for j in range(6)])
        state._traj_t0 = data.time
        state._traj_T  = TRAJ_DURATION
        # Log dynamics once per phase change  (No.4)
        _log_mass_matrix_diag()

# ═══════════════════════════════════════════════════════════════════════════════
# GLFW callbacks  (No.2: keyboard + mouse interaction)
# ═══════════════════════════════════════════════════════════════════════════════
_button_left = _button_right = _button_middle = False
_last_x = _last_y = 0

def _key_cb(window, key, scancode, act, mods):
    if act != glfw.PRESS:
        return
    if key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        _grasp(False)          # release any active weld
        global state, collector
        state = SimState()
        collector.reset()
        print("[SIM] Reset — restarting task")
    elif key == glfw.KEY_S:
        collector.save()
    elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
        glfw.set_window_should_close(window, True)

def _mouse_button_cb(window, button, act, mods):
    global _button_left, _button_right, _button_middle
    _button_left   = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    _button_right  = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    _button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    glfw.get_cursor_pos(window)

def _mouse_move_cb(window, xpos, ypos):
    global _last_x, _last_y
    if _last_x == 0 and _last_y == 0:
        _last_x, _last_y = xpos, ypos
    dx = xpos - _last_x; dy = ypos - _last_y
    _last_x, _last_y = xpos, ypos
    if not (_button_left or _button_right or _button_middle):
        return
    w, h = glfw.get_window_size(window)
    shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
             glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    if _button_right:
        act = mj.mjtMouse.mjMOUSE_MOVE_H if shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif _button_left:
        act = mj.mjtMouse.mjMOUSE_ROTATE_H if shift else mj.mjMOUSE_ROTATE_V
    else:
        act = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, act, dx / h, dy / h, _main_scene, _main_cam_handle)

def _scroll_cb(window, xoff, yoff):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0,
                      -0.05 * yoff, _main_scene, _main_cam_handle)

# ═══════════════════════════════════════════════════════════════════════════════
# No.12 : post-simulation matplotlib summary
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_summary():
    if not HAVE_MPL or collector.frame_count < 2:
        print("[PLOT] matplotlib unavailable or no data — skipping plots")
        return

    t  = np.array(collector.timestamps)
    ee = np.array(collector.ee_positions)
    js = np.array(collector.joint_states)
    tg = np.array(collector.target_positions)
    cp = np.array([s[:3] for s in js])   # car xyz from joint_states
    fsm = np.array(collector.fsm_states)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Mobile Manipulator — Task Summary", fontsize=14, fontweight="bold")

    # (1) EE trajectory XZ, color-coded by FSM
    ax = axes[0, 0]
    for s in np.unique(fsm):
        msk = fsm == s
        ax.plot(ee[msk, 0], ee[msk, 2], '.', label=FSM_NAMES[s], markersize=4)
    ax.plot(TARGET_POS[0], TARGET_POS[2], 'r*', ms=12, label="Target")
    ax.plot(PLACE_POS[0], PLACE_POS[2], 'g*', ms=12, label="Place")
    ax.set_xlabel("EE X (m)"); ax.set_ylabel("EE Z (m)")
    ax.set_title("End-Effector Trajectory (XZ)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    # (2) Joint angles
    ax = axes[0, 1]
    names = ["pan", "lift", "elbow", "wrist", "finger_l", "finger_r"]
    for j in range(6):
        ax.plot(t, js[:, 7+j], label=names[j])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
    ax.set_title("Arm Joint Angles"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (3) Box Z
    ax = axes[1, 0]
    ax.plot(t, tg[:, 2], 'b-', lw=2)
    ax.axhline(TARGET_POS[2], color='gray', ls='--', alpha=0.5, label="Target Z")
    ax.axhline(PLACE_POS[2], color='green', ls='--', alpha=0.5, label="Place Z")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Box Z (m)")
    ax.set_title("Target Box Height"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (4) Car X
    ax = axes[1, 1]
    ax.plot(t, cp[:, 0], 'b-', lw=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Car X (m)")
    ax.set_title("Car Forward Position"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"summary_{_time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[No.12 Plot] Summary saved → {out_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Main  (No.1: basic loop   No.2: GLFW rendering)
# ═══════════════════════════════════════════════════════════════════════════════
def main(headless=False):
    global model, data, state, collector
    global _main_cam_handle, _main_opt, _main_scene, _main_context
    global _fk_data

    # ── Load model (No.1) ─────────────────────────────────────────────────
    model = mj.MjModel.from_xml_path(XML_PATH)
    data  = mj.MjData(model)
    _cache_ids(model)

    # Separate data for FK queries  (No.12: auxiliary MjData)
    _fk_data_obj = mj.MjData(model)
    globals()["_fk_data"] = _fk_data_obj

    state = SimState()
    collector = DataCollector()

    # ── Headless mode: run N episodes, no window ──────────────────────────
    if headless:
        print("=" * 60)
        print(f"Headless data-collection mode  ×{NUM_EPISODES} episodes")
        print(f"Target: {TARGET_POS}  Place: {PLACE_POS}")
        print("=" * 60)
        mj.set_mjcb_control(controller)
        for ep in range(NUM_EPISODES):
            print(f"\n── Episode {ep+1}/{NUM_EPISODES} ──")
            simend = data.time + 60.0
            while data.time < simend and state.fsm != FSM_DONE:
                mj.mj_step(model, data)
            print(f"Episode {ep+1} done. t={data.time:.2f}s  state={FSM_NAMES[state.fsm]}")
            if ep < NUM_EPISODES - 1:
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)
                _grasp(False)
                state = SimState()
        print(f"\nDone. {NUM_EPISODES} episodes, {collector.frame_count} frames total")
        collector.save()
        _plot_summary()
        return

    # ── GLFW window (No.2) ────────────────────────────────────────────────
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(1200, 900, "Mobile Manipulator — Demo + Collect", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    glfw.set_key_callback(window, _key_cb)
    glfw.set_mouse_button_callback(window, _mouse_button_cb)
    glfw.set_cursor_pos_callback(window, _mouse_move_cb)
    glfw.set_scroll_callback(window, _scroll_cb)

    # Render state  (No.2)
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scn = mj.MjvScene(model, maxgeom=10000)
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    cam.azimuth = 130; cam.elevation = -22; cam.distance = 3.5
    cam.lookat = np.array([0.7, 0.0, 0.5])
    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1
    # opt.flags[mj.mjtVisFlag.mjVIS_HEADLIGHT] = 0   # (not available in this MuJoCo version)

    _main_cam_handle = cam
    _main_opt        = opt
    _main_scene      = scn
    _main_context    = ctx

    mj.set_mjcb_control(controller)

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 64)
    print("Mobile Manipulator — All-in-One Demo (No.1–No.13) + Data Collection")
    print("=" * 64)
    print(f"  IK:     Jacobian damped pseudo-inverse (4-DOF)")
    print(f"  Traj:   cubic polynomial, {TRAJ_DURATION:.1f}s per waypoint")
    print(f"  Grasp:  equality-constraint weld")
    print(f"  Data:   {COLLECT_HZ} Hz → {OUT_DIR}/ep_*.npz")
    print(f"  Episodes: {NUM_EPISODES} (auto-repeat)")
    print(f"  Mouse:  drag=rotate  right-drag=pan  scroll=zoom")
    print(f"  Keys:   Backspace=reset  S=save  Q=quit")
    print("=" * 64)
    _log_mass_matrix_diag()

    # ── Render loop (multi-episode) ───────────────────────────────────────
    ep_count = 0
    frame_no = 0
    simend = data.time + 60.0
    try:
        while not glfw.window_should_close(window) and ep_count < NUM_EPISODES:
            t_start = data.time
            while data.time - t_start < 1.0 / 60.0:
                mj.mj_step(model, data)

            if data.time >= simend or state.fsm == FSM_DONE:
                ep_count += 1
                print(f"\n── Episode {ep_count}/{NUM_EPISODES} done ({collector.frame_count} frames) ──")
                if ep_count >= NUM_EPISODES:
                    break
                # Reset for next episode  (global state already declared at top of main)
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)
                _grasp(False)
                state = SimState()
                simend = data.time + 60.0
                print(f"── Episode {ep_count+1}/{NUM_EPISODES} starting... ──\n")
                continue

            # Camera tracks car ↔ target midpoint  (No.13: camera tracking)
            car_p = _car_pos(data)
            cam.lookat[0] = 0.5 * (car_p[0] + TARGET_POS[0])
            cam.lookat[1] = car_p[1]
            cam.lookat[2] = max(0.4, car_p[2] + 0.3)
            cam.distance = 3.0 + abs(car_p[0]) * 0.3

            vp = mj.MjrRect(0, 0, 1200, 900)
            mj.mjv_updateScene(model, data, opt, None, cam,
                               mj.mjtCatBit.mjCAT_ALL.value, scn)
            mj.mjr_render(vp, scn, ctx)
            glfw.swap_buffers(window)
            glfw.poll_events()
            frame_no += 1
    except Exception as e:
        print(f"\n[WARN] Render loop error: {e}")
    finally:
        glfw.terminate()

    # ── Cleanup ───────────────────────────────────────────────────────────
    _grasp(False)

    print("\n" + "=" * 64)
    print(f"Task ended  episodes={ep_count}  t={data.time:.2f}s  state={FSM_NAMES[state.fsm]}")
    print(f"Frames collected: {collector.frame_count}")
    print("=" * 64)

    collector.save()
    _plot_summary()

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    headless = "--headless" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--episodes" and i + 1 < len(sys.argv):
            NUM_EPISODES = int(sys.argv[i+1])
    main(headless=headless)
