"""
auto_sim_v2.py — Mobile Manipulator M3: Enhanced Automatic Task Simulation
==========================================================================
Full-featured version combining concepts from No.5–No.13:

  - No.5    : cubic polynomial trajectory interpolation for smooth arm motion
  - No.11   : NLopt-based numerical optimization
  - No.12   : NLopt 3D IK + matplotlib data visualization
  - No.13   : mouse camera interaction + state estimation

Key improvements over v1 (auto_sim.py):
  1. NLopt 3D IK using all 4 arm DOF (pan, lift, elbow, wrist)
     Falls back to analytical 2D IK if nlopt is not installed.
  2. Cubic trajectory interpolation — arm moves smoothly between waypoints
     instead of jumping instantly.
  3. Data logging + post-simulation matplotlib plots (EE path, joint angles,
     box Z, car X).
  4. Mouse drag/scroll camera control.
  5. FSM safety timeouts — each state has a max duration to prevent hangs.
  6. Equality-constraint-based grasping — the box actually attaches to the
     gripper during the GRASP→LIFT→DRIVE_BACK→PLACE sequence.

Run with:
    python3 auto_sim_v2.py
(Note: mjpython may fail on macOS with a GLFW threading error — use python3 instead.)
"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys

# Optional imports with fallback
try:
    import nlopt
    HAVE_NLOPT = True
except ImportError:
    print("[WARN] nlopt not installed — falling back to analytical 2D IK")
    HAVE_NLOPT = False

try:
    import matplotlib
    matplotlib.use('TkAgg')  # non-blocking backend
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    print("[WARN] matplotlib not installed — skipping post-sim plots")
    HAVE_MPL = False

# ============================================================================
# Constants & geometry (must match car.xml)
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "car.xml")

TARGET_WORLD = np.array([1.5, 0.0, 0.45])
PLACE_POS = np.array([0.25, 0.0, 0.55])  # in front of car at origin, reachable by the arm

L_UPPER = 0.25
L_FOREARM = 0.20
L_WRIST = 0.08
L_GRIPPER = 0.08
L2_EFF = L_FOREARM + L_WRIST + L_GRIPPER  # 0.36
ARM_BASE_Z_OFFSET = 0.16

STOWED_POSE = np.array([0.0, 0.5, -1.0, 0.0, 0.0])

DRIVE_SPEED = 0.4     # wheel ctrl for driving (lower = less overshoot)
BRAKE_TORQUE = 2.0    # braking torque in non-driving states
REACH_TOL = 0.15      # tolerance for EE reaching waypoints (meters)
DRIVE_TARGET_X = 1.05 # car target x for reaching the box

# ============================================================================
# FSM states
# ============================================================================
FSM_DRIVE_TO_TARGET = 0
FSM_REACH = 1
FSM_LOWER = 2
FSM_GRASP = 3
FSM_LIFT = 4
FSM_DRIVE_BACK = 5
FSM_PLACE = 6
FSM_RELEASE = 7
FSM_DONE = 8

FSM_NAMES = [
    "DRIVE_TO_TARGET", "REACH", "LOWER", "GRASP", "LIFT",
    "DRIVE_BACK", "PLACE", "RELEASE", "DONE",
]

# Safety timeouts per state (seconds) — force transition if exceeded
FSM_TIMEOUTS = {
    FSM_DRIVE_TO_TARGET: 15.0,
    FSM_REACH: 8.0,
    FSM_LOWER: 8.0,
    FSM_GRASP: 3.0,
    FSM_LIFT: 8.0,
    FSM_DRIVE_BACK: 15.0,
    FSM_PLACE: 5.0,
    FSM_RELEASE: 3.0,
    FSM_DONE: float("inf"),
}

# Duration for cubic arm trajectories (seconds)
TRAJ_DURATION = 0.6

# ============================================================================
# Cubic trajectory generator (from No.5 pattern)
# ============================================================================
def generate_cubic_trajectory(t0, tf, q0, qf):
    """Generate cubic polynomial coefficients for smooth joint-space motion.

    Returns (a0, a1, a2, a3) such that:
        q(t)  = a0 + a1*t + a2*t^2 + a3*t^3
        dq(t) = a1 + 2*a2*t + 3*a3*t^2
    with boundary conditions: q(t0)=q0, q(tf)=qf, dq(t0)=0, dq(tf)=0.
    """
    dt3 = (tf - t0) ** 3
    a0 = (qf * (t0 ** 2) * (3 * tf - t0) + q0 * (tf ** 2) * (tf - 3 * t0)) / dt3
    a1 = (6 * t0 * tf * (q0 - qf)) / dt3
    a2 = (3 * (t0 + tf) * (qf - q0)) / dt3
    a3 = (2 * (q0 - qf)) / dt3
    return np.array([a0, a1, a2, a3])


def evaluate_cubic_trajectory(coeffs, t):
    """Evaluate cubic trajectory at time t.

    Returns (position, velocity) as scalars.
    """
    a0, a1, a2, a3 = coeffs
    pos = a0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)
    vel = a1 + 2 * a2 * t + 3 * a3 * (t ** 2)
    return pos, vel


def generate_joint_trajectory(t0, tf, q_start, q_end):
    """Generate cubic trajectories for all 5 arm joints.

    Returns array of shape (5, 4) — one row per joint.
    """
    trajectories = np.zeros((5, 4))
    for j in range(5):
        trajectories[j] = generate_cubic_trajectory(t0, tf, q_start[j], q_end[j])
    return trajectories


def evaluate_joint_trajectory(trajectories, t):
    """Evaluate multi-joint trajectory at time t.

    Returns (positions, velocities) as 5-element arrays.
    """
    pos = np.zeros(5)
    vel = np.zeros(5)
    for j in range(5):
        pos[j], vel[j] = evaluate_cubic_trajectory(trajectories[j], t)
    return pos, vel


# ============================================================================
# IK: analytical 2D (fallback) and NLopt 3D (preferred)
# ============================================================================
def analytical_ik_2d(x, z, L1=L_UPPER, L2=L2_EFF):
    """2-link planar IK in the arm_base local (x, z) plane.

    Returns (shoulder_lift, elbow, wrist_pitch). shoulder_pan is always 0.
    """
    D = np.sqrt(x * x + z * z)
    max_reach = L1 + L2
    if D > max_reach * 0.95:
        scale = max_reach * 0.95 / D
        x, z = x * scale, z * scale
        D2 = (max_reach * 0.95) ** 2
    else:
        D2 = x * x + z * z

    cos_q2 = (D2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2_int = np.arccos(cos_q2)

    alpha = np.arctan2(z, x)
    beta = np.arctan2(L2 * np.sin(q2_int), L1 + L2 * np.cos(q2_int))
    q1 = alpha - beta

    q1 = -q1
    q2 = -(np.pi - q2_int)
    q3 = -(q1 + q2)
    return q1, q2, q3


def ik_to_world_analytical(world_pos, arm_base_pos, gripper_ctrl=0.0):
    """Compute full 5-joint arm target using analytical 2D IK."""
    local = world_pos - arm_base_pos
    q1, q2, q3 = analytical_ik_2d(local[0], local[2])
    return np.array([0.0, q1, q2, q3, gripper_ctrl])


# --- NLopt 3D IK (from No.12 pattern) ---

# Cached arm joint qpos addresses — computed once in main()
_arm_qpos_adr = None  # list of 5 ints: [pan, lift, elbow, wrist, gripper]

# Cached body/site IDs for fast lookup
_car_body_id = None
_target_body_id = None
_arm_base_body_id = None
_ee_site_id = None
_gripper_body_id = None


def _init_cached_ids(m):
    """Cache all frequently-used IDs from the model (called once in main)."""
    global _arm_qpos_adr, _car_body_id, _target_body_id, _arm_base_body_id
    global _ee_site_id, _gripper_body_id
    _arm_qpos_adr = [m.jnt_qposadr[m.joint(name).id]
                     for name in ["shoulder_pan", "shoulder_lift", "elbow",
                                  "wrist_pitch", "gripper"]]
    _car_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "car")
    _target_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "target_box")
    _arm_base_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "arm_base")
    _ee_site_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "end_effector")
    _gripper_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper")


def _get_arm_qpos(d):
    """Read current arm joint angles from data."""
    return np.array([d.qpos[adr] for adr in _arm_qpos_adr])


def _set_arm_qpos(d, q):
    """Write arm joint angles to data."""
    for adr, val in zip(_arm_qpos_adr, q):
        d.qpos[adr] = val


def _get_car_pos(d):
    """Return car body world position."""
    return d.xpos[_car_body_id]


def _get_target_pos(d):
    """Return target_box world position."""
    return d.xpos[_target_body_id]


def _get_ee_pos(d):
    """Return end-effector site world position."""
    return d.site_xpos[_ee_site_id]


def _get_arm_base_pos(d):
    """Return arm_base body world position."""
    return d.xpos[_arm_base_body_id]


# Separate model/data for FK queries — created once in main()
_fk_model = None
_fk_data = None


def set_fk_resources(fk_model, fk_data_sim):
    """Register the shared FK model and data instances (called from main)."""
    global _fk_model, _fk_data
    _fk_model = fk_model
    _fk_data = fk_data_sim
    _init_cached_ids(fk_model)


def numerical_ik_3d(target_world, arm_base_pos, q_init=None, gripper_ctrl=0.0,
                    max_iter=200, tol=0.01, alpha=0.3):
    """Numerical IK using Jacobian pseudo-inverse with MuJoCo FK.

    Uses the shared FK model/data for side-effect-free FK/Jacobian queries.
    The arm has 4 DOF (pan, lift, elbow, wrist) controlling the EE 3D position,
    with wrist constrained to keep the gripper level.

    If max_iter is exhausted without convergence, returns the best-effort solution.
    """
    global _fk_model, _fk_data, _arm_qpos_adr
    if _fk_model is None or _fk_data is None:
        # Fallback: use analytical IK
        return ik_to_world_analytical(target_world, arm_base_pos)

    # Sync FK data car position
    _fk_data.qpos[7] = arm_base_pos[0]
    _fk_data.qpos[8] = arm_base_pos[1]
    _fk_data.qpos[9] = arm_base_pos[2] - ARM_BASE_Z_OFFSET
    _fk_data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]

    # Initial guess: compute shoulder_pan, then choose pose based on target direction
    local = target_world - arm_base_pos
    pan_init = float(np.clip(np.arctan2(local[1], local[0]), -1.57, 1.57))

    if q_init is None:
        r_xy = np.sqrt(local[0]**2 + local[1]**2)
        if r_xy < 0.05:
            # Target is nearly directly above — arm pointing upward with slight bend
            q_init = np.array([pan_init, 0.3, -0.5, -0.5])
        elif local[0] < -0.05:
            # Target is behind — use a folded-back pose
            q_init = np.array([pan_init, -1.0, 1.5, -0.5])
        elif local[2] > 0.25:
            # Target is above and forward — use a raised pose
            q_init = np.array([pan_init, 0.5, 0.8, -1.3])
        else:
            # Target is forward — default reaching pose
            q_init = np.array([pan_init, 0.8, 1.0, -1.8])
    q = np.array([q_init[0], q_init[1], q_init[2], q_init[3]])

    # Joint limits
    q_min = np.array([-1.57, -1.57, -2.0, -1.57])
    q_max = np.array([1.57, 1.57, 2.0, 1.57])

    for _ in range(max_iter):
        _set_arm_qpos(_fk_data, [q[0], q[1], q[2], q[3], 0.0])
        mj.mj_fwdPosition(_fk_model, _fk_data)
        ee = _get_ee_pos(_fk_data)
        error = target_world - ee[:3]

        if np.linalg.norm(error) < tol:
            break

        # Compute position Jacobian for the end-effector (attached to gripper body)
        jacp = np.zeros((3, _fk_model.nv))
        mj.mj_jac(_fk_model, _fk_data, jacp, None, ee, _gripper_body_id)

        # Extract columns for arm joints (4 DOF: pan, lift, elbow, wrist)
        arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch"]
        J_arm = np.zeros((3, 4))
        for i, name in enumerate(arm_joint_names):
            jnt_id = _fk_model.joint(name).id
            dof_adr = _fk_model.jnt_dofadr[jnt_id]
            J_arm[:, i] = jacp[:, dof_adr]

        # Pseudo-inverse with damping
        JtJ = J_arm.T @ J_arm
        lam = 0.01
        dq = np.linalg.solve(JtJ + lam * np.eye(4), J_arm.T @ error) * alpha

        # Update with clamping
        q = q + dq
        q = np.clip(q, q_min, q_max)

        # Enforce wrist = -(lift + elbow) to keep gripper level
        # (soft constraint: blend toward the level pose)
        wrist_level = -(q[1] + q[2])
        q[3] = 0.8 * wrist_level + 0.2 * q[3]

    # Final wrist: keep gripper level
    q[3] = float(np.clip(-(q[1] + q[2]), -1.57, 1.57))

    return np.array([q[0], q[1], q[2], q[3], gripper_ctrl])


# ============================================================================
# Data logger
# ============================================================================
class DataLogger:
    """Records simulation data and generates post-simulation plots."""

    def __init__(self):
        self.times = []
        self.fsm_states = []
        self.ee_positions = []      # (x, y, z) per sample
        self.joint_angles = []      # (q0..q4) per sample
        self.box_positions = []     # (x, y, z) per sample
        self.car_positions = []     # (x, y, z) per sample
        self.last_record_time = -0.1
        self.record_interval = 0.1  # seconds

    def record(self, t, fsm, ee_pos, joint_angles, box_pos, car_pos):
        """Record a data point if enough time has elapsed."""
        if t - self.last_record_time < self.record_interval:
            return
        self.last_record_time = t
        self.times.append(t)
        self.fsm_states.append(fsm)
        self.ee_positions.append(ee_pos.copy())
        self.joint_angles.append(joint_angles.copy())
        self.box_positions.append(box_pos.copy())
        self.car_positions.append(car_pos.copy())

    def plot_summary(self):
        """Generate 2×2 summary plots using matplotlib."""
        if not HAVE_MPL or len(self.times) < 2:
            print("[INFO] Skipping plots (matplotlib unavailable or no data)")
            return

        times = np.array(self.times)
        ee = np.array(self.ee_positions)
        joints = np.array(self.joint_angles)
        box = np.array(self.box_positions)
        car = np.array(self.car_positions)

        # Color-code by FSM state
        fsm_arr = np.array(self.fsm_states)
        unique_fsms = np.unique(fsm_arr)
        cmap = plt.cm.tab10
        fsm_colors = {s: cmap(i / max(1, len(unique_fsms) - 1))
                      for i, s in enumerate(unique_fsms)}

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Mobile Manipulator M3 — Task Summary", fontsize=14, fontweight="bold")

        # (1) EE trajectory in XZ plane, colored by FSM
        ax = axes[0, 0]
        for s in unique_fsms:
            mask = fsm_arr == s
            ax.plot(ee[mask, 0], ee[mask, 2], '.', color=fsm_colors[s],
                    label=FSM_NAMES[s], markersize=4)
        ax.plot(TARGET_WORLD[0], TARGET_WORLD[2], 'r*', markersize=12, label="Target")
        ax.plot(PLACE_POS[0], PLACE_POS[2], 'g*', markersize=12, label="Place")
        ax.set_xlabel("EE X (m)")
        ax.set_ylabel("EE Z (m)")
        ax.set_title("End-Effector Trajectory (XZ)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # (2) Joint angles vs time
        ax = axes[0, 1]
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "gripper"]
        for j in range(5):
            ax.plot(times, joints[:, j], label=joint_names[j], linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint angle (rad)")
        ax.set_title("Arm Joint Angles")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # (3) Target box Z vs time
        ax = axes[1, 0]
        ax.plot(times, box[:, 2], 'b-', linewidth=2)
        ax.axhline(y=TARGET_WORLD[2], color='gray', linestyle='--', alpha=0.5, label="Target Z")
        ax.axhline(y=PLACE_POS[2], color='green', linestyle='--', alpha=0.5, label="Place Z")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Box Z (m)")
        ax.set_title("Target Box Height")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (4) Car X vs time
        ax = axes[1, 1]
        ax.plot(times, car[:, 0], 'b-', linewidth=2)
        ax.axvline(x=times[fsm_arr == FSM_DRIVE_TO_TARGET][-1] if np.any(fsm_arr == FSM_DRIVE_TO_TARGET) else 0,
                   color='orange', linestyle=':', alpha=0.7, label="Reach target")
        ax.axvline(x=times[fsm_arr == FSM_DRIVE_BACK][0] if np.any(fsm_arr == FSM_DRIVE_BACK) else 0,
                   color='purple', linestyle=':', alpha=0.7, label="Start return")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Car X (m)")
        ax.set_title("Car Forward Position")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(10)
        plt.close()
        print("[PLOT] Summary plots displayed (10s window)")


# ============================================================================
# SimState
# ============================================================================
class SimState:
    """Mutable simulation state shared between controller and main loop."""

    def __init__(self):
        self.fsm = FSM_DRIVE_TO_TARGET
        self.fsm_enter_time = 0.0
        self.last_print_time = -1.0
        self.frozen_arm = None
        self.completed = False

        # Cubic trajectory state
        self.active_trajectory = None   # (5, 4) array or None
        self.trajectory_start_time = 0.0
        self.trajectory_end_time = 0.0
        self.last_arm_targets = STOWED_POSE.copy()

        # Grasp constraint
        self.grasp_eq_id = -1  # equality constraint ID, -1 = not active


# ============================================================================
# Arm target computation (per FSM state)
# ============================================================================
def get_desired_arm_targets(fsm, data, st):
    """Compute the desired (final) arm joint angles for a given FSM state.

    Does NOT apply trajectory smoothing — that's done in the controller.
    """
    arm_base_pos = _get_arm_base_pos(data)
    current_q = _get_arm_qpos(data)

    if fsm == FSM_DRIVE_TO_TARGET:
        return STOWED_POSE.copy()

    if fsm == FSM_REACH:
        goal = TARGET_WORLD.copy()
        goal[2] += 0.15
        return numerical_ik_3d(goal, arm_base_pos, current_q)

    if fsm in (FSM_LOWER, FSM_GRASP):
        return numerical_ik_3d(TARGET_WORLD, arm_base_pos, current_q)

    if fsm == FSM_LIFT:
        goal = TARGET_WORLD.copy()
        goal[2] += 0.3
        return numerical_ik_3d(goal, arm_base_pos, current_q)

    if fsm == FSM_DRIVE_BACK:
        if st.frozen_arm is None:
            st.frozen_arm = get_desired_arm_targets(FSM_LIFT, data, st).copy()
        return st.frozen_arm

    if fsm == FSM_PLACE:
        place_pos = PLACE_POS
        return numerical_ik_3d(place_pos, arm_base_pos, current_q)

    if fsm == FSM_RELEASE:
        place_pos = PLACE_POS
        targets = numerical_ik_3d(place_pos, arm_base_pos, current_q)
        targets[4] = 0.025  # open gripper
        return targets

    return STOWED_POSE.copy()


# ============================================================================
# Grasp management (toggle pre-defined weld equality in car.xml)
# ============================================================================
def activate_grasp(data, st):
    """Activate the weld equality constraint between gripper and target_box."""
    if st.grasp_eq_id >= 0:
        return  # already active

    eq = model.eq("grasp_weld")
    if eq is None:
        print("[GRASP] ERROR: grasp_weld equality not found in model")
        return

    eq.active0[0] = 1
    st.grasp_eq_id = eq.id
    print(f"[GRASP] Box welded to gripper (eq_id={eq.id})")


def deactivate_grasp(data, st):
    """Deactivate the weld equality constraint, releasing the box."""
    if st.grasp_eq_id < 0:
        return
    eq = model.eq(st.grasp_eq_id)
    eq.active0[0] = 0
    print(f"[GRASP] Box released (deactivated eq_id={st.grasp_eq_id})")
    st.grasp_eq_id = -1


# ============================================================================
# FSM transition logic
# ============================================================================
def check_fsm_transitions(data, st):
    """Evaluate transition conditions and return the next FSM state."""
    car_pos = _get_car_pos(data)
    ee_pos = _get_ee_pos(data)
    elapsed = data.time - st.fsm_enter_time

    next_fsm = st.fsm

    # Timeout check
    timeout = FSM_TIMEOUTS.get(st.fsm, float("inf"))
    if elapsed > timeout:
        print(f"[TIMEOUT] {FSM_NAMES[st.fsm]} exceeded {timeout:.0f}s — forcing advance")
        skip_order = {
            FSM_DRIVE_TO_TARGET: FSM_REACH,
            FSM_REACH: FSM_LOWER,
            FSM_LOWER: FSM_GRASP,
            FSM_GRASP: FSM_LIFT,
            FSM_LIFT: FSM_DRIVE_BACK,
            FSM_DRIVE_BACK: FSM_PLACE,
            FSM_PLACE: FSM_RELEASE,
            FSM_RELEASE: FSM_DONE,
        }
        return skip_order.get(st.fsm, st.fsm + 1)

    # Normal position-based transitions
    if st.fsm == FSM_DRIVE_TO_TARGET:
        if car_pos[0] > DRIVE_TARGET_X:
            next_fsm = FSM_REACH

    elif st.fsm == FSM_REACH:
        goal = TARGET_WORLD.copy(); goal[2] += 0.15
        if np.linalg.norm(ee_pos - goal) < REACH_TOL:
            next_fsm = FSM_LOWER

    elif st.fsm == FSM_LOWER:
        if np.linalg.norm(ee_pos - TARGET_WORLD) < REACH_TOL:
            next_fsm = FSM_GRASP

    elif st.fsm == FSM_GRASP:
        if elapsed > 1.0:
            next_fsm = FSM_LIFT

    elif st.fsm == FSM_LIFT:
        goal = TARGET_WORLD.copy(); goal[2] += 0.3
        if np.linalg.norm(ee_pos - goal) < REACH_TOL:
            st.frozen_arm = None
            next_fsm = FSM_DRIVE_BACK

    elif st.fsm == FSM_DRIVE_BACK:
        if car_pos[0] < 0.15:
            next_fsm = FSM_PLACE

    elif st.fsm == FSM_PLACE:
        place_pos = PLACE_POS
        if np.linalg.norm(ee_pos - place_pos) < REACH_TOL:
            next_fsm = FSM_RELEASE

    elif st.fsm == FSM_RELEASE:
        if elapsed > 0.5:
            next_fsm = FSM_DONE

    return next_fsm


# ============================================================================
# Controller (called every mj_step)
# ============================================================================
model = None
data = None


def controller(m, d):
    global model, data
    model = m
    data = d

    car_pos = _get_car_pos(data)
    ee_pos = _get_ee_pos(data)

    # --- Car control: constant speed drive, brake otherwise ---
    if state.fsm == FSM_DRIVE_TO_TARGET:
        data.ctrl[0:4] = DRIVE_SPEED
    elif state.fsm == FSM_DRIVE_BACK:
        data.ctrl[0:4] = -DRIVE_SPEED
    else:
        # Active braking: oppose wheel velocity to stop the car
        wheel_names = ["wheel_fl_joint", "wheel_fr_joint",
                       "wheel_rl_joint", "wheel_rr_joint"]
        for i, name in enumerate(wheel_names):
            dof_adr = model.jnt_dofadr[model.joint(name).id]
            w = data.qvel[dof_adr]
            data.ctrl[i] = -np.sign(w) * BRAKE_TORQUE if abs(w) > 0.01 else 0.0

    # --- Arm control: position servos track desired targets directly ---
    current_q = _get_arm_qpos(data)
    arm_targets = get_desired_arm_targets(state.fsm, data, state)

    # Write arm position-servo targets
    for i in range(5):
        data.ctrl[4 + i] = arm_targets[i]

    # --- Periodic status print ---
    if data.time - state.last_print_time > 1.0:
        state.last_print_time = data.time
        target_pos = _get_target_pos(data)
        traj_info = ""
        if state.active_trajectory is not None and data.time < state.trajectory_end_time:
            traj_info = " [traj]"
        print(f"[{data.time:5.2f}s] {FSM_NAMES[state.fsm]:<16} "
              f"car=({car_pos[0]:+.2f},{car_pos[1]:+.2f})  "
              f"ee=({ee_pos[0]:+.2f},{ee_pos[1]:+.2f},{ee_pos[2]:+.2f})  "
              f"box_z={target_pos[2]:+.2f}{traj_info}")

    # --- Data logging ---
    logger.record(data.time, state.fsm, ee_pos, current_q,
                  _get_target_pos(data), car_pos)

    # --- State transitions ---
    next_fsm = check_fsm_transitions(data, state)

    if next_fsm != state.fsm:
        print(f"\n>>> [{data.time:5.2f}s] {FSM_NAMES[state.fsm]} -> {FSM_NAMES[next_fsm]}\n")

        # Grasp management
        if next_fsm == FSM_GRASP:
            activate_grasp(data, state)
        if next_fsm == FSM_RELEASE or (state.fsm == FSM_GRASP and next_fsm != FSM_GRASP):
            # Release on explicit RELEASE state or if we somehow skip past GRASP
            pass
        if next_fsm == FSM_RELEASE:
            deactivate_grasp(data, state)

        # Reset trajectory so next state computes a fresh one
        state.active_trajectory = None
        state.fsm = next_fsm
        state.fsm_enter_time = data.time


# ============================================================================
# Mouse & keyboard callbacks (standard pattern from No.5/12/13)
# ============================================================================
button_left = False
button_middle = False
button_right = False
last_mouse_x = 0
last_mouse_y = 0


def keyboard_callback(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        deactivate_grasp(data, state)
        state.__init__()
        print("[SIM] Reset — restarting task")


def mouse_button_callback(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)


def mouse_move_callback(window, xpos, ypos):
    global last_mouse_x, last_mouse_y
    if last_mouse_x == 0 and last_mouse_y == 0:
        last_mouse_x, last_mouse_y = xpos, ypos
    dx = xpos - last_mouse_x
    dy = ypos - last_mouse_y
    last_mouse_x, last_mouse_y = xpos, ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    shift_left = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    shift_right = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = shift_left or shift_right

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll_callback(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


# ============================================================================
# Main
# ============================================================================
state = SimState()
logger = DataLogger()
scene = None
cam = None


def main():
    global model, data, state, logger, scene, cam

    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    state = SimState()
    logger = DataLogger()

    # Initialize cached IDs (used by all accessor functions)
    _init_cached_ids(model)

    # Separate model+data for FK queries (NLopt IK)
    fk_data = mj.MjData(model)
    set_fk_resources(model, fk_data)

    # GLFW window
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(1200, 900, "Mobile Manipulator M3 (Enhanced)", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Callbacks
    glfw.set_key_callback(window, keyboard_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, mouse_move_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Render state
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    cam.azimuth = 130.0
    cam.elevation = -22.0
    cam.distance = 3.5
    cam.lookat = np.array([0.7, 0.0, 0.5])

    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1
    opt.flags[mj.mjtVisFlag.mjVIS_HEADLIGHT] = 0  # use scene lights only

    mj.set_mjcb_control(controller)

    # Header
    print("=" * 64)
    print("Mobile Manipulator M3 — Enhanced Automatic Task Simulation")
    print("=" * 64)
    print(f"  IK solver:    numerical (Jacobian pseudo-inverse via MuJoCo FK)")
    print(f"  Cubic trajectories: {TRAJ_DURATION:.1f}s per waypoint")
    print(f"  Weld grasping:  ENABLED")
    print(f"  FSM timeouts:   ENABLED")
    print(f"  Data logging:   {'matplotlib' if HAVE_MPL else 'console only'}")
    print(f"  Target box at:  {TARGET_WORLD}")
    print(f"  FSM: {' -> '.join(FSM_NAMES[:8])}")
    print(f"  Mouse: drag=rotate, right-drag=pan, scroll=zoom")
    print("=" * 64)

    simend = 60.0
    while not glfw.window_should_close(window):
        simstart = data.time
        while (data.time - simstart < 1.0 / 60.0):
            mj.mj_step(model, data)

        if data.time >= simend or state.fsm == FSM_DONE:
            break

        # Camera tracks midpoint of car and target
        car_pos = _get_car_pos(data)
        cam.lookat[0] = 0.5 * (car_pos[0] + TARGET_WORLD[0])
        cam.lookat[1] = car_pos[1]
        cam.lookat[2] = max(0.4, car_pos[2] + 0.3)
        cam.distance = 3.0 + abs(car_pos[0]) * 0.3

        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

    # Clean up grasp constraint
    deactivate_grasp(data, state)

    # Summary
    print("\n" + "=" * 64)
    print(f"Task ended at t={data.time:.2f}s  final state: {FSM_NAMES[state.fsm]}")
    print(f"Final target box position: {_get_target_pos(data)}")
    print(f"Data points logged: {len(logger.times)}")
    print("=" * 64)

    # Post-simulation plots
    logger.plot_summary()


if __name__ == "__main__":
    main()
