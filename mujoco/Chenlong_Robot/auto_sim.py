"""
auto_sim.py — Mobile Manipulator M2: Automatic Task Simulation
================================================================
No-keyboard version: runs the full mobile-manipulation task automatically.

Task FSM (8 states):
  DRIVE_TO_TARGET  →  REACH  →  LOWER  →  GRASP  →  LIFT  →  DRIVE_BACK  →  PLACE  →  RELEASE  →  DONE

Concepts from No.1-13 exercised here:
  - No.1/2  : model + GLFW viewer
  - No.3    : position servos (PD on each arm joint)
  - No.5    : FSM with phase-based control + timeouts
  - No.6    : 2-link IK to compute arm joint angles for a target
  - No.7    : state-based control (linearize around current pose)
  - No.9    : different joint commands per phase
  - No.13   : position-based state estimation (data.xpos for arm_base, target)

Run with:
    mjpython auto_sim.py
"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# ============================================================================
# Setup
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "car.xml")

# Target box world position (matches car.xml)
TARGET_WORLD = np.array([1.5, 0.0, 0.45])

# Arm geometry (must match car.xml)
L_UPPER = 0.25      # upper_arm length
L_FOREARM = 0.20    # forearm length
L_WRIST = 0.08      # wrist length
L_GRIPPER = 0.08    # gripper offset
L2_EFF = L_FOREARM + L_WRIST + L_GRIPPER  # 0.36
ARM_BASE_Z_OFFSET = 0.16  # arm_base sits 0.16m above car center

# Stowed / home pose (joint angles, rad): [pan, lift, elbow, wrist, gripper]
STOWED_POSE = np.array([0.0, 0.5, -1.0, 0.0, 0.0])

# Drive gains
DRIVE_SPEED = 2.5   # wheel ctrl magnitude while driving
REACH_TOL = 0.08    # how close EE must be to "reach" a target

# ============================================================================
# FSM
# ============================================================================
FSM_DRIVE_TO_TARGET = 0
FSM_REACH           = 1
FSM_LOWER           = 2
FSM_GRASP           = 3
FSM_LIFT            = 4
FSM_DRIVE_BACK      = 5
FSM_PLACE           = 6
FSM_RELEASE         = 7
FSM_DONE            = 8

FSM_NAMES = [
    "DRIVE_TO_TARGET", "REACH", "LOWER", "GRASP", "LIFT",
    "DRIVE_BACK",      "PLACE", "RELEASE", "DONE",
]

# ============================================================================
# IK: 2-link planar IK in arm_base (x, z) plane
# Treats L1 = upper_arm, L2 = forearm + wrist + gripper
# Returns (shoulder_lift, elbow, wrist_pitch)
# ============================================================================
def arm_ik(x, z, L1=L_UPPER, L2=L2_EFF):
    """Solve for (q1, q2, q3) given target (x, z) in arm_base local frame.

    Convention (empirically determined for axis="0 1 0"):
      q1 = shoulder_lift: rotation about y from "straight up"
      q2 = elbow: negative when arm is bent forward
      q3 = wrist_pitch: chosen so gripper stays level (q3 = -(q1+q2))
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
    q2_int = np.arccos(cos_q2)             # interior angle: 0=straight, pi=folded

    alpha = np.arctan2(z, x)
    beta = np.arctan2(L2 * np.sin(q2_int), L1 + L2 * np.cos(q2_int))
    q1 = alpha - beta

    # In MuJoCo axis="0 1 0", positive rotation moves +x toward -z.
    # So when arm tilts forward (+x) we need NEGATIVE q1.
    q1 = -q1
    q2 = -(np.pi - q2_int)                  # bent = negative
    q3 = -(q1 + q2)                         # wrist to keep gripper level
    return q1, q2, q3


def ik_to_world(world_pos, arm_base_pos, gripper_ctrl=0.0):
    """Compute full 5-joint arm target for a world-space end-effector goal."""
    local = world_pos - arm_base_pos
    q1, q2, q3 = arm_ik(local[0], local[2])
    return np.array([0.0, q1, q2, q3, gripper_ctrl])


# ============================================================================
# State
# ============================================================================
class SimState:
    def __init__(self):
        self.fsm = FSM_DRIVE_TO_TARGET
        self.fsm_enter_time = 0.0
        self.last_print_time = -1.0
        self.frozen_arm = None  # arm pose frozen during DRIVE_BACK
        self.completed = False

state = SimState()
model = None
data = None


# ============================================================================
# Per-state arm-target computation
# ============================================================================
def get_arm_targets_for(fsm, data):
    """Return desired arm joint angles for the given FSM state."""
    arm_base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm_base")
    arm_base_pos = data.xpos[arm_base_id]

    if fsm in (FSM_DRIVE_TO_TARGET,):
        return STOWED_POSE.copy()

    if fsm == FSM_REACH:
        # Above target (15cm clearance)
        goal = TARGET_WORLD.copy()
        goal[2] += 0.15
        return ik_to_world(goal, arm_base_pos, gripper_ctrl=0.0)

    if fsm in (FSM_LOWER, FSM_GRASP):
        # At target level
        return ik_to_world(TARGET_WORLD, arm_base_pos, gripper_ctrl=-0.025)

    if fsm == FSM_LIFT:
        # Lifted 30cm above target
        goal = TARGET_WORLD.copy()
        goal[2] += 0.3
        return ik_to_world(goal, arm_base_pos, gripper_ctrl=-0.025)

    if fsm == FSM_DRIVE_BACK:
        # Freeze the LIFT pose so the box doesn't swing
        if state.frozen_arm is None:
            state.frozen_arm = get_arm_targets_for(FSM_LIFT, data).copy()
        return state.frozen_arm

    if fsm in (FSM_PLACE,):
        # Place at world origin, box height
        place_pos = np.array([0.0, 0.0, 0.45])
        return ik_to_world(place_pos, arm_base_pos, gripper_ctrl=-0.025)

    if fsm == FSM_RELEASE:
        place_pos = np.array([0.0, 0.0, 0.45])
        return ik_to_world(place_pos, arm_base_pos, gripper_ctrl=0.0)

    return STOWED_POSE.copy()


# ============================================================================
# Controller (called every mj_step)
# ============================================================================
def controller(model, data):
    global state

    car_pos = data.qpos[0:3]
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "end_effector")
    ee_pos = data.site_xpos[ee_id]

    # --- Car control: drive in driving states, park otherwise ---
    if state.fsm == FSM_DRIVE_TO_TARGET:
        data.ctrl[0:4] = DRIVE_SPEED
    elif state.fsm == FSM_DRIVE_BACK:
        data.ctrl[0:4] = -DRIVE_SPEED
    else:
        data.ctrl[0:4] = 0.0

    # --- Arm control: 5 position servos ---
    arm_targets = get_arm_targets_for(state.fsm, data)
    for i in range(5):
        data.ctrl[4 + i] = arm_targets[i]

    # --- Periodic status print ---
    if data.time - state.last_print_time > 1.0:
        state.last_print_time = data.time
        target_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "target_box")
        target_pos = data.xpos[target_id]
        print(f"[{data.time:5.2f}s] {FSM_NAMES[state.fsm]:<16} "
              f"car=({car_pos[0]:+.2f},{car_pos[1]:+.2f})  "
              f"ee=({ee_pos[0]:+.2f},{ee_pos[1]:+.2f},{ee_pos[2]:+.2f})  "
              f"box_z={target_pos[2]:+.2f}")

    # --- State transitions ---
    next_fsm = state.fsm

    if state.fsm == FSM_DRIVE_TO_TARGET:
        if car_pos[0] > 1.0:
            next_fsm = FSM_REACH
    elif state.fsm == FSM_REACH:
        goal = TARGET_WORLD.copy(); goal[2] += 0.15
        if np.linalg.norm(ee_pos - goal) < REACH_TOL:
            next_fsm = FSM_LOWER
    elif state.fsm == FSM_LOWER:
        if np.linalg.norm(ee_pos - TARGET_WORLD) < REACH_TOL:
            next_fsm = FSM_GRASP
    elif state.fsm == FSM_GRASP:
        # Wait 1 second for gripper to close
        if data.time - state.fsm_enter_time > 1.0:
            next_fsm = FSM_LIFT
    elif state.fsm == FSM_LIFT:
        goal = TARGET_WORLD.copy(); goal[2] += 0.3
        if np.linalg.norm(ee_pos - goal) < REACH_TOL:
            state.frozen_arm = None  # reset before next freeze
            next_fsm = FSM_DRIVE_BACK
    elif state.fsm == FSM_DRIVE_BACK:
        if car_pos[0] < 0.1:
            next_fsm = FSM_PLACE
    elif state.fsm == FSM_PLACE:
        place_pos = np.array([0.0, 0.0, 0.45])
        if np.linalg.norm(ee_pos - place_pos) < REACH_TOL:
            next_fsm = FSM_RELEASE
    elif state.fsm == FSM_RELEASE:
        if data.time - state.fsm_enter_time > 0.5:
            next_fsm = FSM_DONE

    if next_fsm != state.fsm:
        print(f"\n>>> [{data.time:5.2f}s] {FSM_NAMES[state.fsm]} -> {FSM_NAMES[next_fsm]}\n")
        state.fsm = next_fsm
        state.fsm_enter_time = data.time


# ============================================================================
# Main
# ============================================================================
def main():
    global model, data, state

    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    state = SimState()

    # GLFW window
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(1200, 900, "Mobile Manipulator (M2 Auto)", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

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

    # Show joint frames (debug aid for IK)
    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

    mj.set_mjcb_control(controller)

    # Header
    print("=" * 64)
    print("Mobile Manipulator (M2) — Automatic Task Simulation")
    print("=" * 64)
    print(f"Target box at world: {TARGET_WORLD}")
    print(f"Arm: 4-DOF (shoulder_pan/lift, elbow, wrist_pitch) + gripper slide")
    print(f"FSM: {' -> '.join(FSM_NAMES[:8])}")
    print("=" * 64)

    simend = 60.0
    while not glfw.window_should_close(window):
        simstart = data.time
        while (data.time - simstart < 1.0 / 60.0):
            mj.mj_step(model, data)

        if data.time >= simend or state.fsm == FSM_DONE:
            break

        # Camera tracks midpoint of car and target
        cam.lookat[0] = 0.5 * (data.qpos[0] + TARGET_WORLD[0])
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = max(0.4, data.qpos[2] + 0.3)
        # Zoom out as the scene grows
        cam.distance = 3.0 + abs(data.qpos[0]) * 0.3

        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

    # Summary
    print("\n" + "=" * 64)
    print(f"Task ended at t={data.time:.2f}s  final state: {FSM_NAMES[state.fsm]}")
    target_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "target_box")
    print(f"Final target box position: {data.xpos[target_id]}")
    print("=" * 64)


if __name__ == "__main__":
    main()
