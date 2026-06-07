"""
Mobile Manipulator Walker (M1)
==============================
Car with 4-DOF arm + gripper. WASD drives the car (differential), 1-5 select
arm joints, Up/Down adjust the selected joint, G toggles the gripper, 0 homes
the arm, Backspace resets the simulation.

Run with:
    mjpython walker.py
"""
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# ---------- Paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "car.xml")

# ---------- Ctrl channel mapping (MUST match XML actuator order) ----------
# ctrl[0..3] : 4 wheels (motor)
# ctrl[4..8] : 5 arm position servos
ARM_CTRL_START = 4
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "gripper"]
JOINT_LIMITS = np.array([
    (-1.57,  1.57),  # shoulder_pan
    (-1.57,  1.57),  # shoulder_lift
    (-2.00,  2.00),  # elbow
    (-1.57,  1.57),  # wrist_pitch
    (-0.03,  0.03),  # gripper (slide)
])
JOINT_ADJUST_RATE = 0.6   # rad/s when Up/Down held

# Home pose: arm slightly raised, elbow bent
HOME_POSE = np.array([0.0, 0.5, -1.0, 0.0, 0.0])

# ---------- Drive gains ----------
DRIVE_V = 1.8    # m/s equivalent per full forward
DRIVE_W = 1.0    # rad/s equivalent per full turn

# ---------- State (mutable from key callback) ----------
key_state = {
    "forward": False, "backward": False,
    "left": False,    "right": False,
    "brake": False,
    "up": False,      "down": False,
}
arm_targets = HOME_POSE.copy()
selected_joint = 0
gripper_open = True
model = None
data = None


def on_key(window, key, scancode, act, mods):
    """GLFW key callback: toggles key_state + handles one-shot events."""
    global selected_joint, gripper_open, arm_targets

    pressed = (act == glfw.PRESS)

    # ---- Car driving (while held) ----
    if key == glfw.KEY_W:    key_state["forward"]  = pressed
    elif key == glfw.KEY_S:  key_state["backward"] = pressed
    elif key == glfw.KEY_A:  key_state["left"]     = pressed
    elif key == glfw.KEY_D:  key_state["right"]    = pressed
    elif key == glfw.KEY_SPACE: key_state["brake"] = pressed

    # ---- Arm joint selection (1-5) ----
    elif pressed and glfw.KEY_1 <= key <= glfw.KEY_5:
        idx = key - glfw.KEY_1
        if idx < len(JOINT_NAMES):
            selected_joint = idx
            lo, hi = JOINT_LIMITS[idx]
            print(f"[ARM] Selected #{idx+1}: {JOINT_NAMES[idx]} "
                  f"(current={arm_targets[idx]:+.2f}, range=[{lo:+.2f}, {hi:+.2f}])")

    # ---- Arm adjustment (while held) ----
    elif key == glfw.KEY_UP:    key_state["up"]   = pressed
    elif key == glfw.KEY_DOWN:  key_state["down"] = pressed

    # ---- Gripper toggle (one-shot) ----
    elif pressed and key == glfw.KEY_G:
        gripper_open = not gripper_open
        arm_targets[4] = 0.025 if gripper_open else -0.025
        print(f"[GRIPPER] {'OPEN' if gripper_open else 'CLOSED'}")

    # ---- Home pose (one-shot) ----
    elif pressed and key == glfw.KEY_0:
        arm_targets = HOME_POSE.copy()
        print(f"[ARM] Home pose: {HOME_POSE}")

    # ---- Reset simulation (one-shot) ----
    elif pressed and key == glfw.KEY_BACKSPACE:
        if model is not None and data is not None:
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
            arm_targets = HOME_POSE.copy()
            print("[SIM] Reset")


def controller(model, data):
    """Set all ctrl[] each step. Called by mj_step via set_mjcb_control."""

    # ---- Car: differential drive ----
    v = 0.0
    if key_state["forward"]:  v += DRIVE_V
    if key_state["backward"]: v -= DRIVE_V
    w = 0.0
    if key_state["left"]:     w -= DRIVE_W
    if key_state["right"]:    w += DRIVE_W
    if key_state["brake"]:    v = 0.0; w = 0.0

    left  = v - w   # left wheels
    right = v + w   # right wheels
    data.ctrl[0] = left    # wheel_fl
    data.ctrl[1] = right   # wheel_fr
    data.ctrl[2] = left    # wheel_rl
    data.ctrl[3] = right   # wheel_rr

    # ---- Arm: continuous adjustment of selected joint ----
    dt = model.opt.timestep
    if key_state["up"]:
        arm_targets[selected_joint] += JOINT_ADJUST_RATE * dt
    if key_state["down"]:
        arm_targets[selected_joint] -= JOINT_ADJUST_RATE * dt

    # Clamp to joint limits
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        arm_targets[i] = float(np.clip(arm_targets[i], lo, hi))

    # Write arm position-servo targets
    for i in range(len(JOINT_NAMES)):
        data.ctrl[ARM_CTRL_START + i] = arm_targets[i]


def main():
    global model, data

    # ---- Load model ----
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    print(f"Loaded model: {XML_PATH}")
    print(f"  nu (actuators) = {model.nu}")
    print(f"  njnt (joints)   = {model.njnt}")

    # ---- GLFW window ----
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(1200, 900, "Mobile Manipulator (M1)", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, on_key)

    # ---- MuJoCo render state ----
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    cam.azimuth = 135.0
    cam.elevation = -20.0
    cam.distance = 3.5
    cam.lookat = np.array([0.0, 0.0, 0.6])

    # ---- Bind controller ----
    mj.set_mjcb_control(controller)

    # ---- Help ----
    print()
    print("=" * 56)
    print("Mobile Manipulator (M1)  --  controls")
    print("=" * 56)
    print("  Car:  W/A/S/D = forward/left/backward/right")
    print("        Space   = brake")
    print("  Arm:  1-5     = select joint (1=shoulder_pan..5=gripper)")
    print("        Up/Down = adjust selected joint")
    print("        G       = toggle gripper")
    print("        0       = home pose")
    print("        Backspace = reset simulation")
    print("=" * 56)
    print(f"[ARM] Home pose: {HOME_POSE}")
    print(f"[ARM] Joint limits: {JOINT_LIMITS.tolist()}")
    print()

    # ---- Main loop ----
    simend = 600.0  # 10 minutes
    while not glfw.window_should_close(window):
        simstart = data.time
        while (data.time - simstart < 1.0 / 60.0):
            mj.mj_step(model, data)

        if data.time >= simend:
            break

        # Camera follows the car (freejoint qpos layout: [x,y,z, qw,qx,qy,qz])
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = max(0.4, data.qpos[2] + 0.3)

        viewport = mj.MjrRect(0, 0, 1200, 900)
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
