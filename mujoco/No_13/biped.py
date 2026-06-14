import mujoco as mj
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

xml_path = 'biped.xml'
simend = 30

step_no = 0;

FSM_LEG1_SWING = 0
FSM_LEG2_SWING = 1

FSM_KNEE1_STANCE = 0
FSM_KNEE1_RETRACT = 1

FSM_KNEE2_STANCE = 0
FSM_KNEE2_RETRACT = 1

fsm_hip = FSM_LEG2_SWING
fsm_knee1 = FSM_KNEE1_STANCE
fsm_knee2 = FSM_KNEE2_STANCE


# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data):
    """
    This function implements a controller that
    mimics the forces of a fixed joint before release
    """
    global fsm_hip
    global fsm_knee1
    global fsm_knee2
    global step_no

    # State Estimation
    quat_leg1 = data.xquat[1, :]
    euler_leg1 = quat2euler(quat_leg1)
    abs_leg1 = -euler_leg1[1]
    pos_foot1 = data.xpos[2, :]

    quat_leg2 = data.xquat[3, :]
    euler_leg2 = quat2euler(quat_leg2)
    abs_leg2 = -euler_leg2[1]
    pos_foot2 = data.xpos[4, :]

    # Transition check
    if fsm_hip == FSM_LEG2_SWING and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
        fsm_hip = FSM_LEG1_SWING
    if fsm_hip == FSM_LEG1_SWING and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
        fsm_hip = FSM_LEG2_SWING

    if fsm_knee1 == FSM_KNEE1_STANCE and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
        fsm_knee1 = FSM_KNEE1_RETRACT
    if fsm_knee1 == FSM_KNEE1_RETRACT and abs_leg1 > 0.1:
        fsm_knee1 = FSM_KNEE1_STANCE

    if fsm_knee2 == FSM_KNEE2_STANCE and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
        fsm_knee2 = FSM_KNEE2_RETRACT
    if fsm_knee2 == FSM_KNEE2_RETRACT and abs_leg2 > 0.1:
        fsm_knee2 = FSM_KNEE2_STANCE

    # Control
    if fsm_hip == FSM_LEG1_SWING:
        data.ctrl[0] = -0.5
    if fsm_hip == FSM_LEG2_SWING:
        data.ctrl[0] = 0.5

    if fsm_knee1 == FSM_KNEE1_STANCE:
        data.ctrl[2] = 0.0
    if fsm_knee1 == FSM_KNEE1_RETRACT:
        data.ctrl[2] = -0.25

    if fsm_knee2 == FSM_KNEE2_STANCE:
        data.ctrl[4] = 0.0
    if fsm_knee2 == FSM_KNEE2_RETRACT:
        data.ctrl[4] = -0.25

def init_controller(model,data):
    data.qpos[4] = 0.5
    data.ctrl[0] = data.qpos[4]

def quat2euler(quat):
    # SciPy defines quaternion as [x, y, z, w]
    # MuJoCo defines quaternion as [w, x, y, z]
    _quat = np.concatenate([quat[1:], quat[:1]])
    r = R.from_quat(_quat)

    # roll-pitch-yaw is the same as rotating w.r.t
    # the x, y, z axis in the world frame
    euler = r.as_euler('xyz', degrees=False)

    return euler

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Set camera configuration
cam.azimuth = 120.89  # 89.608063
cam.elevation = -15.81  # -11.588379
cam.distance = 8.0  # 5.0
cam.lookat = np.array([0.0, 0.0, 2.0])

#turn the direction of gravity to simulate a ramp
model.opt.gravity[0] = 9.81 * np.sin(0.1)
model.opt.gravity[2] = -9.81 * np.cos(0.1)

init_controller(model,data)

#set the controller
#mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        #simulation step
        mj.mj_step(model, data)
        # Apply control
        controller(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Show joint frames
    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

    # Update scene and render
    cam.lookat[0] = data.qpos[0] #camera follows the robot
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
