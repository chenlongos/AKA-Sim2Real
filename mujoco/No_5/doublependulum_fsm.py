import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = 'doublependulum_fsm.xml'
simend = 5

t_hold = 0.5
t_swing1 = 1.0
t_swing2 = 1.0

FSM_HOLD = 0
FSM_SWING1 = 1
FSM_SWING2 = 2
FSM_STOP = 3

# fsm_state = FSM_HOLD;

# Define setpoints
q_init = np.array([[-1.0], [0.0]])
q_mid = np.array([[0.5], [-2.0]])
q_end = np.array([[1.0], [0.0]])

# Define setpoint times
t_init = t_hold
t_mid = t_hold + t_swing1
t_end = t_hold + t_swing1 + t_swing2

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):

    global fsm_state
    global a_swing1, a_swing2
    fsm_state = FSM_HOLD;

    a_swing1 = generate_trajectory(
        t_init, t_mid, q_init, q_mid)

    a_swing2 = generate_trajectory(
        t_mid, t_end, q_mid, q_end)

def controller(model, data):
    """
    This function implements a PD controller for tracking
    the reference motion.
    """
    global fsm_state
    global a_swing1, a_swing2
    time = data.time

    # Check for state change
    if fsm_state == FSM_HOLD and time >= t_hold:
        fsm_state = FSM_SWING1
    elif fsm_state == FSM_SWING1 and time >= t_mid:
        fsm_state = FSM_SWING2
    elif fsm_state == FSM_SWING2 and time >= t_end:
        fsm_state = FSM_STOP

    # Get reference joint position & velocity
    if fsm_state == FSM_HOLD:
        q_ref = q_init
        dq_ref = np.zeros((2, 1))
    elif fsm_state == FSM_SWING1:
        q_ref = a_swing1[0] + a_swing1[1]*time + \
            a_swing1[2]*(time**2) + a_swing1[3]*(time**3)
        dq_ref = a_swing1[1] + 2 * a_swing1[2] * \
            time + 3 * a_swing1[3]*(time**2)
    elif fsm_state == FSM_SWING2:
        q_ref = a_swing2[0] + a_swing2[1]*time + \
            a_swing2[2]*(time**2) + a_swing2[3]*(time**3)
        dq_ref = a_swing2[1] + 2 * a_swing2[2] * \
            time + 3 * a_swing2[3]*(time**2)
    elif fsm_state == FSM_STOP:
        q_ref = q_end
        dq_ref = np.zeros((2, 1))

    # Define PD gains
    kp = 500
    kv = 50

    # Compute PD control
    torque = kp * (q_ref[:, 0] - data.qpos) + \
         kv * (dq_ref[:, 0] - data.qvel)

    for i in range(0,6):
        data.ctrl[i]=0;

    data.ctrl[0] = torque[0];
    data.ctrl[3] = torque[1];


def generate_trajectory(t0, tf, q0, qf):
    """
    Generates a trajectory
    q(t) = a0 + a1t + a2t^2 + a3t^3
    which satisfies the boundary condition
    q(t0) = q0, q(tf) = qf, dq(t0) = 0, dq(tf) = 0
    """
    tf_t0_3 = (tf - t0)**3
    a0 = qf*(t0**2)*(3*tf-t0) + q0*(tf**2)*(tf-3*t0)
    a0 = a0 / tf_t0_3

    a1 = 6 * t0 * tf * (q0 - qf)
    a1 = a1 / tf_t0_3

    a2 = 3 * (t0 + tf) * (qf - q0)
    a2 = a2 / tf_t0_3

    a3 = 2 * (q0 - qf)
    a3 = a3 / tf_t0_3

    return a0, a1, a2, a3

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

# Set initial condition
data.qpos[0] = -1

# Set camera configuration
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

#initialize the controller
init_controller(model,data);

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
