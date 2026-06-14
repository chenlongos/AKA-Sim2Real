import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
NLOPT_IMPORTED = True
#NOTE: If this gives numpy error try,  pip install numpy --upgrade
try:
    import nlopt
except ImportError:
    print("nlopt not imported, switching to pre-computed solution")
    NLOPT_IMPORTED = False

xml_path = 'ball.xml'
simend = 5

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data):
    #put the controller here
    pass

def simulator(x):
    v, theta, time_of_flight = x[0], x[1], x[2]

    data.qvel[0] = v * np.cos(theta)
    data.qvel[2] = v * np.sin(theta)

    while data.time < time_of_flight:
        # Step simulation environment
        mj.mj_step(model, data)

    # Get position
    pos = np.array([data.qpos[0], data.qpos[2]])

    # Reset Data
    mj.mj_resetData(model, data)

    return pos

def init_controller(model,data):

    # Set initial guess
    v = 10.0
    theta = np.pi / 4
    time_of_flight = 2.0

    if NLOPT_IMPORTED:
        sol = optimize_ic(np.array([v, theta, time_of_flight]))
    else:
        sol = np.array([9.398687489285555, 1.2184054599970882, 1.5654456340479144])

    v_sol, theta_sol = sol[0], sol[1]
    simend = sol[2] + 2

    data.qvel[0] = v_sol * np.cos(theta_sol)
    data.qvel[2] = v_sol * np.sin(theta_sol)

def cost_func(x, grad):
    cost = 0.0
    return cost

def equality_constraints(result, x, grad):
    """
    For details of the API please refer to:
    https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#:~:text=remove_inequality_constraints()%0Aopt.remove_equality_constraints()-,Vector%2Dvalued%20constraints,-Just%20as%20for
    Note: Please open the link in Chrome
    """
    pos = simulator(x)
    result[0] = pos[0] - 5.0
    result[1] = pos[1] - 2.1

def optimize_ic(x):
    """
    Optimization problem is

         min_X      0
    subject to 0.1 ≤ v ≤ ∞
               0.1 ≤ θ ≤ π/2
               0.1 ≤ T ≤ ∞
               x(T) = x^*
               z(T) = z^*

    with X = [v, θ, T]
    """
    # Define optimization problem
    opt = nlopt.opt(nlopt.LN_COBYLA, 3)

    # Define lower and upper bounds
    opt.set_lower_bounds([0.1, 0.1, 0.1])
    opt.set_upper_bounds([10000.0, np.pi / 2 - 0.1, 10000.0])

    # Set objective funtion
    opt.set_min_objective(cost_func)

    # Define equality constraints
    tol = [1e-4, 1e-4]
    opt.add_equality_mconstraint(equality_constraints, tol)

    # Set relative tolerance on optimization parameters
    opt.set_xtol_rel(1e-4)

    # Solve problem
    sol = opt.optimize(x)

    return sol

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
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

# # Set initial guess
# v = 10.0
# theta = np.pi / 4
# time_of_flight = 2.0
#
# if NLOPT_IMPORTED:
#     sol = optimize_ic(np.array([v, theta, time_of_flight]))
# else:
#     sol = np.array([9.398687489285555, 1.2184054599970882, 1.5654456340479144])
#
# v_sol, theta_sol = sol[0], sol[1]
# simend = sol[2] + 2
#
# data.qvel[0] = v_sol * np.cos(theta_sol)
# data.qvel[2] = v_sol * np.sin(theta_sol)

#initialize the controller
init_controller(model,data)

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
    cam.lookat[0] = data.qpos[0] #camera moves with the ball
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
