import mujoco as mj
from mujoco.glfw import glfw
import matplotlib as mpl
import matplotlib.pyplot as plt
import nlopt
import numpy as np
import os

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams["font.size"] = 16

xml_path = 'manipulator.xml'

omega = 0.4
a = 0.25
simend = 0.25 + 2 * np.pi / omega
center_x = None
center_z = None
X_target = None #np.array([0,0]);

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data):
    # Get reference end-effector position
    global X_target

    X_target = get_lemniscate_ref(data.time)

    # Use current joint angles as the initial guess
    qpos = np.array([data.qpos[0], data.qpos[1]])

    # Solve for the joint angles
    sol = inverse_kinematics(qpos)

    # Apply control
    data.ctrl[0] = sol[0]
    data.ctrl[2] = sol[1]

def forward_kinematics(self, q):
    data_sim.qpos[0] = q[0]
    data_sim.qpos[1] = q[1]
    data_sim.ctrl[0] = data_sim.qpos[0]
    data_sim.ctrl[2] = data_sim.qpos[1]

    mj.mj_forward(model, data_sim)

    end_eff_pos = np.array([
        data_sim.sensordata[0],
        data_sim.sensordata[2]
    ])

    return end_eff_pos

def init_controller(model, data):
    global center_x
    global center_z
    global X_target

    # Get center of Lemniscate
    end_eff_pos = forward_kinematics([-0.5, 1.0])
    center_x = end_eff_pos[0] - 0.25
    center_z = end_eff_pos[1]

    # Get initial joint angles
    q_guess = np.array([-0.5, 1.0])
    X_target = get_lemniscate_ref(0.0)
    q_pos = inverse_kinematics(q_guess)

    data.qpos[0] = q_pos[0]
    data.qpos[1] = q_pos[1]

def forward_kinematics(q):
    data_sim.qpos[0] = q[0]
    data_sim.qpos[1] = q[1]
    data_sim.ctrl[0] = data_sim.qpos[0]
    data_sim.ctrl[2] = data_sim.qpos[1]

    mj.mj_forward(model, data_sim)

    end_eff_pos = np.array([
        data_sim.sensordata[0],
        data_sim.sensordata[2]
    ])

    return end_eff_pos

def cost_func(x, grad):
    cost = 0.0
    return cost

def equality_constraints(result, x, grad):
    global X_target

    end_eff_pos = forward_kinematics(x)
    result[0] = end_eff_pos[0] - X_target[0]
    result[1] = end_eff_pos[1] - X_target[1]

def inverse_kinematics(x):
    # Define optimization problem
    opt = nlopt.opt(nlopt.LN_COBYLA, 2)

    # Define lower and upper bounds
    opt.set_lower_bounds([-np.pi, -np.pi])
    opt.set_upper_bounds([np.pi, np.pi])

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

def get_lemniscate_ref(t):
    global center_x
    global center_z
    global omega
    global a

    wt = omega * t
    denominator = 1 + np.sin(wt) * np.sin(wt)

    x = center_x + (a * np.cos(wt)) / denominator
    z = center_z + (a * np.sin(wt) * np.cos(wt)) / denominator

    ref_pos = np.array([x, z])

    return ref_pos

def graph():
    # Measured motion trajectory
    global end_eff_pos
    end_eff_pos_arr = np.concatenate(end_eff_pos, axis=1)

    # Get reference trajectory
    wt = omega * np.linspace(0.0, simend, 500)
    denominator = 1 + np.sin(wt) * np.sin(wt)
    leminiscate_x = center_x + (a * np.cos(wt)) / denominator
    leminiscate_z = center_z + \
        (a * np.sin(wt) * np.cos(wt)) / denominator

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(
        end_eff_pos_arr[0, :],
        end_eff_pos_arr[1, :],
        color="cornflowerblue",
        linewidth=4,
        zorder=-2,
        label=r"$\textbf{Inverse Kinematics}$"
    )
    ax.plot(
        leminiscate_x,
        leminiscate_z,
        color="darkorange",
        linewidth=1,
        zorder=-1,
        label=r"$\textbf{Reference}$"
    )

    ax.grid()
    ax.set_aspect("equal")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.4))


    #plt.show()
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    # plt.savefig("imgs/manipulator_ik.png", dpi=200,
    #             transparent=False, bbox_inches="tight")


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
data_sim = mj.MjData(model)            #data structure for forward/inverse kinematics
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

init_controller(model,data)
# Create list to store data
end_eff_pos = []

#set the controller
# mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 0.1): # Use 1/60 = 0.01667 for slower animation
        controller(model, data)
        mj.mj_step(model, data)

    end_eff_pos_temp = np.array([
        data.sensordata[0],
        data.sensordata[2]])
    end_eff_pos.append(end_eff_pos_temp[:, np.newaxis])

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
graph()
