import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# XML 模型文件路径
xml_path = 'ball.xml'
# 仿真结束时间（秒），simend=20 表示仿真运行 20 秒后自动停止
simend = 20

# ============================================================
# 鼠标状态变量（用于鼠标拖动交互）
# ============================================================
button_left = False   # 鼠标左键是否按下
button_middle = False # 鼠标中键是否按下
button_right = False  # 鼠标右键是否按下
lastx = 0             # 上一次鼠标 x 位置
lasty = 0             # 上一次鼠标 y 位置

# ============================================================
# controller 回调：每一步仿真前调用，可在此写入控制指令
# ============================================================
def controller(model, data):
    """MuJoCo 控制回调，每 mj_step 前自动调用"""
    pass

# ============================================================
# keyboard 回调：键盘事件处理
# ============================================================
def keyboard(window, key, scancode, act, mods):
    """
    Backspace: 重置仿真到初始状态
    - mj_resetData: 重置 data（位置、速度等）
    - mj_forward: 前向计算（更新受力等）
    """
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

# ============================================================
# mouse_button 回调：记录鼠标按键状态
# ============================================================
def mouse_button(window, button, act, mods):
    """记录左/中/右键按下状态，供 mouse_move 使用"""
    global button_left, button_middle, button_right
    button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS)

# ============================================================
# mouse_move 回调：鼠标拖动改变视角
# ============================================================
def mouse_move(window, xpos, ypos):
    """
    鼠标拖动交互：
    - 左键拖动: 旋转视角（水平/垂直）
    - 右键拖动: 移动视角（水平/垂直）
    - 中键拖动: 缩放视角
    - Shift + 拖动: 切换为另一种交互模式
    """
    global lastx, lasty, button_left, button_middle, button_right
    # 计算鼠标位移
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    # 无按键按下时不做任何事
    if not button_left and not button_middle and not button_right:
        return
    width, height = glfw.get_window_size(window)
    # 检测 Shift 键状态，切换交互模式
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT)  == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    # 根据按键和 Shift 状态决定动作类型
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    # 移动相机视角
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

# ============================================================
# scroll 回调：滚轮缩放视角
# ============================================================
def scroll(window, xoffset, yoffset):
    """滚轮滚动缩放视角（向上滚放大，向下滚缩小）"""
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05*yoffset, scene, cam)

# ============================================================
# 模型加载
# ============================================================
dirname = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.join(dirname, xml_path)

# MjModel: 包含模型结构（body、joint、geom 等）
model = mj.MjModel.from_xml_path(abspath)
# MjData: 包含仿真运行时数据（位置、速度、力等）
data = mj.MjData(model)
# MjvCamera: 视角相机（位置、朝向等）
cam = mj.MjvCamera()
# MjvOption: 可视化选项（显示哪些元素）
opt = mj.MjvOption()

# ============================================================
# GLFW 初始化和窗口创建
# ============================================================
glfw.init()
# 创建窗口（宽1200、高900、标题"Demo"）
window = glfw.create_window(1200, 900, "Demo", None, None)
# 将窗口上下文设为当前 OpenGL 上下文
glfw.make_context_current(window)
# 垂直同步：等待显示器刷新率（通常 60fps）
glfw.swap_interval(1)

# ============================================================
# MuJoCo 可视化数据结构初始化
# ============================================================
mj.mjv_defaultCamera(cam)    # 设置相机为默认参数
mj.mjv_defaultOption(opt)    # 设置可视化选项为默认值
# MjvScene: 管理渲染场景的几何体（maxgeom=10000 最多10000个物体）
scene = mj.MjvScene(model, maxgeom=10000)
# MjrContext: OpenGL 渲染上下文（字体缩放）
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ============================================================
# 注册 GLFW 回调和 MuJoCo 控制回调
# ============================================================
glfw.set_key_callback(window, keyboard)           # 键盘事件
glfw.set_cursor_pos_callback(window, mouse_move) # 鼠标移动
glfw.set_mouse_button_callback(window, mouse_button)  # 鼠标按键
glfw.set_scroll_callback(window, scroll)         # 鼠标滚轮
# 控制回调：每步仿真前调用 controller 函数
mj.set_mjcb_control(controller)

# ============================================================
# 主仿真循环
# ============================================================
while not glfw.window_should_close(window):
    # 记录本帧开始时间
    simstart = data.time
    # 仿真直到本帧时间到达 1/60 秒（16.67ms）
    while data.time - simstart < 1.0/60.0:
        mj.mj_step(model, data)  # 执行一步仿真
    # 超过 simend 秒则退出循环
    if data.time >= simend:
        break

    # ---- 渲染部分 ----
    # 获取窗口帧缓冲区大小
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # 更新场景（几何体位置、法线等）
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    # 将场景渲染到窗口
    mj.mjr_render(viewport, scene, context)

    # 交换前后缓冲区（阻塞等待垂直同步）
    glfw.swap_buffers(window)
    # 处理待处理的 GUI 事件（鼠标、键盘等回调）
    glfw.poll_events()

# ============================================================
# 程序退出，清理 GLFW
# ============================================================
glfw.terminate()
