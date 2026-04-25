# No.3 单摆控制仿真

本节介绍倒立摆（Inverted Pendulum）的 MuJoCo 建模与闭环控制，包括关节电机、传感器配置，以及两种控制模式（力矩模式 vs 伺服模式）。

---

## 文件说明

本节的示例文件位于 `mujoco/No_3/` 目录下：

```
mujoco/No_3/
├── no_3.py              # 最小主脚本（使用 viewer.launch_passive）
├── control_pendulum.py   # 完整交互脚本（使用 GLFW）
└── pendulum.xml         # MuJoCo XML 模型文件
```

---

## 一、pendulum.xml 详解（对比 No.2 的 ball.xml）

### No.3 pendulum.xml 完整代码

```xml
<mujoco>
    <!-- 全局仿真选项：重力加速度 -->
    <option gravity="0 0 -9.81">
    </option>

    <worldbody>
        <!-- 场景光源 -->
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

        <!-- 红色平面地面 -->
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

        <!-- 摆杆 body：固定在 (0, 0, 2)，绕 y 轴旋转 -->
        <body pos="0 0 2" euler="0 180 0">
            <!--
                hinge 关节：约束为只能绕一个轴旋转
                对比 No.2：No.2 使用 free 关节（6 自由度）
                          No.3 使用 hinge 关节（1 自由度）
            -->
            <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
            <!-- 绿色圆柱几何体，半径 0.05，长度 0.5，质量 1 -->
            <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
        </body>
    </worldbody>

    <!--
        actuator 驱动器：【No.3 新增】
        对比 No.2：No.2 没有 actuator，只有纯物理仿真
                  No.3 新增了 3 种驱动器用于闭环控制
    -->
    <actuator>
        <!-- 力矩电机：直接控制关节力矩 -->
        <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
        <!-- 位置伺服：使用 PD 控制跟踪目标位置 -->
        <position name="position_servo" joint="pin" kp="10"/>
        <!-- 速度伺服：使用 PD 控制跟踪目标速度 -->
        <velocity name="velocity_servo" joint="pin" kv="0"/>
    </actuator>

    <!--
        sensor 传感器：【No.3 新增】
        对比 No.2：No.2 没有 sensor
                  No.3 新增了位置和速度传感器用于闭环反馈
    -->
    <sensor>
        <!-- 关节位置传感器，带噪声 -->
        <jointpos joint="pin" noise="0.2"/>
        <!-- 关节速度传感器，带噪声 -->
        <jointvel joint="pin" noise="1"/>
    </sensor>
</mujoco>
```

### No.2 ball.xml 完整代码（对比参考）

```xml
<mujoco>
    <worldbody>
        <!-- 场景光源 -->
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

        <!-- 红色平面地面 -->
        <geom type="plane" size="10 1 0.1" rgba=".9 0 0 1"/>

        <!-- 绿色小球，初始位置 z=1 -->
        <body pos="0 0 1">
            <!-- free 关节：6 自由度，可自由平移和旋转 -->
            <joint type="free"/>
            <geom type="sphere" size=".1" rgba="0 .9 0 1"/>
        </body>
    </worldbody>
    <!-- No.2 没有 actuator 和 sensor -->
</mujoco>
```

### XML 配置对比表

| 配置项 | No.2 (ball.xml) | No.3 (pendulum.xml) |
|--------|-----------------|---------------------|
| 关节类型 | `free`（6 自由度） | `hinge`（1 自由度） |
| 关节名称 | 无 | `name="pin"` |
| 几何体 | `sphere`（球） | `cylinder`（圆柱） |
| actuator | **无** | 3 个（motor、position、velocity） |
| sensor | **无** | 2 个（jointpos、jointvel） |
| mass 定义 | 无（自动计算） | `mass="1"` 显式指定 |

### 关节类型详解

| 关节类型 | 自由度 | 适用场景 | No.2 | No.3 |
|----------|--------|----------|------|------|
| `free` | 6 | 自由运动物体（球、无人机） | ✅ | ❌ |
| `hinge` | 1 | 旋转关节（摆、门） | ❌ | ✅ |
| `slide` | 1 | 滑动关节（活塞） | - | - |
| `ball` | 3 | 球形关节（机械臂） | - | - |

---

## 二、no_3.py 详解（最小脚本）

### 完整代码

```python
import mujoco
import mujoco.viewer
import time

# 加载 XML 模型
model = mujoco.MjModel.from_xml_path('pendulum.xml')
data = mujoco.MjData(model)

# 使用 viewer.launch_passive 启动被动查看器
# 对比 No.2：No.2 的 template_mujoco.py 使用完整 GLFW 窗口（170 行）
#           No.3 的 no_3.py 使用简化 viewer（仅 12 行核心代码）
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1/500)  # ~60 Hz real-time，约 500ms 休眠实现实时仿真
```

### 核心 API 对比

| API | No.1 (no_1.py) | No.3 (no_3.py) |
|-----|----------------|----------------|
| 模型加载 | `MjModel.from_xml_path()` | `MjModel.from_xml_path()` |
| 可视化 | `viewer.launch_passive()` | `viewer.launch_passive()` |
| 仿真步进 | `mj_step()` | `mj_step()` |
| 同步方式 | `viewer.sync()` | `viewer.sync()` |
| 主循环条件 | `viewer.is_running()` | `viewer.is_running()` |
| 帧率控制 | 无 | `time.sleep(1/500)` |

---

## 三、control_pendulum.py 详解（完整脚本）

### 完整代码

```python
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# XML 模型文件路径
xml_path = 'pendulum.xml'
# 仿真结束时间（秒），simend=5 表示仿真运行 5 秒后自动停止
simend = 5

# ============================================================
# 鼠标状态变量（用于鼠标拖动交互）
# 【与 No.2 完全相同】
# ============================================================
button_left = False   # 鼠标左键是否按下
button_middle = False # 鼠标中键是否按下
button_right = False  # 鼠标右键是否按下
lastx = 0             # 上一次鼠标 x 位置
lasty = 0             # 上一次鼠标 y 位置

# ============================================================
# controller 回调：每一步仿真前调用，可在此写入控制指令
# 【核心差异】No.2 的 controller 为空 pass，No.3 实现 PD 控制
# ============================================================
def controller(model, data):
    """
    PD 控制器实现

    对比 No.2：
    - No.2 的 controller 为空 pass，无任何控制逻辑
    - No.3 的 controller 根据 actuator_type 实现两种控制模式
    """
    if actuator_type == "torque":
        # 力矩模式：直接写入控制量
        # 对比 No.2：No.2 没有 actuator_gainprm 配置
        model.actuator_gainprm[0, 0] = 1
        # PD 控制：ctrl = -kp * pos_error - kv * vel_error
        data.ctrl[0] = -10 * \
            (data.sensordata[0] - 0.0) - \
            1 * (data.sensordata[1] - 0.0)
    elif actuator_type == "servo":
        # 伺服模式：配置增益参数
        kp = 10.0
        model.actuator_gainprm[1, 0] = kp
        model.actuator_biasprm[1, 1] = -kp
        data.ctrl[1] = -0.5

        kv = 1.0
        model.actuator_gainprm[2, 0] = kv
        model.actuator_biasprm[2, 2] = -kv
        data.ctrl[2] = 0.0

# ============================================================
# keyboard 回调：键盘事件处理
# 【与 No.2 完全相同】
# ============================================================
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

# ============================================================
# mouse_button 回调：记录鼠标按键状态
# 【与 No.2 完全相同】
# ============================================================
def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

# ============================================================
# mouse_move 回调：鼠标拖动改变视角
# 【与 No.2 完全相同】
# ============================================================
def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not button_left and not button_middle and not button_right:
        return

    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

# ============================================================
# scroll 回调：滚轮缩放视角
# 【与 No.2 完全相同】
# ============================================================
def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05*yoffset, scene, cam)

# ============================================================
# 模型加载
# 【与 No.2 完全相同】
# ============================================================
dirname = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.join(dirname, xml_path)

model = mj.MjModel.from_xml_path(abspath)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# ============================================================
# GLFW 初始化和窗口创建
# 【与 No.2 完全相同】
# ============================================================
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# ============================================================
# MuJoCo 可视化数据结构初始化
# 【与 No.2 完全相同】
# ============================================================
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# ============================================================
# 注册 GLFW 回调和 MuJoCo 控制回调
# 【与 No.2 完全相同】
# ============================================================
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)
mj.set_mjcb_control(controller)

# ============================================================
# 初始条件设置
# 【核心差异】
# No.2：data.qvel[0] = 2.0（设置初速度）
# No.3：data.qpos[0] = np.pi/2（设置初始角度）
# ============================================================
data.qpos[0] = np.pi/2  # 摆杆初始角度 90°（朝上位置）

# 设置相机视角
cam.azimuth = 90.0
cam.distance = 5.0
cam.elevation = -5
cam.lookat = np.array([0.012768, -0.000000, 1.254336])

# ============================================================
# 主仿真循环
# 【与 No.2 完全相同】
# ============================================================
while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time >= simend):
        break

    # 获取窗口帧缓冲区大小
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # 更新场景并渲染
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # 交换 OpenGL 缓冲区
    glfw.swap_buffers(window)
    # 处理 GUI 事件
    glfw.poll_events()

glfw.terminate()
```

### 控制模式详解

#### 模式 1：力矩模式（torque）

```python
model.actuator_gainprm[0, 0] = 1
data.ctrl[0] = -10 * (data.sensordata[0] - 0.0) - 1 * (data.sensordata[1] - 0.0)
```

| 参数 | 说明 | 来源 |
|------|------|------|
| `sensordata[0]` | 关节位置（rad） | `<sensor><jointpos>` |
| `sensordata[1]` | 关节速度（rad/s） | `<sensor><jointvel>` |
| `kp=10` | 位置增益 | 手动设置 |
| `kv=1` | 速度增益 | 手动设置 |
| 目标位置 | 0.0 | 手动设置 |

#### 模式 2：伺服模式（servo）

```python
kp = 10.0
model.actuator_gainprm[1, 0] = kp
model.actuator_biasprm[1, 1] = -kp
data.ctrl[1] = -0.5  # 目标位置
```

使用 XML 中定义的 `position_servo`，通过配置增益参数实现 PD 控制。

---

## 四、No.2 与 No.3 完整代码对比

### 代码结构对比

| 模块 | No.2 (template_mujoco.py) | No.3 (control_pendulum.py) |
|------|--------------------------|---------------------------|
| import | ✅ | ✅ |
| 全局变量（鼠标状态） | ✅ | ✅ |
| controller 回调 | `pass`（空） | **PD 控制器** |
| keyboard 回调 | ✅ | ✅ |
| mouse_button 回调 | ✅ | ✅ |
| mouse_move 回调 | ✅ | ✅ |
| scroll 回调 | ✅ | ✅ |
| 模型加载 | ✅ | ✅ |
| GLFW 初始化 | ✅ | ✅ |
| 可视化结构 | ✅ | ✅ |
| 回调注册 | ✅ | ✅ |
| 初始条件 | `qvel` 设置速度 | **`qpos` 设置角度** |
| 主循环 | ✅ | ✅ |
| glfw.terminate | ✅ | ✅ |

### 关键差异代码片段

#### 1. 初始条件设置

**No.2（template_mujoco.py）：**
```python
# No initial velocity in template, but in projectile.py:
data.qvel[0] = 2.0  # vx
data.qvel[2] = 5.0  # vz
```

**No.3（control_pendulum.py）：**
```python
data.qpos[0] = np.pi/2  # 摆杆初始角度 90°（朝上）
```

#### 2. 控制回调

**No.2（template_mujoco.py）：**
```python
def controller(model, data):
    """控制回调，每 mj_step 前自动调用"""
    pass  # 无任何控制逻辑
```

**No.3（control_pendulum.py）：**
```python
actuator_type = "torque"  # 或 "servo"

def controller(model, data):
    if actuator_type == "torque":
        model.actuator_gainprm[0, 0] = 1
        data.ctrl[0] = -10 * (data.sensordata[0] - 0.0) - 1 * (data.sensordata[1] - 0.0)
    elif actuator_type == "servo":
        kp = 10.0
        model.actuator_gainprm[1, 0] = kp
        model.actuator_biasprm[1, 1] = -kp
        data.ctrl[1] = -0.5
```

---

## 五、运行方法

在 `mujoco/No_3/` 目录下执行：

```bash
# 最小脚本（仅可视化，无控制）
python no_3.py

# 完整脚本（带控制）
python control_pendulum.py
```

运行效果：
- `no_3.py`：摆杆从 90° 位置自由摆动（无控制），仅做可视化
- `control_pendulum.py`：摆杆在 PD 控制下稳定在目标位置（0 rad）

---

## 六、与 No.2 的整体对比总结

### 功能特性对比

| 特性 | No.2 (弹球) | No.3 (倒立摆) |
|------|-------------|---------------|
| **模型类型** | 弹球 + 地面 | 单摆 + 地面 |
| **关节类型** | `free`（6 自由度） | `hinge`（1 自由度） |
| **驱动器** | 无 | 3 个（motor、position、velocity） |
| **传感器** | 无 | 2 个（jointpos、jointvel） |
| **控制方式** | 无（被动仿真） | 力矩控制 + 伺服控制 |
| **初始状态设置** | `data.qvel`（速度） | `data.qpos`（角度） |
| **主脚本行数** | 170 行（GLFW） | 12 行（viewer）/ 171 行（GLFW） |
| **控制器复杂度** | 无 | PD 控制实现 |

### 学习路径

```
No.1: 基础建模 + viewer 可视化（被动窗口）
  ↓
No.2: GLFW 窗口 + 鼠标交互 + 回调机制 + 弹球物理
  ↓
No.3: 关节控制 + 传感器读取 + 闭环控制 + 倒立摆
```

### 代码复用情况

| 代码模块 | No.2 → No.3 |
|----------|-------------|
| 鼠标状态变量 | 完全相同 |
| keyboard 回调 | 完全相同 |
| mouse_button 回调 | 完全相同 |
| mouse_move 回调 | 完全相同 |
| scroll 回调 | 完全相同 |
| GLFW 初始化 | 完全相同 |
| 可视化结构 | 完全相同 |
| 回调注册 | 完全相同 |
| 主循环 | 完全相同 |
| **controller 回调** | **完全不同（空 → PD）** |
| **初始条件** | **完全不同（速度 → 角度）** |
| **XML 模型** | **完全不同（弹球 → 倒立摆）** |

---

## 七、常见问题

### 1. 摆杆直接穿透地面

**原因**：初始位置或关节配置错误。

**解决**：检查 `euler="0 180 0"` 确保摆杆朝下，检查 `joint axis` 方向。

### 2. 控制不稳定

**原因**：
- 比例增益（kp）过大
- 缺少重力补偿
- 传感器噪声过大

**解决**：调整 `kp`、`kv` 参数，或启用重力补偿。

### 3. `AttributeError: 'NoneType' object has no attribute '...'`

**原因**：GLFW 窗口未正确创建。

**解决**：`glfw.create_window()` 返回 None 时，检查显示器配置或降低分辨率。