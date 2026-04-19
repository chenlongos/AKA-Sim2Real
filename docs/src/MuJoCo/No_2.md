# No.2 交互式仿真与鼠标控制

本节介绍如何使用 GLFW 窗口创建交互式 3D 仿真，包括鼠标视角控制和控制器回调。

---

## 文件说明

本节的示例文件位于 `mujoco/No_2/temp_mjcpy/` 目录下：

```
mujoco/No_2/temp_mjcpy/
├── ball.xml              # MuJoCo XML 场景描述文件
├── projectile.py         # 基础交互式仿真脚本
└── template_mujoco.py    # 带详细注释的模板脚本
```

---

## ball.xml 详解

一个简单的弹球场景：

```xml
<mujoco>
  <worldbody>
    <!-- 场景光源 -->
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

    <!-- 红色平面地面 -->
    <geom type="plane" size="10 1 0.1" rgba=".9 0 0 1"/>

    <!-- 绿色小球，初始位置 z=1 -->
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size=".1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
```

| 元素 | 说明 |
|------|------|
| `plane` | 平面地面，半尺寸 10×1×0.1 |
| `sphere` | 绿色小球，半径 0.1 |

---

## template_mujoco.py 详解

### 核心结构

```python
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# 1. 模型加载
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()      # 视角相机
opt = mj.MjvOption()      # 可视化选项

# 2. GLFW 窗口初始化
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# 3. 可视化数据结构
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 4. 注册回调函数
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)
mj.set_mjcb_control(controller)

# 5. 主循环
while not glfw.window_should_close(window):
    mj.mj_step(model, data)
    # 渲染...
    glfw.poll_events()

glfw.terminate()
```

### 回调函数

#### controller

每步仿真前调用的控制回调，可写入外力：

```python
def controller(model, data):
    """控制回调，每 mj_step 前自动调用"""
    pass
```

在 `projectile.py` 中的示例实现了空气阻力：

```python
def controller(model, data):
    vx, vy, vz = data.qvel[0], data.qvel[1], data.qvel[2]
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    c = 1.0  # 阻力系数
    data.qfrc_applied[0] = -c * v * vx
    data.qfrc_applied[1] = -c * v * vy
    data.qfrc_applied[2] = -c * v * vz
```

#### keyboard

键盘事件处理：

```python
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
```

| 按键 | 动作 |
|------|------|
| Backspace | 重置仿真到初始状态 |

#### mouse_move

鼠标拖动改变视角：

| 组合 | 动作 |
|------|------|
| 左键拖动 | 旋转视角 |
| 右键拖动 | 移动视角 |
| 中键拖动 | 缩放 |
| Shift + 拖动 | 切换交互模式 |

#### scroll

滚轮缩放视角。

---

## 交互操作

### 视角控制

| 操作 | 功能 |
|------|------|
| 左键拖动 | 旋转视角（水平/垂直） |
| 右键拖动 | 移动视角 |
| 滚轮 | 缩放 |
| Shift + 拖动 | 切换模式 |

### 键盘

| 按键 | 功能 |
|------|------|
| Backspace | 重置仿真 |

---

## 初始条件设置

```python
# 设置小球初始位置
data.qpos[2] = 0.1

# 设置小球初速度 (vx=2, vy=0, vz=5)
data.qvel[0] = 2.0
data.qvel[2] = 5.0

# 设置相机视角
cam.azimuth = 90.0
cam.distance = 8.0
cam.elevation = -45.0
```

---

## 运行方法

在 `mujoco/No_2/temp_mjcpy/` 目录下执行：

```bash
python projectile.py
# 或带注释模板
python template_mujoco.py
```

运行效果：绿色弹球以初速度 (2, 0, 5) 抛出，受重力下落并受空气阻力影响。

---

## 与 No.1 的区别

| 特性 | No.1 (Passive) | No.2 (GLFW) |
|------|----------------|-------------|
| 窗口 | `mujoco.viewer` 被动窗口 | GLFW 主动创建窗口 |
| 视角控制 | 受限 | 鼠标自由控制 |
| 控制回调 | 无 | `mj.set_mjcb_control` |
| 渲染控制 | 自动同步 | 手动调用 `mjr_render` |
| 帧率控制 | 自动 | 手动控制循环频率 |
