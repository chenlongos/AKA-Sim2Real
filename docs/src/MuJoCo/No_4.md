# No.4 双摆控制仿真

本节介绍双摆（Double Pendulum）的 MuJoCo 建模与反馈线性化控制，包括多体系统层级结构、RK4 积分器、以及基于动力学模型的控制器设计。

---

## 文件说明

本节的示例文件位于 `mujoco/No_4/` 目录下：

```
mujoco/No_4/
├── no_4.py              # 最小主脚本（使用 viewer.launch_passive）
├── double_pendulum.py   # 完整交互脚本（使用 GLFW）
├── doublependulum.xml   # MuJoCo XML 模型文件
└── template_mujoco.py   # GLFW 模板（参考）
```

---

## 一、doublependulum.xml 详解（对比 No.3 的 pendulum.xml）

### No.4 doublependulum.xml 完整代码

```xml
<mujoco>
    <!--
        全局仿真选项：【No.4 新增】
        - timestep：仿真步长 0.0001s（比默认 0.002 更精细）
        - integrator：使用 RK4（4阶龙格-库塔）积分器，精度更高
        对比 No.3：No.3 使用默认积分器（Euler）， timestep 0.002
    -->
    <option timestep="0.0001" integrator="RK4">
    </option>

    <worldbody>
        <!-- 场景光源 -->
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

        <!-- 红色平面地面 -->
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

        <!--
            第一连杆 body：固定在 (0, 0, 2.5)，绕 y 轴旋转
            对比 No.3：No.3 只有单个摆杆，No.4 有两个级联的摆杆
        -->
        <body pos="0 0 2.5" euler="0 0 0">
            <!--
                hinge 关节：约束为只能绕一个轴旋转
                注意：pos="0 0 -0.5" 表示关节在连杆的下端
            -->
            <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 -0.5"/>
            <!-- 绿色圆柱几何体，半径 0.05，长度 0.5，质量 1 -->
            <geom type="cylinder" size="0.05 0.5" rgba="0 .9 0 1" mass="1"/>

            <!--
                第二连杆 body：【No.4 新增】
                嵌套在第一连杆内部，形成层级结构
                位置 pos="0 0.1 1" 相对于父 body
            -->
            <body pos="0 0.1 1" euler="0 0 0">
                <!-- 第二关节 -->
                <joint name="pin2" type="hinge" axis="0 -1 0" pos="0 0 -0.5"/>
                <!-- 蓝色圆柱几何体 -->
                <geom type="cylinder" size="0.05 0.5" rgba="0 0 0.9 1" mass="1"/>
            </body>
        </body>
    </worldbody>
    <!-- No.4 没有 actuator 和 sensor，使用 qfrc_applied 直接施加力 -->
</mujoco>
```

### No.3 pendulum.xml 完整代码（对比参考）

```xml
<mujoco>
    <option gravity="0 0 -9.81">
    </option>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

        <!-- 摆杆 body -->
        <body pos="0 0 2" euler="0 180 0">
            <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
            <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
        </body>
    </worldbody>

    <!-- actuator 驱动器 -->
    <actuator>
        <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
        <position name="position_servo" joint="pin" kp="10"/>
        <velocity name="velocity_servo" joint="pin" kv="0"/>
    </actuator>

    <!-- sensor 传感器 -->
    <sensor>
        <jointpos joint="pin" noise="0.2"/>
        <jointvel joint="pin" noise="1"/>
    </sensor>
</mujoco>
```

### XML 配置对比表

| 配置项 | No.3 (pendulum.xml) | No.4 (doublependulum.xml) |
|--------|---------------------|---------------------------|
| 关节类型 | `hinge`（1 个） | `hinge`（2 个，层级嵌套） |
| 关节名称 | `name="pin"` | `name="pin"`, `name="pin2"` |
| 几何体 | `cylinder`（单色） | `cylinder`（双色） |
| body 层级 | 单层 | 双层嵌套 |
| actuator | 3 个（motor、position、velocity） | **无（使用 qfrc_applied）** |
| sensor | 2 个（jointpos、jointvel） | **无** |
| 积分器 | 默认（Euler） | `RK4`（4阶龙格-库塔） |
| timestep | 0.002 | 0.0001 |

### 积分器类型详解

| 积分器 | 精度 | 计算成本 | 适用场景 | No.3 | No.4 |
|--------|------|----------|----------|------|------|
| Euler | 低 | 最低 | 快速预览 | ✅ | ❌ |
| RK4 | 高 | 4 倍 Euler | 高精度仿真 | ❌ | ✅ |
| implicit | 中 | 中 | 刚体系统 | - | - |

---

## 二、no_4.py 详解（最小脚本）

### 完整代码

```python
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('ball.xml')  # 注：实际加载 ball.xml
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1/500)  # ~60 Hz real-time
```

### 核心 API 对比

| API | No.3 (no_3.py) | No.4 (no_4.py) |
|-----|----------------|----------------|
| 模型加载 | `MjModel.from_xml_path()` | `MjModel.from_xml_path()` |
| 可视化 | `viewer.launch_passive()` | `viewer.launch_passive()` |
| 仿真步进 | `mj_step()` | `mj_step()` |
| 同步方式 | `viewer.sync()` | `viewer.sync()` |
| 主循环条件 | `viewer.is_running()` | `viewer.is_running()` |
| 帧率控制 | `time.sleep(1/500)` | `time.sleep(1/500)` |

**注意**：`no_4.py` 实际加载的是 `ball.xml`（单球模型），而非 `doublependulum.xml`。如需查看双摆，请使用 `double_pendulum.py`。

---

## 三、double_pendulum.py 详解（完整脚本）

### 完整代码

```python
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = 'doublependulum.xml'
simend = 50  # 仿真时间 50 秒

# 鼠标状态变量
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data):
    """
    反馈线性化控制器（Feedback Linearization）

    对比 No.3：
    - No.3 使用简单的 PD 控制：ctrl = -kp * pos_error - kv * vel_error
    - No.4 使用反馈线性化，基于系统动力学模型
    """
    # 计算位置相关能量（势能）
    mj.mj_energyPos(model, data)

    # 计算速度相关能量（动能）
    mj.mj_energyVel(model, data)

    # 将稀疏惯性矩阵 M 转换为满矩阵
    M = np.zeros((2, 2))
    mj.mj_fullM(model, M, data.qM)

    # PD 增益和参考角度
    Kp = 100 * np.eye(2)  # 比例增益
    Kd = 10 * np.eye(2)   # 微分增益
    qref = np.array([[-0.5], [-1.6]])  # 目标角度（弧度）

    # f 补偿科里奥利力和重力
    f = data.qfrc_bias[:, np.newaxis]

    # τ = M * ddqref（反馈线性化公式）
    ddqref = Kp @ (qref - data.qpos[:2][:, np.newaxis]) + \
        Kd @ (0 - data.qvel[:2][:, np.newaxis])
    tau = M @ ddqref

    # 直接施加力到系统
    data.qfrc_applied = (tau + f)[:, 0]
```

### 反馈线性化控制器详解

#### 动力学模型

双摆系统的动力学方程：

```
M(q) * qdd + C(q, qd) + g(q) = τ
```

其中：
- `M(q)`：惯性矩阵（2x2）
- `C(q, qd)`：科里奥利力和离心力
- `g(q)`：重力
- `τ`：控制力矩

#### 控制律

```python
ddqref = Kp @ (qref - q) + Kd @ (0 - qd)  # 伪加速度
tau = M @ ddqref                           # 力矩 = 惯性矩阵 × 加速度
data.qfrc_applied = tau + f                # 加上重力/科里奥利补偿
```

| 参数 | 说明 | 来源 |
|------|------|------|
| `qref` | 目标角度 `[-0.5, -1.6]` rad | 手动设置 |
| `Kp` | 位置增益 `100 * I` | 手动设置 |
| `Kd` | 速度增益 `10 * I` | 手动设置 |
| `M` | 惯性矩阵 | `mj_fullM()` 从 `data.qM` 计算 |
| `f` | 偏置力（重力+科里奥利） | `data.qfrc_bias` |

#### 与 No.3 PD 控制的对比

| 方面 | No.3 PD 控制 | No.4 反馈线性化 |
|------|-------------|----------------|
| 控制量计算 | `ctrl = -kp * e - kv * ed` | `τ = M @ (Kp*e + Kd*ed)` |
| 模型知识 | 无需 | 需要惯性矩阵 M |
| 重力补偿 | 无 | 通过 `qfrc_bias` 补偿 |
| 科里奥利补偿 | 无 | 通过 `qfrc_bias` 补偿 |
| 控制精度 | 中等 | 高（理论上精确跟踪） |

---

## 四、no_4.py 与 double_pendulum.py 对比

### 代码结构对比

| 模块 | no_4.py | double_pendulum.py |
|------|---------|-------------------|
| import | ✅ | ✅ |
| 全局变量（鼠标状态） | ❌ | ✅ |
| controller 回调 | ❌ | ✅（反馈线性化） |
| keyboard 回调 | ❌ | ✅ |
| mouse_button 回调 | ❌ | ✅ |
| mouse_move 回调 | ❌ | ✅ |
| scroll 回调 | ❌ | ✅ |
| 模型加载 | `ball.xml` | `doublependulum.xml` |
| GLFW 窗口 | ❌ | ✅ |
| 主循环 | viewer 被动模式 | GLFW 主循环 |

---

## 五、运行方法

在 `mujoco/No_4/` 目录下执行：

```bash
# 最小脚本（仅可视化，实际加载的是 ball.xml）
python no_4.py

# 完整脚本（带反馈线性化控制）
python double_pendulum.py
```

运行效果：
- `double_pendulum.py`：双摆在反馈线性化控制下稳定到目标角度 `[-0.5, -1.6]` rad

---

## 六、与 No.3 的整体对比总结

### 功能特性对比

| 特性 | No.3 (单摆) | No.4 (双摆) |
|------|-------------|-------------|
| **模型类型** | 单摆 + 地面 | 双摆 + 地面 |
| **关节数量** | 1 | 2 |
| **body 层级** | 单层 | 双层嵌套 |
| **积分器** | Euler | RK4 |
| **timestep** | 0.002 | 0.0001 |
| **驱动器** | actuator（motor、servo） | qfrc_applied |
| **传感器** | jointpos、jointvel | 无 |
| **控制方式** | PD 控制 | 反馈线性化 |
| **初始条件** | `qpos[0] = np.pi/2` | `qpos[0] = 0.1` |

### 学习路径

```
No.1: 基础建模 + viewer 可视化（被动窗口）
  ↓
No.2: GLFW 窗口 + 鼠标交互 + 回调机制 + 弹球物理
  ↓
No.3: 单摆关节控制 + 传感器读取 + PD 闭环控制
  ↓
No.4: 双摆层级结构 + RK4 积分器 + 反馈线性化控制
```

### 代码复用情况

| 代码模块 | No.3 → No.4 |
|----------|--------------|
| 鼠标状态变量 | 完全相同 |
| keyboard 回调 | 完全相同 |
| mouse_button 回调 | 完全相同 |
| mouse_move 回调 | 完全相同 |
| scroll 回调 | 完全相同 |
| GLFW 初始化 | 完全相同 |
| 可视化结构 | 完全相同 |
| 回调注册 | 完全相同 |
| 主循环 | 完全相同 |
| **controller 回调** | **完全不同（PD → 反馈线性化）** |
| **初始条件** | **完全不同（单自由度 → 双自由度）** |
| **XML 模型** | **完全不同（单摆 → 双摆）** |

---

## 七、常见问题

### 1. 双摆运动不稳定

**原因**：
- timestep 过大（RK4 需要更小的步长）
- 增益参数不合适

**解决**：No.4 已使用 `timestep="0.0001"` 和 RK4 积分器，如仍不稳定可进一步减小 timestep。

### 2. 初始位置不正确

**原因**：`qpos[0]` 只设置了第一个关节的角度。

**解决**：如需设置两个关节的初始角度：
```python
data.qpos[0] = 0.1      # 第一关节
data.qpos[1] = 0.2      # 第二关节
```

### 3. 反馈线性化控制器不工作

**原因**：`qfrc_bias` 包含重力但控制器没有正确补偿。

**解决**：
```python
data.qfrc_applied = (tau + f)[:, 0]  # 确保加上偏置力
```

### 4. `no_4.py` 显示的不是双摆

**原因**：`no_4.py` 加载的是 `ball.xml` 而非 `doublependulum.xml`。

**解决**：使用 `double_pendulum.py` 查看双摆模型。