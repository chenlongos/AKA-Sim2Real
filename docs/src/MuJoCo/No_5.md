# No.5 双摆有限状态机（FSM）轨迹跟踪

本节介绍在 No.4 双摆模型的基础上，引入**有限状态机（Finite State Machine, FSM）** 实现**多段轨迹跟踪**。系统按时间在 `HOLD → SWING1 → SWING2 → STOP` 四个状态间切换，每个阶段跟踪一条三次多项式轨迹。

---

## 文件说明

本节的示例文件位于 `mujoco/No_5/` 目录下：

```
mujoco/No_5/
├── no_5.py                    # 最小主脚本（使用 viewer.launch_passive）
├── doublependulum_fsm.py      # 完整交互脚本（使用 GLFW + FSM 控制器）
└── doublependulum_fsm.xml     # MuJoCo XML 模型文件
```

---

## 一、doublependulum_fsm.xml 详解（对比 No.4 的 doublependulum.xml）

### No.5 doublependulum_fsm.xml 完整代码

```xml
<mujoco>
    <!--
        全局仿真选项：
        - timestep：0.0001s
        - integrator：RK4
        - flag：【No.5 新增】开启 energy/contact 监测
    -->
    <option timestep="0.0001" integrator="RK4" >
        <flag energy="enable" contact="enable" />
    </option>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

        <!--
            第一连杆 body：
            - 位置 pos="0 0 1.25"（比 No.4 的 2.5 更低，留出向上摆动空间）
            - euler="0 180 0"：【No.5 新增】绕 X 轴翻转 180°，使摆杆初始朝下悬挂
        -->
        <body pos="0 0 1.25" euler="0 180 0">
            <joint name="pin" type="hinge" axis = "0 -1 0" pos="0 0 -0.5"/>
            <geom type="cylinder" size="0.05 0.5" rgba="0 .9 0 1" mass="1"/>

            <!--
                第二连杆 body：层级嵌套，结构与 No.4 相同
            -->
            <body pos="0 0.1 1" euler="0 0 0">
                <joint name="pin2" type="hinge" axis = "0 -1 0" pos="0 0 -0.5"/>
                <geom type="cylinder" size="0.05 0.5" rgba="0 0 .9 1" mass="1"/>
            </body>
        </body>
    </worldbody>

    <!--
        actuator 驱动器：【No.5 新增】
        - 两个 motor：通过 ctrl 直接施加力矩（替代 No.4 的 qfrc_applied）
        - position/velocity servo：预留接口（kp/kv=0，未启用）
    -->
    <actuator>
        <motor joint="pin"  name="torque"  gear="1" ctrllimited="true" ctrlrange="-100 100" />
        <position name="pservo1" joint="pin"  kp="0" />
        <velocity name="vservo1" joint="pin"  kv="0" />
        <motor joint="pin2" name="torque2" gear="1" ctrllimited="true" ctrlrange="-100 100" />
        <position name="pservo2" joint="pin2" kp="0" />
        <velocity name="vservo2" joint="pin2" kv="0" />
    </actuator>
</mujoco>
```

### XML 配置对比表

| 配置项 | No.4 (doublependulum.xml) | No.5 (doublependulum_fsm.xml) |
|--------|---------------------------|-------------------------------|
| 关节 | `hinge` × 2 | `hinge` × 2（相同） |
| body 层级 | 双层嵌套 | 双层嵌套（相同） |
| 初始 body 位置 | `(0, 0, 2.5)` | `(0, 0, 1.25)`（更低） |
| 初始 body 朝向 | `euler="0 0 0"` | `euler="0 180 0"`（**新增**，翻转朝下） |
| 积分器 | RK4 | RK4（相同） |
| timestep | 0.0001 | 0.0001（相同） |
| `<flag>` 元素 | 无 | **新增** `energy` / `contact` |
| actuator | **无** | **新增** 2× motor + 4× servo |
| 驱动方式 | `qfrc_applied` | `data.ctrl`（通过 actuator） |

### 关键变更说明

#### 1. 高度降低 + 朝向翻转
No.5 把第一连杆固定点从 z=2.5 降到 z=1.25，并加上 `euler="0 180 0"`，让摆杆**初始朝下悬挂**。这是因为 FSM 任务是要让双摆**向上甩到顶部**（类似杂技「手倒立」），需要低悬挂点 + 朝下初始位形。

#### 2. 引入 actuator
No.4 通过 `data.qfrc_applied` 直接施加广义力，**不经过 actuator**。No.5 改用 actuator 中的 motor + `data.ctrl` 通道，这是更标准的工业控制接口。

#### 3. actuator 通道映射

| actuator 名 | 关节 | ctrl 索引 | 用途 |
|------------|------|----------|------|
| `torque` | pin | `ctrl[0]` | 第一个关节的力矩输入 |
| `torque2` | pin2 | `ctrl[3]` | 第二个关节的力矩输入 |
| `pservo1` / `vservo1` | pin | ctrl[1] / ctrl[2] | 预留（kp=0, kv=0，未启用） |
| `pservo2` / `vservo2` | pin2 | ctrl[4] / ctrl[5] | 预留（kp=0, kv=0，未启用） |

> 注意：motor 和 servo 共享同一关节的 ctrl 索引空间，所以 `pin` 的 motor 占 `ctrl[0]`，servo 占 `ctrl[1]/[2]`；`pin2` 的 motor 占 `ctrl[3]`，servo 占 `ctrl[4]/[5]`。控制器中需要先清空所有 6 个 ctrl 通道再赋值。

---

## 二、no_5.py 详解（最小脚本）

### 完整代码

```python
import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('doublependulum_fsm.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1e-3)
```

### 与 no_4.py 的核心差异

| 项 | no_4.py | no_5.py |
|----|---------|---------|
| `import time` 位置 | 放在 `import mujoco` 之后 | 放在**最顶部**（PEP 8 标准） |
| 加载的 XML | `ball.xml`（**bug**，本意是双摆） | `doublependulum_fsm.xml`（**修正**） |
| 帧率控制 | `time.sleep(1/500)` ≈ 2ms | `time.sleep(1e-3)` = 1ms |
| controller | ❌ | ❌（仅观察模型，无控制） |
| 初始条件 | 未设置 | 未设置（XML 默认下垂） |

> 关键改进：`no_4.py` 加载的是 `ball.xml`（一个**误用**），而 `no_5.py` 正确加载了双摆模型。最小脚本只验证「模型能加载、能跑」，**不展示 FSM 控制效果**——要看 FSM 必须运行 `doublependulum_fsm.py`。

---

## 三、doublependulum_fsm.py 详解（完整脚本）

### 完整代码

```python
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = 'doublependulum_fsm.xml'
simend = 5

# 时序参数（单位：秒）
t_hold   = 0.5   # 起始保持阶段
t_swing1 = 1.0   # 第一段上摆
t_swing2 = 1.0   # 第二段上摆

# FSM 状态枚举
FSM_HOLD   = 0
FSM_SWING1 = 1
FSM_SWING2 = 2
FSM_STOP   = 3

# 三段轨迹的目标点
q_init = np.array([[-1.0], [0.0]])    # 初始下垂
q_mid  = np.array([[ 0.5], [-2.0]])   # 中间姿态
q_end  = np.array([[ 1.0], [0.0]])    # 末端顶部倒立

t_init = t_hold
t_mid  = t_hold + t_swing1
t_end  = t_hold + t_swing1 + t_swing2

def init_controller(model, data):
    """初始化 FSM 状态，并预生成两段三次多项式轨迹。"""
    global fsm_state, a_swing1, a_swing2
    fsm_state = FSM_HOLD
    a_swing1 = generate_trajectory(t_init, t_mid, q_init, q_mid)
    a_swing2 = generate_trajectory(t_mid,  t_end, q_mid,  q_end)

def controller(model, data):
    """FSM + PD 轨迹跟踪控制器。"""
    global fsm_state, a_swing1, a_swing2
    time = data.time

    # 状态机转移（仅基于时间）
    if fsm_state == FSM_HOLD and time >= t_hold:
        fsm_state = FSM_SWING1
    elif fsm_state == FSM_SWING1 and time >= t_mid:
        fsm_state = FSM_SWING2
    elif fsm_state == FSM_SWING2 and time >= t_end:
        fsm_state = FSM_STOP

    # 各状态下的参考轨迹
    if fsm_state == FSM_HOLD:
        q_ref  = q_init
        dq_ref = np.zeros((2, 1))
    elif fsm_state == FSM_SWING1:
        a = a_swing1
        q_ref  = a[0] + a[1]*time + a[2]*time**2 + a[3]*time**3
        dq_ref = a[1] + 2*a[2]*time + 3*a[3]*time**2
    elif fsm_state == FSM_SWING2:
        a = a_swing2
        q_ref  = a[0] + a[1]*time + a[2]*time**2 + a[3]*time**3
        dq_ref = a[1] + 2*a[2]*time + 3*a[3]*time**2
    elif fsm_state == FSM_STOP:
        q_ref  = q_end
        dq_ref = np.zeros((2, 1))

    # PD 控制（增益比 No.4 大 5 倍）
    kp, kv = 500, 50
    torque = kp * (q_ref[:, 0] - data.qpos) + kv * (dq_ref[:, 0] - data.qvel)

    # 清空所有 6 个 ctrl 通道（motor+servo 共享索引空间）
    for i in range(6):
        data.ctrl[i] = 0
    # 写入两个 motor 通道
    data.ctrl[0] = torque[0]   # pin  → torque
    data.ctrl[3] = torque[1]   # pin2 → torque2

def generate_trajectory(t0, tf, q0, qf):
    """
    三次多项式轨迹：q(t) = a0 + a1*t + a2*t^2 + a3*t^3
    满足边界条件：q(t0)=q0, q(tf)=qf, dq(t0)=0, dq(tf)=0
    """
    tf_t0_3 = (tf - t0)**3
    a0 = (qf*(t0**2)*(3*tf - t0) + q0*(tf**2)*(tf - 3*t0)) / tf_t0_3
    a1 = (6*t0*tf*(q0 - qf)) / tf_t0_3
    a2 = (3*(t0 + tf)*(qf - q0)) / tf_t0_3
    a3 = (2*(q0 - qf)) / tf_t0_3
    return a0, a1, a2, a3
```

### 3.1 有限状态机（FSM）架构

```
时间 ─────────────────────────────────────────────────▶
  │                                                      
  ├─ t ∈ [0, 0.5)        ├─ [0.5, 1.5)  ├─ [1.5, 2.5)  ├─ [2.5, ∞)
  │   HOLD               │   SWING1     │   SWING2     │   STOP
  │   q_ref = q_init     │   三次曲线   │   三次曲线   │   q_ref = q_end
  │   dq_ref = 0         │   init→mid   │   mid→end    │   dq_ref = 0
```

**状态转移条件**：纯粹基于仿真时间 `data.time`，是一种**时间驱动的确定性 FSM**。

| 状态 | 时长 | 目标位姿 | 控制目标 |
|------|------|----------|----------|
| `HOLD` | 0.5s | `q_init = [-1, 0]` rad | 保持下垂，等待起摆 |
| `SWING1` | 1.0s | 沿三次曲线扫过 `q_init → q_mid` | 第一段上摆 |
| `SWING2` | 1.0s | 沿三次曲线扫过 `q_mid → q_end` | 第二段上摆，到达顶部 |
| `STOP` | 永久 | `q_end = [1, 0]` rad | 保持顶部倒立姿态 |

### 3.2 三次多项式轨迹生成

目标：在 `t0` 时刻处于 `q0`、在 `tf` 时刻处于 `qf`，且**起止速度均为 0**。这是最小jerk风格的边界条件。

求解后得到 4 个系数 `a0, a1, a2, a3`（形状 `(2, 1)`，对应 2 个关节）：

```
q(t)   = a0 + a1·t + a2·t² + a3·t³
dq(t)  = a1 + 2·a2·t + 3·a3·t²
```

`generate_trajectory()` 在 `init_controller` 中**预先**计算好两段轨迹的系数，运行时只做多项式求值，避免每步反解线性方程组。

### 3.3 PD 控制器（与 No.4 对比）

| 控制器特性 | No.4 (反馈线性化) | No.5 (PD + FSM) |
|------------|------------------|----------------|
| 控制律 | `τ = M·ddqref + f` | `τ = kp·e + kv·ed` |
| 增益 `kp` | `100·I` | **500**（5 倍） |
| 增益 `kv` | `10·I` | **50**（5 倍） |
| 需惯性矩阵 M | ✅ | ❌ |
| 需补偿重力/科氏 | ✅（用 `qfrc_bias`） | ❌（靠大 kp/kv 隐式抑制） |
| 驱动接口 | `data.qfrc_applied` | `data.ctrl`（actuator） |
| 跟踪目标 | **固定位姿** `qref` | **时变轨迹** `q(t), dq(t)` |

> 设计权衡：No.4 用模型做精确补偿 + 较小增益；No.5 用高增益 PD 暴力跟踪，不依赖模型知识，**更鲁棒于参数误差**但控制信号更"硬"。

### 3.4 ctrl 通道的写入

```python
for i in range(6):
    data.ctrl[i] = 0          # 清空（含预留的 servo 通道）
data.ctrl[0] = torque[0]      # pin 的 motor
data.ctrl[3] = torque[1]      # pin2 的 motor
```

由于 `pin` 的 motor 占 `ctrl[0]`，而 `pin` 的 position/velocity servo 占 `ctrl[1]/[2]`，所以必须**全部清零**再写 motor，否则 servo 通道的 kp=0 不会注入任何力，但保险起见仍清零。

---

## 四、no_5.py 与 doublependulum_fsm.py 对比

| 模块 | no_5.py | doublependulum_fsm.py |
|------|---------|----------------------|
| import 风格 | `import time` 在最顶 | 同 |
| 模型加载 | `doublependulum_fsm.xml` ✅ | 同 |
| `init_controller` | ❌ | ✅（预生成轨迹） |
| FSM 状态变量 | ❌ | ✅ |
| controller 回调 | ❌ | ✅（FSM + PD） |
| 轨迹生成器 | ❌ | ✅（三次多项式） |
| keyboard 回调 | ❌ | ✅（Backspace 重置） |
| mouse_button 回调 | ❌ | ✅ |
| mouse_move 回调 | ❌ | ✅ |
| scroll 回调 | ❌ | ✅ |
| GLFW 窗口 | ❌ | ✅（1200×900） |
| 仿真时长 | 无限（手动关闭） | 5 秒（`simend=5`） |
| 帧率控制 | `time.sleep(1e-3)` | 内层 `1.0/60.0` 步进 + `glfw.swap_interval(1)` |

> 注意：`doublependulum_fsm.py` 没有 `no_5.py` 中的 `time.sleep(1e-3)`，因为它用 GLFW 的 `swap_interval(1)`（v-sync）来限速在 60Hz。

---

## 五、运行方法

在 `mujoco/No_5/` 目录下执行：

```bash
# 最小脚本（仅可视化双摆自由下落，没有 FSM 控制）
mjpython no_5.py

# 完整脚本（FSM + 轨迹跟踪，把双摆甩到顶部）
mjpython doublependulum_fsm.py
```

> macOS 上必须用 `mjpython` 启动（含 MJPEG 编码器）；Linux/Windows 可用普通 `python`。

预期效果：
- `no_5.py`：双摆从下垂自然摆动，没有控制输入。
- `doublependulum_fsm.py`：
  - `t ∈ [0, 0.5)`：保持下垂。
  - `t ∈ [0.5, 1.5)`：第一关节大幅正向加速，把第二关节甩起。
  - `t ∈ [1.5, 2.5)`：第二关节继续被驱动到顶部。
  - `t > 2.5`：保持在 `q_end = [1, 0]` rad（顶部倒立）。

---

## 六、与 No.4 的整体对比总结

### 功能特性对比

| 特性 | No.4 (双摆 + 反馈线性化) | No.5 (双摆 + FSM) |
|------|--------------------------|-------------------|
| **模型** | 双摆，下垂初始 | 双摆，**下垂 + 翻转 180°** |
| **驱动接口** | `qfrc_applied` | `data.ctrl`（actuator） |
| **actuator** | 无 | 2× motor + 4× servo |
| **控制目标** | 单点镇定 | **多段轨迹跟踪** |
| **控制器** | 反馈线性化 | PD |
| **状态机** | ❌ | ✅（4 状态） |
| **轨迹生成** | ❌ | 三次多项式 |
| **模型知识需求** | 需 M(q)、qfrc_bias | 无（高增益 PD） |
| **增益** | kp=100, kv=10 | kp=500, kv=50 |
| **仿真时长** | 50s | 5s |
| **最小脚本** | 加载 `ball.xml`（bug） | 加载正确 XML（修复） |

### 控制思想对比

```
No.4 的「反馈线性化」思维：
   已知 M(q), g(q) → 算出补偿项 → 用 M⁻¹ 精确跟踪参考加速度
   优点：理论精度高、增益需求小
   缺点：依赖模型精度，参数不准时性能下降

No.5 的「FSM + 高增益 PD」思维：
   不知道精确模型 → 用大 kp/kv 暴力跟踪
   优点：鲁棒于参数误差、结构清晰、易扩展多阶段
   缺点：控制信号"硬"、可能激发未建模动力学
```

### 学习路径

```
No.1: 基础建模 + viewer 可视化（被动窗口）
  ↓
No.2: GLFW 窗口 + 鼠标交互 + 回调机制
  ↓
No.3: 单摆关节控制 + 传感器读取 + PD 闭环控制
  ↓
No.4: 双摆层级结构 + RK4 + 反馈线性化（单点镇定）
  ↓
No.5: 双摆 + actuator + FSM + 三次多项式轨迹（多段跟踪）  ← 当前
  ↓
（未来）No.6: 接触 / 抓取 / 强化学习策略
```

### 代码复用情况

| 代码模块 | No.4 → No.5 |
|----------|-------------|
| 鼠标状态变量 | 完全相同 |
| keyboard / mouse / scroll 回调 | 完全相同 |
| GLFW 初始化 / 可视化结构 | 完全相同 |
| 关节定义（pin、pin2） | 完全相同 |
| 几何体 | 完全相同 |
| 积分器 / timestep | 完全相同 |
| **驱动接口** | **完全不同**（`qfrc_applied` → `data.ctrl`） |
| **控制器** | **完全不同**（反馈线性化 → FSM + PD） |
| **轨迹生成** | **No.5 新增** |
| **状态机** | **No.5 新增** |
| **XML 初始位姿** | **完全不同**（无翻转 → 翻转 180°） |
| **actuator 配置** | **No.5 新增** |

---

## 七、常见问题

### 1. XML 报错 `unrecognized attribute: 'sensornoise'`

**原因**：`<flag sensornoise="enable" ...>` 是老版本 MuJoCo 的写法，新版已移除该属性。

**解决**：删掉 `sensornoise="enable"`，噪声在 `<sensor>` 元素中单独配置。

### 2. `from_xml_path` 报 `ValueError: XML Error`

**原因**：常见两种——
- 传了 `.py` 文件（误用）
- XML 里有 schema 不识别的属性

**解决**：确认路径是 `.xml`，并检查 XML 中所有属性是否在 [MuJoCo XML 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html) 中。

### 3. FSM 不切换状态

**原因**：`init_controller` 未在 `set_mjcb_control` **之前**调用，导致 `fsm_state` 未初始化。

**解决**：
```python
init_controller(model, data)  # 必须在 set_mjcb_control 之前
mj.set_mjcb_control(controller)
```

### 4. 摆杆抖动剧烈

**原因**：
- timestep 不够小
- kp/kv 过大激发数值不稳定
- actuator 的 ctrlrange 不够大

**解决**：
- 减小 `timestep`（当前 0.0001 已较小）
- 适度降低 kp/kv（先各降 50% 试试）
- 确认 `ctrlrange` 足够大（当前 ±100）

### 5. `data.ctrl[i]` 写入但摆杆没反应

**原因**：写入了非 motor 通道（如 `ctrl[1]` 是 position servo），而 servo 的 kp=0 不产生力。

**解决**：只写 `ctrl[0]`（pin motor）和 `ctrl[3]`（pin2 motor），其余清零。

### 6. `no_5.py` 看不出 FSM 效果

**原因**：`no_5.py` 是最小脚本，**没有注册 controller**，所以双摆自由下落。

**解决**：要观察 FSM 控制效果，必须运行 `mjpython doublependulum_fsm.py`。