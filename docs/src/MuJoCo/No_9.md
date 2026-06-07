# No.9 单腿跳跃机器人（Hopper）4 状态 FSM

本节介绍一个**单腿跳跃机器人**控制：4 状态 FSM 模拟完整的「空中 → 落地 → 压缩 → 蹬地 → 再次腾空」跳跃循环，**运行时切换 PD 增益**实现不同的刚度/阻尼需求。这是 Raibert hopper 风格的经典控制范式，也是强化学习 locomotion 任务的 baseline controller。

---

## 文件说明

```
mujoco/No_9/
├── hopper.xml     # MuJoCo XML 模型文件
└── hopper.py      # 完整脚本：4 状态 FSM + 动态增益切换
```

> No.9 **没有**最小脚本（`no_9.py`）。要看效果必须跑 `hopper.py`。

---

## 一、hopper.xml 详解

```xml
<mujoco>
    <!-- 【No.9 新增】视觉配置：启用头灯 -->
    <visual>
        <headlight ambient="0.5 0.5 0.5"/>
    </visual>

    <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81">
        <flag contact="enable" energy="enable"/>
    </option>

    <worldbody>
        <!-- 地面（很长 100） -->
        <geom type="plane" size="100 1 0.1" rgba=".9 0 0 1"/>

        <!--
            躯干 torso：sphere 形状（mass=1）
            2 个 slide 关节：x（前进）、z（上下）
        -->
        <body name="torso" pos="0 0 2">
            <joint name="x" type="slide" pos="0 0 0" axis="1 0 0" />
            <joint name="z" type="slide" pos="0 0 0" axis="0 0 1" />
            <geom type="sphere" size="0.1" rgba=".9 .9 .9 1" mass="1"/>

            <!--
                大腿 leg：cylinder（mass=1）
                1 个 hinge 关节：hip（髋关节，绕 y 旋转）
            -->
            <body name="leg" pos="0 0 -0.5" euler="0 0 0">
                <joint name="hip" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
                <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>

                <!--
                    脚 foot：cylinder（mass=0，视觉）+ sphere（mass=0.1，碰撞）
                    1 个 slide 关节：knee（膝关节，垂直伸缩）
                -->
                <body name="foot" pos="0 0 -0.75">
                    <joint name="knee" type="slide" pos="0 0 0.25" axis="0 0 -1" />
                    <geom type="cylinder" pos="0 0 0.125" size=".01 .125" rgba="0 0 .9 1" mass="0"/>
                    <geom type="sphere"  size="0.05"            rgba=".9 .9 0 1"  mass="0.1"/>
                </body>
            </body>
        </body>
    </worldbody>

    <!--
        【No.9 关键】4 个 actuator 通道：每个关节都有 position + velocity servo
        初始 kp/kv=0，由 init_controller 和 controller 在运行时设置
    -->
    <actuator>
        <position name="pservo-hip"  joint="hip"  kp="0"/>
        <velocity name="vservo-hip"  joint="hip"  kv="0"/>
        <position name="pservo-knee" joint="knee" kp="0"/>
        <velocity name="vservo-knee" joint="knee" kv="0"/>
    </actuator>
</mujoco>
```

### 关键设计说明

#### 1. 机器人形态（4 DOF）

| DOF 索引 | 关节名 | 类型 | 物理含义 | 范围 |
|---------|--------|------|---------|------|
| 0 | `x`  | slide | 躯干前后平移 | 无界 |
| 1 | `z`  | slide | 躯干上下平移 | 无界 |
| 2 | `hip`| hinge | 髋关节旋转 | ±π |
| 3 | `knee`| slide | 膝关节伸缩 | 无界 |

**特点**：**没有「前进/后退」的主动驱动** —— 跳跃产生的反作用力是**唯一**的水平推力来源。

#### 2. 4 个 actuator 通道：每个关节 position + velocity servo

```xml
<position name="pservo-hip"  joint="hip"  kp="0"/>   <!-- ctrl[0] -->
<velocity name="vservo-hip"  joint="hip"  kv="0"/>   <!-- ctrl[1] -->
<position name="pservo-knee" joint="knee" kp="0"/>   <!-- ctrl[2] -->
<velocity name="vservo-knee" joint="knee" kv="0"/>   <!-- ctrl[3] -->
```

每个关节有**两个 actuator**（一个 P、一个 D），它们**叠加**出力：

```
F_total = F_position_servo + F_velocity_servo
        = kp · (ctrl[0] - q) + kv · (ctrl[1] - qd)
```

> **技巧**：通过运行时修改 kp/kv，可以**同一个 actuator 在不同状态下表现得像弹簧、阻尼器、刚性关节**。

#### 3. `<visual><headlight>`

```xml
<visual>
    <headlight ambient="0.5 0.5 0.5"/>
</visual>
```

启用**头灯**（跟随相机移动的虚拟光源），让画面有立体感。`ambient` 是环境光强度。

#### 4. foot 的双 geom 设计

```xml
<geom type="cylinder" ... mass="0"/>     <!-- 视觉圆柱 -->
<geom type="sphere"  ... mass="0.1"/>    <!-- 物理碰撞球 -->
```

一个轻量球（mass=0.1）做**接触碰撞**，一个零质量圆柱做**视觉装饰**。这是机器人模型常用的「**质量集中**」技巧。

---

## 二、核心：4 状态 FSM（跳跃循环）

### 2.1 状态定义

```python
FSM_AIR1   = 0   # 空中，下落中
FSM_STANCE1= 1   # 刚落地，压缩中
FSM_STANCE2= 2   # 蹬地，向上加速
FSM_AIR2   = 3   # 离地后再腾空
```

### 2.2 状态转移图

```
                            脚高度 > 0.05
   ┌────────────────────────────────────────┐
   │                                        │
   ▼                                        │
┌────────┐  脚高<0.05    ┌────────┐  vz>0   ┌────────┐
│ AIR1   │ ──────────────▶│ STANCE1│ ───────▶│ STANCE2│
│  下落   │                │  压缩   │         │  蹬地   │
└────────┘                └────────┘         └────────┘
   ▲                                            │
   │              vz<0                          │
   │  ┌──────────────────────────┐              │
   │  │                                          │
   │  │  脚高>0.05  ▼                             │
   │ ┌────────┐                                  │
   │ │ AIR2   │                                  │
   │ │  腾空   │ ─────────────────────────────────┘
   │ └────────┘
```

### 2.3 状态转移条件详解

```python
body_no = 3
z_foot = data.xpos[body_no, 2]     # 脚的世界坐标 z
vz_torso = data.qvel[1]            # 躯干 z 方向速度

# AIR1 → STANCE1：脚触地
if fsm == FSM_AIR1 and z_foot < 0.05:
    fsm = FSM_STANCE1

# STANCE1 → STANCE2：躯干开始上升（压缩反弹）
if fsm == FSM_STANCE1 and vz_torso > 0.0:
    fsm = FSM_STANCE2

# STANCE2 → AIR2：脚离开地面
if fsm == FSM_STANCE2 and z_foot > 0.05:
    fsm = FSM_AIR2

# AIR2 → AIR1：躯干开始下落（完成一个跳跃周期）
if fsm == FSM_AIR2 and vz_torso < 0.0:
    fsm = FSM_AIR1
    step_no += 1
```

| 转移 | 触发条件 | 物理含义 |
|------|---------|---------|
| AIR1 → STANCE1 | `z_foot < 0.05` | 脚接触地面（脚 z 坐标小于 5cm） |
| STANCE1 → STANCE2 | `vz_torso > 0.0` | 躯干开始反弹上升 |
| STANCE2 → AIR2 | `z_foot > 0.05` | 脚蹬离地面 |
| AIR2 → AIR1 | `vz_torso < 0.0` | 到达最高点，开始下落 |

> **这是事件驱动 FSM**（跟 No.5 的时间驱动、No.8 的单事件触发都不同）—— 转移条件是**物理量**而不是时间或单一标志位。

### 2.4 body_no = 3 是怎么定的？

MuJoCo 按 XML 声明顺序给 body 编号：

| 索引 | body | 来源 |
|------|------|------|
| 0 | worldbody | 隐含 |
| 1 | torso | XML 第一个 `<body>` |
| 2 | leg | torso 内嵌的 `<body>` |
| 3 | foot | leg 内嵌的 `<body>` |

所以 `body_no = 3` 是 foot。**这是个脆弱的硬编码**，XML 一改就错。

---

## 三、动态增益切换：每个状态不同刚度/阻尼

### 3.1 状态-增益对照表

| 状态 | pservo-hip kp | vservo-hip kv | pservo-knee kp | vservo-knee kv | ctrl[0] |
|------|-------------|---------------|----------------|----------------|---------|
| **AIR1** | 100 | 10 | 100 | 10 | 0 |
| **STANCE1** | 1000 | 0 | 1000 | 0 | 0 |
| **STANCE2** | 1000 | 0 | 1000 | 0 | **-0.2** |
| **AIR2** | 100 | 10 | 100 | 10 | 0 |

**规律**：
- **空中**（AIR1, AIR2）：kp=100, kv=10 —— **软弹簧 + 阻尼**，落地不冲击
- **着地**（STANCE1, STANCE2）：kp=1000, kv=0 —— **硬弹簧、无阻尼**，存储弹性势能
- **蹬地**（STANCE2）：髋关节目标设为 -0.2 rad —— **腿向后摆**，把身体「弹」出去

### 3.2 增益函数

```python
def set_position_servo(actuator_no, kp):
    model.actuator_gainprm[actuator_no, 0] = kp
    model.actuator_biasprm[actuator_no, 1] = -kp

def set_velocity_servo(actuator_no, kv):
    model.actuator_gainprm[actuator_no, 0] = kv
    model.actuator_biasprm[actuator_no, 2] = -kv
```

### 3.3 运行时改增益的机制

`model.actuator_gainprm` 和 `model.actuator_biasprm` 是**模型参数**，但 MuJoCo **允许运行时修改**。

MuJoCo 通用 actuator 模型：

```
actuator_force = gainprm[0] · ctrl + biasprm[0]
               + biasprm[1] · (actuated_quantity)         ← 位置反馈
               + biasprm[2] · (actuated_velocity)         ← 速度反馈
               + biasprm[3] · ctrl²                       ← 二阶项（一般不用）
```

对于**位置伺服**（`actuator_gainprm[0] = kp, biasprm[1] = -kp`）：

```
F = kp · ctrl + (-kp) · q
  = kp · (ctrl - q)
```

对于**速度伺服**（`actuator_gainprm[0] = kv, biasprm[2] = -kv`）：

```
F = kv · ctrl + (-kv) · qd
  = kv · (ctrl - qd)
```

> **关键设计**：`gainprm[0]` 既可能是 P 也可能是 D 的系数，靠 `biasprm` 的索引区分作用位置。这是个**非常紧凑的通用模型**。

### 3.4 为什么 STANCE 阶段 kv=0？

蹬地时**不**要阻尼 —— 你想让弹簧**完全弹性**地释放能量。如果有阻尼，弹性势能会**被消耗**而不是**全部转化为动能**。

| 阶段 | kp | kv | 行为 |
|------|-----|-----|------|
| 空中 | 100 | 10 | 软着地、摆动平稳 |
| 落地 | 1000 | 0 | **储能**（弹簧压缩） |
| 蹬地 | 1000 | 0 | **释能**（弹簧反弹） |

**没有阻尼**意味着「**完全弹性碰撞**」—— 落到地面的动能全部存进弹簧，弹起时全部释放。这就是 Raibert hopper 的精髓。

---

## 四、视觉：headlight + 跟随相机

### 4.1 headlight

```xml
<visual>
    <headlight ambient="0.5 0.5 0.5"/>
</visual>
```

注释掉了两个 `<light>` 元素。headlight 是**相机跟随的虚拟光源**，自然地随相机移动，照亮场景。

### 4.2 跟随相机

```python
cam.lookat[0] = data.qpos[0]   # 相机 lookat 的 x 跟随躯干 x
```

```python
# 在主循环里（注意：原始代码里是在 mj_step 之后）
cam.lookat[0] = data.qpos[0]
mj.mjv_updateScene(model, data, opt, None, cam, ...)
```

**效果**：相机**永远盯着躯干**，hopper 跳到哪，相机就看到哪。

> **小 bug**：`cam.lookat` 是 numpy 数组，但 `cam.lookat[0] = data.qpos[0]` 这种**属性赋值**在某些 MuJoCo 版本里**不会真正更新到 MjrRect**。可能需要 `cam.lookat = np.array([data.qpos[0], 0, 1.5])` 重新赋值。

---

## 五、init_controller / controller 分工

### 5.1 init_controller（启动时调一次）

```python
def init_controller(model, data):
    set_position_servo(0, 100)   # pservo-hip
    set_velocity_servo(1, 10)    # vservo-hip
    set_position_servo(2, 1000)  # pservo-knee
    set_velocity_servo(3, 0)     # vservo-knee
```

**默认值**：髋软、膝硬。然后 controller 会根据状态重新设。

### 5.2 controller（每物理步调一次）

```python
if fsm == FSM_AIR1:
    set_position_servo(2, 100)   # 膝软
    set_velocity_servo(3, 10)    # 膝阻尼

if fsm == FSM_STANCE1:
    set_position_servo(2, 1000)  # 膝硬
    set_velocity_servo(3, 0)     # 膝无阻尼

if fsm == FSM_STANCE2:
    set_position_servo(2, 1000)  # 膝硬
    set_velocity_servo(3, 0)     # 膝无阻尼
    data.ctrl[0] = -0.2          # 髋目标 -0.2 rad（腿后摆）

if fsm == FSM_AIR2:
    set_position_servo(2, 100)   # 膝软
    set_velocity_servo(3, 10)    # 膝阻尼
    data.ctrl[0] = 0.0           # 髋目标归零
```

**只有 pservo-hip（ctrl[0]）在 STANCE2 期间被显式设目标** —— 用来在蹬地瞬间**把腿向后甩**。其他时候 ctrl[0]=0，意思是「**腿保持竖直**」。

---

## 六、跟 No.5/No.8 FSM 的对比

| 维度 | No.5 FSM | No.8 FSM | **No.9 FSM** |
|------|---------|---------|--------------|
| 状态数 | 4 | 2 | **4** |
| 触发条件 | 时间 | 单一物理量（`qpos[5] > 1.0`）| **多个物理量**（z_foot, vz_torso）|
| 触发类型 | 时间驱动 | 事件驱动 | **多事件驱动** |
| 切换内容 | 切 PD 参考 | 切 `qfrc_applied` | **切 kp/kv + 切 ctrl[0]** |
| 是否运行时改模型 | ❌ | ❌ | **✅ 改 `actuator_gainprm`** |
| 应用域 | 关节空间 | 约束力 | **完整运动周期** |

### 控制思想对比

```
No.5: 时间表 → 切任务（HOLD / SWING1 / SWING2 / STOP）
No.8: 单事件 → 切物理交互（SWING / FREE）
No.9: 多事件 + 动态增益 → 切运动阶段（AIR / STANCE / 蹬地）
```

**No.9 是第一个把「**运行时调整模型参数**」作为控制手段的例**。

---

## 七、整体控制流程图

```
启动:  init_controller(model, data)
  ├─ 设默认增益: pservo-hip kp=100, vservo-hip kv=10
  ├─ 设默认增益: pservo-knee kp=1000, vservo-knee kv=0
  └─ mj.set_mjcb_control(controller)

主循环 (60Hz) ─────────────────────────────────────
  内层 1000Hz: mj_step → 每步自动调 controller():
  │
  │   读 z_foot = data.xpos[3, 2]
  │   读 vz_torso = data.qvel[1]
  │
  │   ┌─ 状态转移（多条件检查）─┐
  │   │  AIR1  + z_foot<0.05  → STANCE1
  │   │  STANCE1 + vz>0       → STANCE2
  │   │  STANCE2 + z_foot>0.05 → AIR2
  │   │  AIR2  + vz<0         → AIR1
  │   └────────────────────────┘
  │
  │   ┌─ 状态-增益映射 ─┐
  │   │  AIR*:   kp=100,  kv=10    (软着地)
  │   │  STANCE: kp=1000, kv=0     (储能/释能)
  │   │  STANCE2: ctrl[0] = -0.2   (腿后摆)
  │   └────────────────┘
  │
  │   set_position_servo(2, kp)   ← 改 model 参数
  │   set_velocity_servo(3, kv)   ← 改 model 参数
  │   data.ctrl[0] = target       ← 改控制目标
  │
  └─ 物理: mj_step 应用所有力，更新 qpos, qvel
  外层: 渲染 + cam.lookat[0] = data.qpos[0]  (跟随)
```

---

## 八、运行方法

```bash
cd mujoco/No_9/
mjpython hopper.py
```

预期效果：
- Hopper **原地（或缓慢前进）跳跃**
- 大约 0.5-1 秒一跳，`step_no` 累计
- `simend = 20` 秒应该看到 **15-25 跳**
- 相机**自动跟随**（`lookat[0] = qpos[0]`）

---

## 九、调试 / 验证方法

### 1. 打印状态和步数

```python
def controller(model, data):
    global fsm, step_no
    body_no = 3
    z_foot = data.xpos[body_no, 2]
    vz_torso = data.qvel[1]
    print(f"t={data.time:.2f}  fsm={fsm}  z_foot={z_foot:.3f}  vz={vz_torso:.2f}  step={step_no}")
    # ... 原有代码
```

### 2. 验证增益确实被改了

```python
def controller(model, data):
    print(f"hip kp={model.actuator_gainprm[0, 0]:.0f}  "
          f"hip kv={model.actuator_gainprm[1, 0]:.0f}  "
          f"knee kp={model.actuator_gainprm[2, 0]:.0f}  "
          f"knee kv={model.actuator_gainprm[3, 0]:.0f}")
```

### 3. 调参方向

| 想改 | 改什么 |
|------|--------|
| 跳得更高 | 增大 `pservo-knee` 的 kp，或延长 STANCE 阶段 |
| 跳得更稳 | 增大 `vservo-knee` 在 AIR 阶段的 kv |
| 跳得更远 | STANCE2 时设 `data.ctrl[0] = -0.5`（腿更向后摆）|
| 跳得更快 | 把状态转移阈值 `0.05` 改小（更快检测着地/离地）|

---

## 十、常见问题

### 1. Hopper 跳不起来

**原因**：STANCE 阶段 kp 太小，没有储能。

**解决**：
- 增大 `pservo-knee` 在 STANCE 的 kp（从 1000 试到 2000）
- 检查 `vz_torso` 是不是真的能 > 0
- 试着增加躯干质量（gravity 改小也行）

### 2. Hopper 触地后「粘在地上」

**原因**：蹬地力度不够，弹不起来。

**解决**：
- 检查 STANCE2 状态有没有真的进入（看 `z_foot > 0.05`）
- `data.ctrl[0] = -0.2` 改大（比如 -0.5）
- vservo 在 STANCE 阶段必须 `kv=0`，不然会**消耗弹性势能**

### 3. `body_no = 3` 报错 / 不对

**原因**：XML 改了，body 顺序变了。

**解决**：用名字查 id 而不是硬编码：
```python
foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot")
z_foot = data.xpos[foot_id, 2]
```

### 4. `data.xpos[body_no, 2]` 是脚的哪点？

是 foot body **质心**的 z 坐标（不是脚底）。脚底可能比质心低 0.05m 左右，所以阈值 `0.05` 实际上对应**脚底刚刚触地**。

### 5. `cam.lookat[0] = data.qpos[0]` 没生效

**原因**：某些 MuJoCo 版本对 `MjvCamera.lookat` 的**元素赋值**不更新底层 C 结构。

**解决**：
```python
cam.lookat = np.array([data.qpos[0], 0.0, 1.5])  # 整体赋值
```

### 6. 运行时改 `actuator_gainprm` 会有性能影响吗？

**没有**。这是直接修改内存里的 float 值，下一步 `mj_step` 就用新值。**对仿真速度零影响**。

但有**限制**：必须**每步**改才能持续生效（虽然 `model` 是持久的，但如果你重置 `data` 会保留 `model` 的修改）。

### 7. 为什么没有 LQR、IK、反馈线性化这些高级控制？

**因为跳跃是高度非线性 + 不连续的**（着地瞬间）。这些方法都基于「**线性化点附近**」假设，跳跃时**严重偏离**任何平衡点。

FSM + 动态增益是**事件驱动**的简单方案，**鲁棒**且**可解释**，是经典 Raibert hopper 的做法。

### 8. 跟强化学习里的 locomotion 任务什么关系？

这是**经典控制**（model-based FSM），RL 的 PPO/SAC 是**学习控制**（model-free）。RL baseline 通常也是 4 状态 FSM + 简单 PD，但**学习**状态转移的精确时机和增益。

---

## 十一、整体公式对应

```
─────── 机器人形态 ───────
torso (sphere, mass=1) + leg (cylinder, mass=1) + foot (sphere, mass=0.1)
nv = 4 (x, z, hip, knee)
4 个 actuator: 2 关节 × (position + velocity) servo

─────── 状态机 ───────
fsm ∈ {AIR1, STANCE1, STANCE2, AIR2}
转移条件: 脚高度 / 躯干速度 阈值

─────── 控制律 ───────
每个状态一对增益 (kp, kv):
  F_joint = kp · (ctrl_target - q) + kv · (ctrl_vel_target - qd)
运行时改 model.actuator_gainprm / biasprm

─────── 周期 ───────
AIR1  (0.4s)  → STANCE1  (0.05s)
     → STANCE2  (0.05s)  [data.ctrl[0]=-0.2 腿后摆]
     → AIR2    (0.4s)   [step_no++]
     → AIR1
```

---

## 十二、一句话总结

> **No.9 = 「4 状态事件驱动 FSM + 运行时动态增益切换」**。这是 Raibert hopper 风格的经典控制范式 —— 跳跃周期用**物理事件**（脚高度、躯干速度）分段，每段用**不同的 PD 增益**实现软着地 / 硬储能 / 完全弹性反弹。**核心创新**是 `set_position_servo / set_velocity_servo` 运行时改 `actuator_gainprm`，让同一个 actuator 在不同时刻扮演弹簧/阻尼器/刚性关节。
