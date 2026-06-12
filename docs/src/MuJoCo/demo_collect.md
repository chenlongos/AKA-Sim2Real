# 统一 Demo：移动操作机器人 + 数据采集（No.1–No.13 融合）

本节把 No.1–No.13 全部概念融合到一个文件 `demo_collect.py` 中：一辆带 4-DOF 机械臂的小车，通过 **9 状态 FSM** 自动完成"捡地上的盒子 → 运回来 → 放下"，并**同步采集训练数据**。

---

## 文件说明

```
mujoco/Chenlong_Robot/
├── car.xml           # MuJoCo 模型（小车 + 2× 臂 + 双指夹爪 + 28 路传感器）
├── demo_collect.py   # 统一 Demo + 数据采集（~800 行）
└── episodes/         # 采集数据：ep_*.npz + summary_*.png
```

运行：

```bash
python3 demo_collect.py                     # GUI，默认 5 轮
python3 demo_collect.py --headless --episodes 10  # 无头，10 轮
```

---

## 一、car.xml 模型详解

```xml
<mujoco>
    <option timestep="0.001" integrator="RK4" gravity="0 0 -15"/>
    <!--
        timestep=0.001 → 1000Hz 仿真频率
        integrator=RK4 → 四阶龙格库塔，比默认 Euler 更精确
        gravity=0 0 -15 → 重力稍强于地球（9.81），加快掉落
    -->

    <default>
        <geom friction="1 0.1 0.1"/>
        <!-- 所有 geom 的默认摩擦：滑动1、扭转0.1、滚动0.1 -->
    </default>

    <worldbody>
        <!-- 5 点均匀光照：无阴影，各方向可见 -->
        <light diffuse="0.6 0.6 0.6" pos="0 0 6"  dir="0 0 -1"/>

        <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"
              friction="2.5 1.0 0.001"/>
        <!--
            地面：5×5m 灰色平面
            friction: 滑动2.5 / 扭转1.0 / 滚动0.001
        -->

        <!-- ===== 目标盒子：自由落体，初始放在地上 ===== -->
        <body name="target_box" pos="1.5 0 0.04">
            <freejoint/>
            <!-- freejoint → 6-DOF 自由运动 -->
            <geom type="box" size="0.04 0.04 0.04"
                  rgba="1.0 0.85 0.2 1" mass="0.05"/>
            <!-- 8cm 黄色立方体，质量 50g -->
        </body>

        <!-- ===== 小车主体 ===== -->
        <body name="car" pos="0 0 0.22">
            <freejoint/>
            <!-- 小车也是 freejoint → 可以自由驾驶 -->

            <geom type="box" size="0.5 0.3 0.15"
                  rgba="0.2 0.6 0.8 1" mass="10"
                  contype="0" conaffinity="0"/>
            <!--
                车身：1m×0.6m×0.3m 蓝色盒子
                contype=0 conaffinity=0 → 车身不参与碰撞，只有轮子接触地面
            -->

            <!-- ===== 4-DOF 机械臂 ===== -->
            <body name="arm_base" pos="0 0 0.22">
                <!--
                    臂基座：在车顶上方 22cm
                    shoulder_pan：绕 Z 轴旋转（水平转台），±360°
                -->
                <joint name="shoulder_pan" type="hinge" axis="0 0 1"
                       range="-360 360" armature="0.10" damping="0.5"/>
                <geom type="cylinder" size="0.08 0.07"
                      rgba="0.9 0.5 0.1 1" mass="0.35"/>

                <body name="upper_arm" pos="0 0 0.12">
                    <!--
                        upper_arm：上臂，长 0.50m，粗 0.065m
                        shoulder_lift：绕 Y 轴俯仰，±360°
                    -->
                    <joint name="shoulder_lift" type="hinge" axis="0 1 0"
                           range="-360 360" armature="0.10" damping="0.5"/>
                    <geom type="capsule" size="0.065"
                          fromto="0 0 0 0 0 0.50"
                          rgba="0.9 0.5 0.1 1" mass="0.40"/>

                    <body name="forearm" pos="0 0 0.50">
                        <!--
                            forearm：前臂，长 0.40m，粗 0.055m
                            elbow：肘关节，±360°
                        -->
                        <joint name="elbow" type="hinge" axis="0 1 0"
                               range="-360 360" armature="0.10" damping="0.5"/>
                        <geom type="capsule" size="0.055"
                              fromto="0 0 0 0 0 0.40"
                              rgba="0.9 0.5 0.1 1" mass="0.30"/>

                        <body name="wrist" pos="0 0 0.40">
                            <!--
                                wrist：手腕，长 0.16m，粗 0.045m
                                wrist_pitch：腕部俯仰，±360°
                            -->
                            <joint name="wrist_pitch" type="hinge" axis="0 1 0"
                                   range="-360 360" armature="0.06" damping="0.5"/>
                            <geom type="capsule" size="0.045"
                                  fromto="0 0 0 0 0 0.16"
                                  rgba="0.9 0.5 0.1 1" mass="0.15"/>

                            <!-- 双指平行夹爪 -->
                            <body name="gripper_palm" pos="0 0 0.16">
                                <!-- 手掌：固定不动 -->
                                <geom type="box" pos="0 0 -0.03"
                                      size="0.04 0.06 0.03"
                                      rgba="0.3 0.3 0.3 1" mass="0.10"/>

                                <body name="finger_l" pos="0 0.03 0.04">
                                    <!--
                                        左指：slide 沿 +Y，范围 0→0.04m
                                        position=0 → 手指内收（夹紧）
                                        position=0.04 → 手指张开
                                    -->
                                    <joint name="finger_l_j" type="slide"
                                           axis="0 1 0" range="0 0.04"
                                           armature="0.01" damping="1.0"/>
                                    <geom type="box" size="0.015 0.008 0.06"
                                          rgba="0.5 0.5 0.5 1" mass="0.03"/>
                                    <site name="touch_l" pos="0 -0.01 0"
                                          size="0.004" rgba="0 1 0 0.5"/>
                                    <!-- touch_l：触觉 site，位于指尖内侧 -->
                                </body>

                                <body name="finger_r" pos="0 -0.03 0.04">
                                    <!-- 右指：slide 沿 -Y，镜像左指 -->
                                    <joint name="finger_r_j" type="slide"
                                           axis="0 -1 0" range="0 0.04"
                                           armature="0.01" damping="1.0"/>
                                    <geom type="box" size="0.015 0.008 0.06"
                                          rgba="0.5 0.5 0.5 1" mass="0.03"/>
                                    <site name="touch_r" pos="0 0.01 0"
                                          size="0.004" rgba="0 1 0 0.5"/>
                                </body>

                                <site name="end_effector" pos="0 0 0.05"
                                      size="0.02" rgba="1 0 0 1"/>
                                <!-- EE site：红色标记点，用于 IK 目标 -->
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- ===== 四个轮子 ===== -->
            <body name="wheel_fl" pos="0.4 0.3 -0.1">
                <joint name="wheel_fl_joint" type="hinge"
                       axis="0 1 0" damping="0.5"/>
                <geom type="cylinder" size="0.12 0.05" euler="90 0 0"
                      rgba="0.1 0.1 0.1 1" mass="1"/>
                <!--
                    euler="90 0 0" → 把圆柱从 Z 轴翻转到 Y 轴（变成竖直轮子）
                    size: 半径0.12m, 半宽0.05m
                -->
            </body>
            <!-- wheel_fr, wheel_rl, wheel_rr 结构相同，省略 -->
        </body>
    </worldbody>

    <actuator>
        <!-- 4 个轮子 motor -->
        <motor joint="wheel_fl_joint" name="motor_wheel_fl"
               gear="3" ctrllimited="true" ctrlrange="-5 5"/>
        <!-- gear=3 → 力矩放大3倍  ctrlrange=±5 → 速度限制 -->

        <!-- 4 个臂关节 position servo（PD 控制器） -->
        <position name="pservo_shoulder_pan"  joint="shoulder_pan"  kp="200" kv="10"/>
        <position name="pservo_shoulder_lift" joint="shoulder_lift" kp="200" kv="10"/>
        <position name="pservo_elbow"          joint="elbow"         kp="200" kv="10"/>
        <position name="pservo_wrist_pitch"    joint="wrist_pitch"   kp="120" kv="6"/>
        <!--
            position servo：ctrl = kp*(target - q) - kv*qvel
            kp=200, kv=10 → 快速跟踪，有一定阻尼
            wrist_pitch 降低增益（防止末端抖动）
        -->

        <!-- 2 个手指 servo -->
        <position name="pservo_finger_l" joint="finger_l_j" kp="200" kv="10"/>
        <position name="pservo_finger_r" joint="finger_r_j" kp="200" kv="10"/>
    </actuator>

    <equality>
        <!-- 抓取 weld：激活时把盒子焊在手掌上 -->
        <weld name="grasp_weld" body1="gripper_palm" body2="target_box"
              active="false" solref="0.02 1" solimp="0.9 0.95 0.001"/>
        <!--
            active="false" → 初始不激活，GRASP 状态时激活
            solref="0.02 1" → 约束刚度：时间常数20ms，阻尼比1（临界）
        -->
    </equality>

    <sensor>
        <!-- 6 个臂关节 × 2（位置+速度）= 12 -->
        <jointpos joint="shoulder_pan"  name="arm_pan_pos"/>
        <jointvel joint="shoulder_pan"  name="arm_pan_vel"/>
        <!-- ... lift/elbow/wrist/finger_l/finger_r 各两个 ... -->

        <!-- 4 个轮子 × 2 = 8 -->
        <jointpos joint="wheel_fl_joint" name="wheel_fl_pos"/>
        <jointvel joint="wheel_fl_joint" name="wheel_fl_vel"/>
        <!-- ... -->

        <!-- 末端位姿+速度 = 6 -->
        <framepos    objtype="site" objname="end_effector" name="ee_pos"/>
        <framelinvel objtype="site" objname="end_effector" name="ee_vel"/>

        <!-- 手指触觉 = 2 -->
        <touch site="touch_l" name="finger_l_touch"/>
        <touch site="touch_r" name="finger_r_touch"/>
        <!-- touch sensor：接触时输出 >0，不接触 =0 -->
    </sensor>
</mujoco>
```

### 臂几何参数

| 段 | 长度 | 半径 | 关节 | 范围 |
|----|------|------|------|------|
| 上臂 | 0.50m | 0.065m | shoulder_lift | ±360° |
| 前臂 | 0.40m | 0.055m | elbow | ±360° |
| 手腕 | 0.16m | 0.045m | wrist_pitch | ±360° |
| 夹爪 | — | — | finger_l/r slide | 0→0.04m |
| **总臂展** | **1.06m** | | | |

---

## 二、demo_collect.py 逐段详解

### 2.1 常量与几何参数（第 59–92 行）

```python
# 盒子放在地上 (z=0.04 是盒子半高，底部刚好贴地)
TARGET_POS = np.array([1.5, 0.0, 0.04])
PLACE_POS  = np.array([0.25, 0.0, 0.55])  # 放到原点前方

# 臂长 —— 必须与 car.xml 一致
L_UPPER   = 0.50
L_FOREARM = 0.40
L_WRIST   = 0.16
L_GRIPPER = 0.08
L2_EFF    = L_FOREARM + L_WRIST + L_GRIPPER  # = 0.64，用于 2 连杆 IK

# 控制参数
DRIVE_SPEED    = 0.5   # 轮子速度
REACH_TOL      = 0.18  # 末端到达容差（18cm）
TRAJ_DURATION  = 0.6   # 三次轨迹时长（秒）
NUM_EPISODES   = 5     # 默认运行 5 轮

# 夹爪位置
STOWED = [0.0, 0.5, -1.0, 0.0, 0.04, 0.04]  # [pan, lift, elbow, wrist, f_l, f_r]
#                                                手指=0.04 → 张开

# 关节限位（必须与 XML 中 range 一致）
Q_MIN = np.array([-6.28, -6.28, -6.28, -6.28])  # ±360°
Q_MAX = np.array([ 6.28,  6.28,  6.28,  6.28])
```

---

### 2.2 模型 ID 缓存（第 109–124 行）

```python
def _cache_ids(m):
    """一次性查表，存下所有 body/site/joint 的 ID，后续用整数寻址比字符串快。"""
    for name in ("car", "arm_base", "target_box"):
        _ids[name] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
    _ids["ee"]       = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "end_effector")
    _ids["gripper"]  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper_palm")

    arm_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch",
                 "finger_l_j", "finger_r_j"]
    _arm_qpos_adr = [m.jnt_qposadr[m.joint(n).id] for n in arm_names]
    _arm_dof_adr  = [m.jnt_dofadr[m.joint(n).id]  for n in arm_names]
    # jnt_qposadr → 关节在 qpos 数组中的起始下标
    # jnt_dofadr  → 关节在 qvel 数组中的起始下标
```

**为什么需要 `_arm_qpos_adr`？** 因为模型中还有 target_box freejoint、car freejoint 在 qpos 前面，不能直接假设下标。通过 `jnt_qposadr` 查表才是正确的。

---

### 2.3 状态读取器（第 129–134 行）

```python
def _car_pos(d):      return d.xpos[_ids["car"]]        # (3,) 世界坐标
def _arm_base_pos(d): return d.xpos[_ids["arm_base"]]    # (3,)
def _target_pos(d):   return d.xpos[_ids["target_box"]]  # (3,)
def _ee_pos(d):       return d.site_xpos[_ids["ee"]]     # (3,) site 的世界坐标
def _arm_qpos(d):     return np.array([d.qpos[a] for a in _arm_qpos_adr])  # (6,)
def _car_quat(d):     return d.xquat[_ids["car"]].copy() # (4,) w,x,y,z
```

**`xpos` vs `qpos`**：`xpos` 是笛卡尔世界坐标（已通过正运动学计算），`qpos` 是关节空间原始值。用 `xpos` 读位置不需要手工做坐标系变换。

---

### 2.4 解析 2D IK（第 140–155 行）—— No.6 核心

```python
def _analytical_ik_2d(x, z, L1=L_UPPER, L2=L2_EFF):
    """2 连杆平面 IK：已知目标 (x,z)，求 (lift, elbow, wrist)。"""
    D = np.hypot(x, z)           # 目标距臂基座的距离
    max_r = L1 + L2              # 最大臂展
    if D > max_r * 0.95:         # 太远 → 缩放到 95% 最大范围
        s = max_r * 0.95 / D
        x, z = x * s, z * s

    # 余弦定理求肘关节角
    D2 = x*x + z*z
    cos_q2 = np.clip((D2 - L1*L1 - L2*L2) / (2*L1*L2), -1.0, 1.0)
    q2_int = np.arccos(cos_q2)   # 肘部内角（0=伸直, π=完全折叠）

    alpha = np.arctan2(z, x)     # 目标方向角
    beta  = np.arctan2(L2 * np.sin(q2_int), L1 + L2 * np.cos(q2_int))
    q1 = -(alpha - beta)         # shoulder_lift（负号来自 MuJoCo 轴方向）
    q2 = -(np.pi - q2_int)       # elbow
    q3 = -(q1 + q2)              # wrist_pitch（保持末端水平）
    return q1, q2, q3
```

**几何意义**：

```
        L1        L2
   ●─────────●─────────●  (目标)
  arm_base   elbow      EE

已知 arm_base → EE 的距离 D，用余弦定理算出 elbow 折叠角 q2_int，
再通过 α（目标方向）和 β（上臂相对目标线的偏角）合成 q1。
```

---

### 2.5 数值 IK 精调（第 164–211 行）—— No.11 核心

```python
def numerical_ik(target_world, arm_base_pos, q_guess=None,
                 max_iter=40, tol=0.015, alpha=0.5, gripper_cmd=0.0):
    m, d = model, _fk_data

    # Step 1: 解析 IK 给出初值
    local = target_world - arm_base_pos
    r_xy = np.hypot(local[0], local[1])
    pan0 = float(np.clip(np.arctan2(local[1], local[0]), -1.57, 1.57))
    lift0, elbow0, wrist0 = _analytical_ik_2d(r_xy, local[2])
    q = np.clip([pan0, lift0, elbow0, wrist0], Q_MIN, Q_MAX)

    # Step 2: 把主仿真状态拷贝到独立的 FK data 中（不污染主数据）
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = data.qvel[:]
    mj.mj_fwdPosition(m, d)

    # Step 3: Jacobian 伪逆迭代
    for _ in range(max_iter):
        # 把当前关节角写入 FK data 并计算正运动学
        for adr, val in zip(_arm_qpos_adr,
                            [q[0], q[1], q[2], q[3], gripper_cmd, gripper_cmd]):
            d.qpos[adr] = val
        mj.mj_fwdPosition(m, d)
        ee = d.site_xpos[_ids["ee"]]
        err = target_world - ee[:3]      # 位置误差（世界坐标）

        if np.linalg.norm(err) < tol:
            break                         # 收敛

        # 计算末端位置雅可比（3×nv），提取 4 个臂关节列
        jacp = np.zeros((3, m.nv))
        mj.mj_jac(m, d, jacp, None, ee[:3], _ids["gripper"])
        J = np.zeros((3, 4))
        for i, adr in enumerate(_arm_dof_adr[:4]):
            J[:, i] = jacp[:, adr]        # 只取 pan/lift/elbow/wrist 四列

        # 阻尼伪逆：Δq = (JᵀJ + λI)⁻¹ Jᵀ · err · α
        lam = 0.05
        dq = np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ err) * alpha
        q = np.clip(q + dq, Q_MIN, Q_MAX)
        q[3] = float(np.clip(-(q[1] + q[2]), -1.57, 1.57))
        # ↑ 强制 wrist = -(lift+elbow)，保持末端水平

    return np.array([q[0], q[1], q[2], q[3], gripper_cmd, gripper_cmd])
```

**为什么用独立 FK data（No.12 模式）**：迭代过程中的 `mj_fwdPosition` 是纯数学计算，不能影响主仿真状态。独立的 `MjData` 保证隔离。

**阻尼 λ 的作用**：当 J 接近奇异（臂伸直）时，JᵀJ 不可逆。λ=0.05 的阻尼项保证始终有解。

---

### 2.6 三次轨迹插值（第 217–225 行）—— No.5 核心

```python
def _cubic_coeffs(q0, qf, T):
    """q(t) = a₀ + a₂t² + a₃t³，边界速度=0"""
    return np.array([q0, 0.0, 3*(qf - q0)/(T*T), -2*(qf - q0)/(T*T*T)])

def _eval_cubic(c, t):
    """求值：返回 (pos, vel)"""
    pos = c[0] + c[1]*t + c[2]*t*t + c[3]*t*t*t
    vel = c[1] + 2*c[2]*t + 3*c[3]*t*t
    return pos, vel
```

**为什么用三次多项式？** 给定起点 q₀、终点 q_f、起止速度=0，三次多项式是唯一满足这 4 个约束的最低次多项式。比直接跳变平滑，不会激发机械臂振动。

---

### 2.7 动力学提取（第 231–238 行）—— No.4 核心

```python
def _log_mass_matrix_diag():
    """每次 FSM 切换时，提取并打印质量矩阵对角线。"""
    M = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, M, data.qM)  # 从稀疏 qM 构建稠密 M
    print(f"[No.4 Dynamics] mass-matrix diagonal (nv={model.nv}): ...")
```

质量矩阵对角线反映了各关节的等效惯量。数值大 → 该关节"重"，加速慢。

---

### 2.8 抓取管理（第 243–250 行）—— No.8 核心

```python
def _grasp(activate):
    """激活/停用 grasp_weld 约束。"""
    eq = model.eq("grasp_weld")
    eq.active0[0] = 1 if activate else 0
```

weld 是 MuJoCo 的 equality constraint，把两个 body 刚性连接。处理方式比纯摩擦力可靠。

---

### 2.9 DataCollector 数据采集器（第 255–330 行）

```python
class DataCollector:
    def maybe_record(self, t, fsm, ee, target, last_action):
        """10Hz 采样：状态、动作、传感器数据"""
        if t - self._last_coll_t < 0.1:   # 10Hz = 0.1s 间隔
            return
        self.joint_states.append(np.concatenate([
            _car_pos(data), _car_quat(data), _arm_qpos(data)]))  # 7+6=13
        self.sensordata.append(data.sensordata.copy())  # 28 维全部传感器
        self.actions.append(last_action.copy())          # 4轮+6臂=10

    def save(self):
        """保存为 .npz"""
        np.savez_compressed(path,
            joint_states=..., ee_position=..., target_position=...,
            actions=..., sensordata=..., fsm_state=..., timestamps=...)
```

---

### 2.10 FSM 状态机（第 334–458 行）—— No.5 + No.9 核心

**9 个状态**：

```
DRIVE(0) → REACH(1) → LOWER(2) → GRASP(3) → LIFT(4)
  → DRIVE_BACK(5) → PLACE(6) → RELEASE(7) → DONE(8)
```

**状态转换函数**：

```python
def _check_transitions(st):
    car_x = _car_pos(data)[0]
    ee    = _ee_pos(data)
    elap  = data.time - st.fsm_enter

    # 超时保护：防止某状态永远达不到条件
    if elap > FSM_TIMEOUT.get(st.fsm, float("inf")):
        return 下一个状态

    if fsm == FSM_DRIVE:
        if car_x > DRIVE_TARGET_X:          # 车开到 x>1.05 就停
            return FSM_REACH

    elif fsm == FSM_REACH:
        goal = TARGET_POS.copy(); goal[2] += 0.15
        if np.linalg.norm(ee - goal) < REACH_TOL:  # EE 到目标上方 18cm 内
            return FSM_LOWER

    elif fsm == FSM_LOWER:
        goal[2] = max(TARGET_POS[2], box_z + 0.10)
        if np.linalg.norm(ee - goal) < REACH_TOL:  # EE 到盒子附近
            return FSM_GRASP

    elif fsm == FSM_GRASP:
        if elap > 1.0:                      # 等 1 秒让手指夹紧
            return FSM_LIFT

    elif fsm == FSM_LIFT:
        goal = TARGET_POS.copy(); goal[2] += 0.15
        if np.linalg.norm(ee - goal) < REACH_TOL:  # 抬起到安全高度
            return FSM_DRIVE_BACK

    elif fsm == FSM_DRIVE_BACK:
        if car_x < 0.30:                    # 开回原点附近
            return FSM_PLACE

    elif fsm == FSM_PLACE:
        if np.linalg.norm(ee - PLACE_POS) < REACH_TOL:  # EE 到放置点
            return FSM_RELEASE

    elif fsm == FSM_RELEASE:
        if elap > 0.5:                      # 等 0.5 秒手指张开
            return FSM_DONE
```

**FSM 切换时做的事**（在 controller 中）：

```python
if nxt != fsm:
    if nxt == FSM_GRASP:
        _grasp(True)                  # 激活 weld，锁住盒子
    if nxt == FSM_RELEASE:
        _grasp(False)                 # 释放 weld
    state.fsm = nxt
    state.fsm_enter = data.time
    _compute_arm_target(nxt, state)   # 为新状态算一次 IK（只算一次！）
    # 启动三次轨迹：从当前 q → 新目标，TRAJ_DURATION 秒内平滑过渡
    state._traj_coeffs = np.array([_cubic_coeffs(cur_q[j],
        state._cached_target[j], TRAJ_DURATION) for j in range(6)])
    _log_mass_matrix_diag()           # No.4 动力学快照
```

---

### 2.11 controller 控制回调（第 465–557 行）

每步仿真（1000Hz）调用一次，是整个系统的"大脑"：

```python
def controller(m, d):
    fsm = state.fsm

    # ── ① 车控制 ──
    if fsm == FSM_DRIVE:
        # 比例速度：离目标越近越慢
        dist = max(0, DRIVE_TARGET_X - car_x)
        speed = min(DRIVE_SPEED, 0.8 * dist + 0.15)
        d.ctrl[0:4] = max(0.1, speed)          # 最低 0.1 克服静摩擦
    elif fsm == FSM_DRIVE_BACK:
        dist = max(0, car_x - 0.3)
        speed = min(DRIVE_SPEED, 0.8 * dist + 0.15)
        d.ctrl[0:4] = -max(0.1, speed)         # 反向
    else:
        # 刹车：用速度反馈抵抗剩余速度
        for i in range(4):
            wv = d.qvel[wheel_dof_start + i]
            d.ctrl[i] = -np.sign(wv) * BRAKE_TORQUE if abs(wv) > 0.01 else 0.0

    # ── ② 臂控制 ──
    # 每 0.5 秒刷新一次 IK（车刹停后可能还在漂移）
    if fsm in (FSM_REACH, FSM_LOWER, FSM_LIFT, FSM_PLACE):
        if data.time - state._last_ik_time > 0.5:
            _compute_arm_target(fsm, state)
            # 如果新目标和旧目标差距 > 0.03 rad，重新启动轨迹
            if np.linalg.norm(new - old) > 0.03:
                # 生成新三次轨迹
                state._traj_coeffs = ...

    # 轨迹插值：在轨迹期间用三次多项式，结束后直接 hold
    if state._traj_coeffs active:
        arm_cmd = eval_cubic(traj, data.time)   # 平滑过渡
    else:
        arm_cmd = state._cached_target           # 保持在目标位置

    for i in range(6):
        d.ctrl[4 + i] = arm_cmd[i]              # 写入 position servo

    # ── ③ 数据记录 ──
    collector.maybe_record(data.time, fsm, ee, target, last_action)

    # ── ④ FSM 转换 ──
    nxt = _check_transitions(state)
    if nxt != fsm:
        # 抓取/释放管理 + 重新算IK + 启动新轨迹
```

**ctrl 数组布局**：

| 下标 | 内容 | 驱动方式 |
|------|------|---------|
| 0–3 | 4 轮 motor | 速度指令 |
| 4 | shoulder_pan | position servo (PD) |
| 5 | shoulder_lift | position servo |
| 6 | elbow | position servo |
| 7 | wrist_pitch | position servo |
| 8 | finger_l | position servo |
| 9 | finger_r | position servo |

---

### 2.12 GLFW 鼠标/键盘（第 562–609 行）—— No.2 模式

```python
# Backspace → 重置仿真
mj.mj_resetData(model, data)
mj.mj_forward(model, data)
_grasp(False)
state = SimState()          # 重新创建 FSM 状态

# S → 手动保存数据
collector.save()

# Q/Esc → 退出
glfw.set_window_should_close(window, True)

# 鼠标拖拽 → 旋转/平移/缩放
mj.mjv_moveCamera(model, action, dx, dy, scene, cam)
```

---

### 2.13 main 主循环（第 671–793 行）—— No.1 + No.2

```python
def main(headless=False):
    # 加载模型
    model = mj.MjModel.from_xml_path(XML_PATH)
    data  = mj.MjData(model)
    _cache_ids(model)
    _fk_data = mj.MjData(model)   # 独立的 FK 数据（No.12）

    if headless:
        # 无头模式：纯 mj_step 循环，N 轮自动重置
        for ep in range(NUM_EPISODES):
            while data.time < simend and state.fsm != FSM_DONE:
                mj.mj_step(model, data)
            mj.mj_resetData(model, data)  # 轮间重置
        collector.save()
        return

    # GUI 模式：
    # ① 创建 GLFW 窗口
    # ② 设置鼠标/键盘回调
    # ③ 渲染循环:
    while not glfw.window_should_close(window):
        # 每 1/60 秒渲染一帧，期间跑多个 mj_step
        while data.time - t_start < 1/60:
            mj.mj_step(model, data)

        # FSM 完成后自动重置进入下一轮
        if state.fsm == FSM_DONE:
            ep_count += 1
            mj.mj_resetData(model, data)
            state = SimState()

        # 相机跟踪小车
        cam.lookat = 0.5*(car + TARGET)  # 中点

        # 渲染
        mj.mjv_updateScene(...)
        mj.mjr_render(...)
        glfw.swap_buffers(window)

    # 结束：保存数据 + matplotlib 总结图
    collector.save()
    _plot_summary()
```

**渲染 vs 仿真频率**：仿真跑 1000Hz（timestep=0.001），渲染跑 60Hz。每帧之间跑 ~16 个 `mj_step`。

---

## 三、数据流总览

```
mj_step (1000Hz)
  └── controller (每步)
        ├── 车: 比例速度 / 刹车
        ├── 臂: 读缓存目标 → 三次轨迹插值 → 写 ctrl[4:10]
        ├── 记录: collector.maybe_record (10Hz)
        └── FSM: 检查转换条件 → (切换?) → 算 IK → 启动新轨迹
              └── GRASP 时激活 weld, RELEASE 时释放

渲染循环 (60Hz)
  └── mj_step × ~16 → updateScene → render → swapBuffers

结束后
  └── collector.save() → ep_*.npz
  └── matplotlib → summary_*.png
```

---

## 四、No.1–No.13 概念对照

| No. | 概念 | 在 demo_collect.py 中的位置 |
|-----|------|---------------------------|
| No.1 | 基础仿真循环 | `main()` — `mj_step` + render loop |
| No.2 | GLFW 渲染 | `main()` — GLFW window + camera + scene + 鼠标回调 |
| No.3 | 位置伺服 PD | `controller` — `data.ctrl[4:10]` 写 position servo 目标 |
| No.4 | 动力学提取 | `_log_mass_matrix_diag` — `mj_fullM` 提取 M 矩阵对角线 |
| No.5 | FSM + 三次轨迹 | `_check_transitions` 9状态机 + `_cubic_coeffs` 轨迹插值 |
| No.6 | Jacobian IK | `numerical_ik` — `mj_jac` 算雅可比 + 伪逆 |
| No.7 | 状态反馈控制 | `controller` — 比例速度驱动 + 速度反馈刹车 |
| No.8 | 约束管理 | `_grasp` — 激活/停用 weld equality |
| No.9 | 位置触发 | `_check_transitions` — car_x/EE距离触发状态切换 |
| No.11 | 数值优化 | `numerical_ik` — 阻尼伪逆迭代 |
| No.12 | 独立 FK + 绘图 | `_fk_data` 独立 MjData + `_plot_summary` matplotlib |
| No.13 | 姿态估计 | `_car_quat` — `xquat` 读车身四元数 + 相机跟踪 |
