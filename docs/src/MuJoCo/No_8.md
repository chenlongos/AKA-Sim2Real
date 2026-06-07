# No.8 双摆约束力「移交」仿真

本节介绍一个**混合仿真**（hybrid simulation）技巧：用**约束反作用力**作为「虚拟连杆」，在两个**没有真实机械连接**的 body 之间传递力，并在适当时机「释放」。这是把 `connect` equality 约束当作**传感器**用的典型例子。

---

## 文件说明

```
mujoco/No_8/
├── pendulum.xml        # MuJoCo XML 模型文件
└── hybrid_pendulum.py  # 完整脚本：含约束力读取、FSM 状态机
```

> No.8 **没有**最小脚本（`no_8.py`）。要看效果必须跑 `hybrid_pendulum.py`。

---

## 一、pendulum.xml 详解（对比 No.7）

```xml
<mujoco>
    <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81">
        <flag contact="enable" energy="enable"/>
    </option>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.1" rgba=".9 0 0 1"/>

        <!--
            第一个 body：x 平移 + z 平移 + 旋转（3 个 slide + 1 个 hinge）
            【No.8 关键】所有 3 个 DOF 都参与约束求解
        -->
        <body name="pole" pos="0 0 2" euler="0 0 0">
            <joint name="x"   type="slide" pos="0 0 0.5" axis="1 0 0" />
            <joint name="z"   type="slide" pos="0 0 0.5" axis="0 0 1" />
            <joint name="pin" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
            <geom type="cylinder" size=".05 .5" rgba="0 .9 0 0.1" mass="1"/>
        </body>

        <!--
            第二个 body：完全自由（不参与任何约束）
        -->
        <body name="pole2" pos="0 -1 2" euler="0 0 0">
            <joint name="x2"   type="slide" pos="0 0 0.5" axis="1 0 0" />
            <joint name="z2"   type="slide" pos="0 0 0.5" axis="0 0 1" />
            <joint name="pin2" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
            <geom type="cylinder" size=".05 .5" rgba=".9 .9 .9 1" mass="1"/>
        </body>
    </worldbody>

    <!--
        【No.8 关键】connect equality：把 pole 的 anchor 点钉在世界的 (0,0,0.5)
        这是 3D 约束（x, y, z 三个分量都约束）
    -->
    <equality>
        <connect body1="pole" body2="world" anchor="0 0 0.5" />
    </equality>

    <actuator>
        <motor name="torque" joint="pin" gear="1" ctrlrange="-100 100" ctrllimited="true"/>
        <position name="position_servo" joint="pin" kp="0"/>
        <velocity name="velocity_servo" joint="pin" kv="0"/>
    </actuator>
</mujoco>
```

### XML 配置对比表

| 配置项 | No.7 (LQR) | No.8 (Hybrid) |
|--------|-----------|---------------|
| body 数量 | 2（嵌套双摆）| 2（**独立**两个 body） |
| body 关系 | 父子嵌套 | **互不连接** |
| 关节类型 | 2 个 hinge | 6 个（3 slide + 3 hinge）|
| DOF 总数 | 2 | **6** |
| 自由度 | `nv=2` | `nv=6` |
| actuator | 1 motor | 1 motor + 2 servo（servo kp/kv=0） |
| equality | ❌ | **`<connect>` 新增**（3D 约束）|
| 重力 | 默认 | 默认（开启）|
| 约束求解 | 无 | **有**（connect 是硬约束）|
| timestep | 0.001 | 0.001 |

### 关键变更说明

#### 1. 两个 body 没有机械连接

No.7 的两个 body 是**父子嵌套**（自动连成一根杆）。No.8 的 `pole` 和 `pole2` 是**两个独立的 body** —— 它们**没有真实的关节相连**。

#### 2. `<connect>` 等式约束

```xml
<connect body1="pole" body2="world" anchor="0 0 0.5" />
```

含义：把 `pole` 上**局部坐标** `(0, 0, 0.5)` 这个点，**钉死**在世界的 `(0, 0, 0.5)` 上。MuJoCo 内部用**拉格朗日乘子**强制这个约束成立。

**约束维度**：3（x, y, z 三个分量）。所以 `nefc = 3`（活跃约束数 = 3）。

#### 3. 6 个 DOF

| DOF | 关节 | 物理含义 | 是否被 connect 约束 |
|-----|------|---------|-------------------|
| 0 | `x`  | pole 沿 x 平移 | ✅ |
| 1 | `z`  | pole 沿 z 平移 | ✅ |
| 2 | `pin`| pole 绕 y 旋转 | ✅ |
| 3 | `x2` | pole2 沿 x 平移 | ❌ 自由 |
| 4 | `z2` | pole2 沿 z 平移 | ❌ 自由 |
| 5 | `pin2`| pole2 绕 y 旋转 | ❌ 自由 |

> **关键事实**：`pole` 实际**只剩 0 个自由 DOF**（被 connect 全部钉死）。`pole2` 才有 3 个真正自由的 DOF。

---

## 二、核心概念：connect 等式约束与约束 Jacobian

### 2.1 约束的本质

```xml
<connect body1="pole" body2="world" anchor="0 0 0.5" />
```

这是一个**硬约束**：在仿真**每一步**，MuJoCo 都要求：

```
position_of_anchor_on_pole_in_world == (0, 0, 0.5)    (3D 等式)
```

**不满足就施加拉格朗日乘子力**让它满足。

### 2.2 约束 Jacobian（efc_J）

`efc_J` 是约束对各 DOF 的**偏导数矩阵**：

```
[约束违反量] = J · qvel
[3×1]       [3×6][6×1]
```

每一行对应一个约束维度（x, y, z 的位置误差），每一列对应一个 DOF。

| 索引 | 形状 | 含义 |
|------|------|------|
| `data.efc_J` | `(nefc*nv,)` 1D 平铺 | 约束 Jacobian（**MuJoCo 3.x 是 1D**） |
| `data.efc_force` | `(nefc,)` 1D | 约束反作用力（拉格朗日乘子） |
| `data.nefc` | int | 当前活跃约束数（connect 给 3） |
| `model.nv` | int | 系统自由度数（这里 = 6） |

### 2.3 约束反作用力（efc_force）

`efc_force` 是**保持约束所需的力**（3D 笛卡尔力）：

```
F0 = efc_force[:3]     # (3,)  ← x, y, z 三个方向的约束反力
```

物理含义：把 `pole` 的 anchor 钉在 (0, 0, 0.5) **需要的力有多大**。重力 + 摆动 + 一切外力产生的「偏离趋势」都由这个力抵消。

---

## 三、controller 详解：FSM + 力移交

### 3.1 完整代码

```python
def controller(model, data):
    global fsm

    # 1. 读约束 Jacobian（reshape 后切片）
    J = data.efc_J[:data.nefc * model.nv].reshape((data.nefc, model.nv))[:3, :3]

    # 2. 读约束反作用力
    F0 = data.efc_force[:3][:, np.newaxis]   # (3, 1)

    # 3. 把约束力「投影」到 pole 的 3 个 DOF
    JT_F = J.T @ F0                            # (3, 1)

    # 4. 状态转移：pin2 旋转 > 1.0 rad → 释放
    if fsm == FSM_SWING and data.qpos[5] > 1.0:
        fsm = FSM_FREE

    # 5. SWING 阶段：驱动 pin + 移交约束力到 pole2
    if fsm == FSM_SWING:
        data.qfrc_applied[2] = -1 * (data.qvel[2] - 5.0)        # pin 速度控制
        data.qfrc_applied[3] = JT_F[0, 0]                       # x2 ← 约束力 x
        data.qfrc_applied[4] = JT_F[1, 0]                       # z2 ← 约束力 y
        data.qfrc_applied[5] = JT_F[2, 0] + data.qfrc_applied[2]  # pin2 ← 约束力 z + pin 力矩

    # 6. FREE 阶段：pole2 自由飞
    elif fsm == FSM_FREE:
        data.qfrc_applied[3] = 0.0
        data.qfrc_applied[4] = 0.0
        data.qfrc_applied[5] = 0.0
```

### 3.2 状态机

```
                          pin2 旋转 > 1.0 rad
       ┌───────────────────────────────────────────┐
       │                                           │
       ▼                                           │
  ┌─────────┐                                   ┌─────────┐
  │ FSM_SWING│                                  │ FSM_FREE│
  │  摆动 +   │ ──────────────────────────────▶ │   释放   │
  │  力移交   │                                  │ 自由飞行 │
  └─────────┘                                   └─────────┘
       │                                           │
       └───────────────────────────────────────────┘
                  持续仿真直到 simend
```

| 状态 | 条件 | 行为 |
|------|------|------|
| `FSM_SWING` | 初始 / pin2 角度 ≤ 1.0 | 驱动 pin 加速到 5 rad/s + 把约束力移交到 pole2 |
| `FSM_FREE` | pin2 角度 > 1.0 rad | pole2 自由，**不再受约束力** |

### 3.3 关键公式链

```
约束力 (3D 笛卡尔):   F0 = efc_force[:3]                       (3, 1)
约束 Jacobian:        J = efc_J reshape → (3, 6)               (3, 6)
pole 的子块:           J_pole = J[:, :3]                         (3, 3)
pole DOF 上的力:      f_pole = J_pole.T @ F0                     (3, 1)
                       = (3, 3).T @ (3, 1) = (3, 1)
```

**核心思想**：`f_pole` 是**本来作用在 pole 上的力**，现在**直接写到 pole2 的 DOF**（`qfrc_applied[3, 4, 5]`）。

这就是「**力移交**」—— pole 被约束钉住，约束反力被「偷」过来，**转手**给 pole2。

---

## 四、「力移交」机制详解

### 4.1 为什么要做力移交？

因为 `pole` 和 `pole2` **没有真实关节连接**，但你**想让它们互动**。三种方案：

| 方案 | 实现 | 优缺点 |
|------|------|--------|
| 真实关节 | 在 XML 加 `<joint>` | 永久连接，不能分离 |
| 距离约束 | `<distance>` 等式 | 软连接，可调刚度 |
| **力移交（No.8 方案）** | 读约束力 + 写到另一 body | **可释放**，FSM 控制 |

### 4.2 为什么 `pin2` 旋转会让 pole2 受力？

直觉：

1. `pole` 被 connect 钉死在 (0, 0, 0.5)。
2. 你**强行**给 `pin` 加速（`qvel[2] → 5`），但 `pin` 旋转会**试图**让 anchor 偏离 (0, 0, 0.5)。
3. MuJoCo 算出「保持约束」需要的 3D 力 F0。
4. **`F0` 不是真的在 anchor 点上**——它在每个 DOF 上的**投影**才是 pole 实际感受到的。
5. 把这个投影**写**到 pole2 上，pole2 就「**好像被连着**」了。

### 4.3 释放时会发生什么？

```python
if fsm == FSM_SWING and data.qpos[5] > 1.0:
    fsm = FSM_FREE
```

当 `pin2` 旋转超过 1.0 rad（约 57°），FSM 切换到 `FSM_FREE`：

```python
elif fsm == FSM_FREE:
    data.qfrc_applied[3] = 0.0
    data.qfrc_applied[4] = 0.0
    data.qfrc_applied[5] = 0.0
```

**不再写力给 pole2**。此时 pole2 已经在旋转+平移中获得动能，**带着这个动量飞出去**。pole 仍然被 connect 钉住（如果 MuJoCo 的约束**没有**被释放）。

> **注意**：代码里**只**移除了 `qfrc_applied` 的力，**没有**移除 `<connect>` 约束本身（XML 没改）。所以 pole 仍然被钉，pole2 自由飞。

---

## 五、MuJoCo 版本陷阱：efc_J 形状

### 5.1 历史变化

| 版本 | `data.efc_J` 形状 | 索引方式 |
|------|------------------|---------|
| < 3.0 | `(nefc, nv)` 2D 密集 | `J[i, j]` |
| **≥ 3.0（当前 3.8.0）** | `(njmax * nv,)` 1D 平铺 | `J[i*nv + j]` |

老代码 `data.efc_J[:3, :3]` **直接报错**（`IndexError: too many indices`）。

### 5.2 正确写法（已修复）

```python
J = data.efc_J[:data.nefc * model.nv].reshape((data.nefc, model.nv))[:3, :3]
```

**解释**：
- `data.nefc` = 3（connect 给 3 个约束）
- `model.nv` = 6（总 DOF）
- `data.efc_J[:18]` 取前 18 个元素（3×6）
- `.reshape((3, 6))` 变回 2D
- `[:3, :3]` 保留原作者的语义（取 pole 的 3×3 子块）

### 5.3 如果想知道完整的 Jacobian

```python
J_full = data.efc_J[:data.nefc * model.nv].reshape((data.nefc, model.nv))
# 形状 (3, 6): 行=约束维度(x,y,z), 列=DOF
# [:, :3] = pole 的 3 个 DOF
# [:, 3:] = pole2 的 3 个 DOF
```

---

## 六、跟 No.4–No.7 的本质区别

| 维度 | No.4 反馈线性化 | No.5 PD+FSM | No.6 IK | No.7 LQR | **No.8 力移交** |
|------|----------------|------------|---------|----------|----------------|
| **物理模型** | 串联双摆 | 串联双摆 | 串联双摆 | 串联双摆 | **两个独立 body** |
| **连接方式** | 真实关节 | 真实关节 | 真实关节 | 真实关节 | **connect 等式约束** |
| **控制律** | `τ = M·v + f` | FSM + PD | `Δq = J⁻¹·Δx` | `u = K·x` | **读约束力 + 写到另一 body** |
| **需要模型吗** | 需要 M | 否 | 需要 J | 需要 A, B | **需要 efc_J, efc_force** |
| **可释放？** | ❌ | ❌（FSM 切状态） | ❌ | ❌ | **✅（FSM_FREE 切走）** |
| **核心数学** | 矩阵运算 | 时序逻辑 | Jacobian 逆 | Riccati 方程 | **约束 Jacobian 转置** |

### 控制思想的演进

```
No.4: 已知模型 → 用模型「抵消」非线性
No.5: 不知道模型 → 用大增益「硬追」
No.6: 不知道关节怎么动 → 问 Jacobian（末端映射）
No.7: 已知模型 → 线性化 + 最优控制
No.8: 已知约束 → 读约束反力 → 「借」力给另一 body
```

**No.8 的核心价值**：**不修改 XML 就能在两个 body 之间建立可释放的「虚拟连接」**。

---

## 七、运行方法

```bash
cd mujoco/No_8/
mjpython hybrid_pendulum.py
```

预期效果：
- **前段（FSM_SWING）**：pole 在原地摆动（被钉住），pole2 跟着动（接收约束反力）。
- **pin2 转过 1.0 rad**（约 1-2 秒后）：FSM 切到 FREE。
- **后段（FSM_FREE）**：pole2 自由飞行，pole 仍被钉。

---

## 八、跟 No.5 FSM 的对比

| | No.5 FSM | No.8 FSM |
|---|---------|---------|
| 状态数 | 4（HOLD/SWING1/SWING2/STOP） | 2（SWING/FREE） |
| 切换条件 | 时间触发 | 状态触发（`qpos[5] > 1.0`） |
| 切换逻辑 | 时间表 | 物理量阈值 |
| 状态行为 | 切 PD 参考 | 切是否施加约束反力 |

**No.5 是「按时间表跑任务」，No.8 是「按物理事件触发」**。

---

## 九、常见问题

### 1. `IndexError: too many indices for array: array is 1-dimensional`

**原因**：MuJoCo 3.x 把 `efc_J` 改成 1D 平铺了。

**解决**：
```python
J = data.efc_J[:data.nefc * model.nv].reshape((data.nefc, model.nv))
```

### 2. 约束力算出来是 0

**原因**：`<connect>` 的 `body2="world"` 让 anchor 钉在**世界**。如果 pole 不动（重力被某物平衡），约束力为 0。

**检查**：
```python
print("efc_force =", data.efc_force)
print("nefc =", data.nefc)
```

### 3. pole2 飞得不对（方向/角度错）

**可能原因**：力移交的**几何不严格**。代码把约束力**直接当 pole2 的关节力矩**用，**忽略了力臂**。

**严格做法**：
- 在 pole2 上定义一个等效 anchor（用 `<site>`）
- 算这个 anchor 的 Jacobian `J_anchor2`
- `qfrc_applied_pole2 = J_anchor2.T @ F0`

### 4. 怎么验证「力移交」真的发生了？

```python
# 在 controller 里加：
print("F0 =", F0.flatten())
print("JT_F =", JT_F.flatten())
print("qfrc_applied[3:6] =", data.qfrc_applied[3:6])
```

F0 非零 + JT_F 非零 → 力确实在传。

### 5. 怎么「真的」释放 pole 本身？

代码只移除了 `qfrc_applied` 的力，**没**移除 `<connect>` 约束（XML 里的 `body1="pole"` 还在）。

要真释放，**得在仿真中改 XML**（用 `mj_deleteConnection` 或等价的 Python API）—— 这是 MuJoCo 的另一个大话题（**动态模型修改**）。

### 6. 状态变量 `qpos[5]` 是哪个？

是 **pole2 的 pin2 角度**（DOF 5）。`qpos` 顺序按 XML 声明：
```
qpos[0] = pole.x
qpos[1] = pole.z
qpos[2] = pole.pin
qpos[3] = pole2.x2
qpos[4] = pole2.z2
qpos[5] = pole2.pin2   ← FSM 用这个
```

### 7. 不用约束力，直接加 `<joint>` 连两个 body 行不行？

**可以** —— 但那就不是「**可释放**」的连接了。No.8 的精妙之处就是**用约束当传感器**，**不**用真实关节，**保留「释放」的能力**。

### 8. 跟强化学习里的「虚拟力」有什么关系？

No.8 是把**约束反力**当作 RL 里的**奖励塑形信号**或**辅助控制**。MuJoCo 里这种「**借约束力**」的技巧也常用于**接触感知**（`efc_force` 反映接触力大小）。

---

## 十、整体公式对应

```
──────── 系统定义（XML）──────
两个独立 body (pole, pole2) + connect 等式约束
nefc = 3 (x, y, z), nv = 6 (3+3 个 DOF)

──────── 约束求解（mj_step）──────
约束 Jacobian: J (3×6) = ∂constraint/∂q
约束反力:      F0 (3×1) = 拉格朗日乘子
保持 anchor 钉在 (0, 0, 0.5) 所需要的力

──────── 力移交（controller）──────
J_pole = J[:, :3]              (3×3)
f_pole = J_pole.T @ F0         (3×1) ← pole 上要施加的力
data.qfrc_applied[3:6] = f_pole  ← 改写到 pole2 的 DOF

──────── FSM 状态机 ──────
FSM_SWING: 驱动 pin + 移交力 + 检查 pin2 > 1.0
FSM_FREE:  pole2 自由，不再移交

──────── 物理（mj_step）──────
重力 + 约束反力 + 用户的 qfrc_applied → 更新 qpos, qvel
```

---

## 十一、一句话总结

> **No.8 = 「读约束反力 + 写给另一 body + FSM 释放」**。这是把 MuJoCo 的等式约束**当传感器用**的经典技巧 —— 不修改 XML 也能在两个 body 之间建立**可释放**的虚拟连接。**核心数学**是 `f_pole = J_pole.T @ F0`，**核心代码**是 FSM 切换 `qfrc_applied` 的写入。
