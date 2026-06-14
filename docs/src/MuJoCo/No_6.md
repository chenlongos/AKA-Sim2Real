# No.6 双摆逆运动学（IK）

本节介绍双摆的**逆运动学（Inverse Kinematics, IK）** 控制 —— **从「指挥关节」升级到「指挥末端执行器」**。核心思想是用 Jacobian 把「末端想去哪」翻译成「关节该怎么动」。

---

## 文件说明

```
mujoco/No_6/
├── doublependulum.xml     # MuJoCo XML 模型文件
└── doublependulum_ik.py   # 完整脚本：含 IK 控制器
```

> No.6 **没有**最小脚本（`no_6.py`）。要看效果必须跑 `doublependulum_ik.py`。

---

## 一、doublependulum.xml 详解（对比 No.5）

```xml
<mujoco>
    <option timestep="0.0001" integrator="RK4" gravity="0 0 0" >  <!-- 重力关掉 -->
        <flag energy="enable" contact="disable" />
    </option>

    <worldbody>
        <light diffuse=".5 .5 0.5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

        <body pos="0 0 1.25" euler="0 90 0">  <!-- euler="0 90 0" 让关节在 (x,z) 平面内运动 -->
            <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 -0.5"/>
            <geom type="cylinder" size="0.05 0.5" rgba="0 .9 0 1" mass="1"/>
            <body pos="0 0.1 1" euler="0 0 0">
                <joint name="pin2" type="hinge" axis="0 -1 0" pos="0 0 -0.5"/>
                <geom type="cylinder" size="0.05 0.5" rgba="0 0 .9 1" mass="1"/>
                <!-- 【No.6 新增】在第二根杆末端定义「末端点」 -->
                <site name="endeff" pos="0 0 0.5" size="0.1"/>
            </body>
        </body>
    </worldbody>

    <!-- 【No.6 新增】位置伺服（代替 No.5 的 motor） -->
    <actuator>
        <position name="pservo1" joint="pin"  kp="100" />
        <velocity name="vservo1" joint="pin"  kv="10"  />
        <position name="pservo2" joint="pin2" kp="100" />
        <velocity name="vservo2" joint="pin2" kv="10"  />
    </actuator>

    <!-- 【No.6 新增】暴露末端点位置给 Python -->
    <sensor>
        <framepos    objtype="site" objname="endeff"/>
        <framelinvel objtype="site" objname="endeff"/>
    </sensor>
</mujoco>
```

### XML 配置对比表

| 配置项 | No.5 (FSM) | No.6 (IK) |
|--------|-----------|-----------|
| 重力 | 默认开启 | **`gravity="0 0 0"`（关）** |
| 朝向 | `euler="0 180 0"` | `euler="0 90 0"`（**关节在 x-z 平面**） |
| 末端标记 | ❌ | **`<site name="endeff">` 新增** |
| actuator | 2 motor + 4 servo | **2 position servo + 2 velocity servo** |
| 增益 | 无（motor） | **kp=100, kv=10**（PD 内部由 MuJoCo 算） |
| sensor | ❌ | **`framepos` + `framelinvel` 新增** |

### 关键变更说明

#### 1. `gravity="0 0 0"`：关掉重力

**为什么**？IK 任务是「末端画圆」，**不需要**与重力对抗。**纯运动学演示**让控制更干净。

#### 2. `euler="0 90 0"`：让运动平面是 (x, z)

`axis="0 -1 0"` 配合 `euler="0 90 0"` 之后，关节旋转轴变成水平方向 → 两个关节的运动**全在 (x, z) 平面内**。这让 IK 任务**降维到 2D**（x、z 两个分量），Jacobian 变成 2×2 方阵**可逆**。

#### 3. `<site name="endeff">`：定义「末端执行器」

```xml
<site name="endeff" pos="0 0 0.5" size="0.1"/>
```

`site` 是 MuJoCo 里的**虚拟标记点**，附在 body 上，不参与碰撞。`pos="0 0 0.5"` 表示在第二根 body 的**末端**（局部 z=0.5）。`size="0.1"` 是可视化半径（橙色小球）。

> **关键概念**：`site` 是「**逻辑上**的末端」，**不**是物理实体。它让控制器能问「末端在哪」「末端的 Jacobian 是什么」。

#### 4. 位置伺服代替 motor

No.5 的 `motor` 是**力矩输入**（要自己写 PD）。No.6 的 `position` servo 是**目标关节角**（MuJoCo 内部帮你做 PD）：

```
motor  + 自己写 PD      =  τ           (低层)
position servo          =  q_target    (高层)
```

**控制层级上移**：你只关心**目标位置**，不关心**怎么到位**。

#### 5. 传感器：暴露末端信息

```xml
<framepos    objtype="site" objname="endeff"/>  → sensordata[0:3] = (x, y, z)
<framelinvel objtype="site" objname="endeff"/>  → sensordata[3:6] = (vx, vy, vz)
```

`framepos` 把末端**世界坐标**暴露给 Python。这是「**仿真器作弊**」——它直接把答案告诉你（见下文）。

---

## 二、controller 详解：逆运动学核心

### 2.1 任务定义

让末端在 (x, z) 平面**画一个圆**：

```python
# 在主循环初始化时算：
mj.mj_forward(model, data)         # 算当前末端位置

x_0 = data.sensordata[0] - r       # 圆心 x = 当前末端 x - 0.5
z_0 = data.sensordata[2]           # 圆心 z = 当前末端 z

# 在 controller 里：
x_target = x_0 + r * cos(t)        # 圆参数化（周期 2π）
z_target = z_0 + r * sin(t)
```

### 2.2 完整 controller 代码

```python
def controller(model, data):
    # 1. 读末端当前位置（来自 <sensor><framepos>）
    end_eff_pos = data.sensordata[:3]

    # 2. 算末端 Jacobian（3D 位置 × 2 关节）
    jacp = np.zeros((3, 2))
    mj.mj_jac(model, data, jacp, None, end_eff_pos, 2)  # 2 = body id

    # 3. 切到 (x, z) 子空间（忽略 y），变 2×2 方阵
    J = jacp[[0, 2], :]

    # 4. 末端位置误差
    dx = np.array([
        [x_0 + r * np.cos(data.time) - data.sensordata[0]],
        [z_0 + r * np.sin(data.time) - data.sensordata[2]]
    ])

    # 5. 逆运动学：Δq = J⁻¹ · Δx
    dq = inv(J) @ dx

    # 6. 命令位置伺服：让关节去 q + Δq
    data.ctrl[0] = data.qpos[0] + dq[0, 0]   # pin
    data.ctrl[2] = data.qpos[1] + dq[1, 0]   # pin2
```

### 2.3 核心公式：Δq = J⁻¹ · Δx

**Jacobian J 是「关节速度 → 末端速度」的线性映射**：

```
ẋ_末端 = J(q) · q̇_关节
   (3D)    (3×2)  (2D)
```

**反解**：给定末端期望速度 → 求关节速度

```
q̇ = J⁻¹ · ẋ
```

No.6 取 J 的 x、z 两行 → 2×2 **方阵**可逆。

### 2.4 公式链

```
目标末端位置: x*(t) = (x_0 + r·cos t, z_0 + r·sin t)
末端误差:     Δx = x*(t) - x_current
关节修正:     Δq = J⁻¹ · Δx
目标关节角:   q_target = q_current + Δq
ctrl[0]      = q_target_pin
ctrl[2]      = q_target_pin2
内部 kp=100   → MuJoCo 帮你做 PD 到位
```

---

## 三、`mj_jac` 的两个非显然参数

```python
mj.mj_jac(model, data, jacp, None, end_eff_pos, 2)
```

| 参数位置 | 含义 | 这个值 |
|---------|------|--------|
| `jacp` | 位置 Jacobian（输出） | 3×2 零数组（被填） |
| 第 4 参数 | 旋转 Jacobian | `None`（不要） |
| `end_eff_pos` | 算 Jacobian 的**世界坐标点** | 末端当前世界坐标 |
| `2` | 这个点**附在哪个 body** | body id 2（第二根杆） |

> **易错点**：`end_eff_pos` 必须是**世界坐标**（不是 body 局部坐标）；`2` 是 body 在 MuJoCo 内部的 id（按 XML 声明顺序：world=0, 第一根杆=1, 第二根杆=2）。

---

## 四、初始化的精妙之处

```python
data.qpos[0] = -0.5
data.qpos[1] = 1.0
mj.mj_forward(model, data)            # ← 关键

x_0 = data.sensordata[0] - r          # 圆心 = 当前末端 x - 0.5
z_0 = data.sensordata[2]              # 圆心 z = 当前末端 z
```

### 为什么 `mj_forward`？

`data.sensordata` **不是**赋值 `qpos` 后立即更新的。要算「当前末端在哪」必须**先** `mj_forward`（不消耗时间的前向计算）。

### 圆心为什么这么算？

```
圆方程: x(t) = x_0 + r·cos(t),  z(t) = z_0 + r·sin(t)
t=0 时:  x(0) = x_0 + r,            z(0) = z_0
        = 当前末端位置    = 当前末端位置
```

所以**圆心 = 末端当前位置向左移 r**。这样画出来的圆**从末端当前位置出发**，自然过渡。

---

## 五、仿真器「作弊」的两处

### 5.1 第一处：直接读 sensordata

```python
end_eff_pos = data.sensordata[:3]   # ← 偷懒：直接拿答案
```

真实代码应该**自己算正运动学 (FK)**：

```python
# 自己写 FK
L1, L2 = 0.5, 0.5  # 杆长
x_ee = L1 * cos(q[0]) + L2 * cos(q[0] + q[1])
z_ee = L1 * sin(q[0]) + L2 * sin(q[0] + q[1])
end_eff_pos = np.array([x_ee, 0, z_ee])
```

**效果完全一样**（因为 sensor 内部就是这个 FK），但**不依赖 MuJoCo 内部状态**。

### 5.2 第二处：mj_jac 算 Jacobian

```python
mj.mj_jac(model, data, jacp, None, end_eff_pos, 2)  # ← MuJoCo 自动求导
```

真实代码应该**手写 Jacobian 解析式**（FK 的导数）：

```python
# 自己写 Jacobian
J = np.array([
    [-L1·sin(q1) - L2·sin(q1+q2),  -L2·sin(q1+q2)],
    [ L1·cos(q1) + L2·cos(q1+q2),   L2·cos(q1+q2)]
])
J = J[[0, 1], :]  # 提取 x, z 行
```

> **关键洞察**：**「不知道末端位置」是假问题**。知道 `q` 就能算出 `x`（FK 是确定性函数）。代码里用 `sensordata` 和 `mj_jac` 只是「**仿真器帮你算好**」的便利，**真实部署**时把这两步换成自己的 FK 和 J 即可。

---

## 六、跟 No.4/5 的本质区别

| 维度 | No.4 反馈线性化 | No.5 PD+FSM | **No.6 IK** |
|------|----------------|------------|-------------|
| **控制空间** | 关节空间 q | 关节空间 q | **任务空间 x（末端位置）** |
| **目标** | 固定点 qref | 时变轨迹 q(t) | **空间几何曲线** x(t)（圆） |
| **核心数学** | `τ = M·v + f` | FSM + PD | **`Δq = J⁻¹·Δx`** |
| **驱动** | 力矩（qfrc） | 力矩（motor） | **位置（servo）** |
| **需要模型吗** | 需要 M | 不需要 | **需要 J**（比 M 简单） |

### 控制思想的演进

```
No.4: 已知模型 → 用模型「抵消」非线性
No.5: 不知道模型 → 用大增益「硬追」
No.6: 不知道关节该怎么动 → 问 Jacobian（局部线性映射）
```

**No.6 的核心价值**：**你不用关心关节怎么动，只告诉系统「末端应该到哪」**。

---

## 七、运行方法

```bash
cd mujoco/No_6/
mjpython doublependulum_ik.py
```

预期效果：末端**绕圆心逆时针画圆**（周期 2π ≈ 6.28 秒），双臂在 (x, z) 平面内优雅地摆动。

---

## 八、常见问题

### 1. 末端画的不是圆

**原因**：Jacobian 奇异点（`det(J) = 0`）。两个杆完全伸直或完全折叠时，末端速度被「锁死」，`J⁻¹` 数值爆炸。

**解决**：
- 减小圆半径 r（让运动范围远离奇异点）
- 用 **DLS**（阻尼最小二乘）：`J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹`，牺牲精度换稳定

### 2. `mj_jac` 报 body id 错误

**原因**：body id 不是 `2`。

**检查**：
```python
print("body 0 =", model.body(0).name)  # world
print("body 1 =", model.body(1).name)  # 第一根杆
print("body 2 =", model.body(2).name)  # 第二根杆（含 site）
```

### 3. 末端从一开始就不在预期位置

**原因**：初始化时没 `mj_forward`，sensordata 还是上一步的旧值。

**解决**：在 `data.qpos = ...` 之后**立即** `mj.mj_forward(model, data)`。

### 4. `dx` 永远不收敛到 0

**原因**：
- IK 公式是「**一步**修正」（不是积分），理论上**单步**就能闭合误差
- 但因为 `data.qpos` 在 `mj_step` 过程中变化了，**单帧内** dx 不会完全为 0
- 控制器每物理步跑 10000 次（因为 timestep=0.0001），所以**视觉上**会很快收敛

### 5. 没有重力，物理意义是什么？

No.6 是**纯运动学演示**：测试 IK 数学是否正确，**不**关心物理真实性。

如果要恢复物理真实性：移除 `gravity="0 0 0"`，并把 `position servo` 换成 `motor + 自己写 PD`。

### 6. 跟 No.7 LQR 怎么选？

| 场景 | 用 IK | 用 LQR |
|------|-------|--------|
| 末端走几何路径 | ✅ 直观 | ❌ 麻烦 |
| 关节镇定到平衡点 | ❌ 不自然 | ✅ 经典用法 |
| 避障 | ✅ 显式规划 | ❌ 难 |
| 抗扰 | ❌（单步 IK 不抗扰）| ✅ 闭环稳定 |

**简单规则**：控制**末端**用 IK，**镇定关节**用 LQR。

---

## 九、整体公式对应

```
───────── 任务（任务空间）─────────
x*(t) = (x_0 + r·cos t, z_0 + r·sin t)    ← 圆参数化

───────── 误差（任务空间）─────────
Δx = x* - x_current                       ← sensordata[:3]

───────── 逆运动学（任务空间→关节空间）──
J = ∂fk/∂q                                ← 3×2
J = J[[0, 2], :]                          ← 切到 2×2
Δq = J⁻¹ · Δx                            ← IK 核心

───────── 执行（关节空间）─────────
ctrl[0] = qpos[0] + Δq[0]                ← pin 的目标角
ctrl[2] = qpos[1] + Δq[1]                ← pin2 的目标角

───────── 物理（mujoco）─────────
position servo (kp=100, kv=10)
  → 让关节实际到达 q_target
  → 末端到达 x*
```

---

## 十、一句话总结

> **No.6 = 「指挥末端而不是关节」**。核心是 Jacobian 逆 `Δq = J⁻¹·Δx` —— 把几何意图翻译成关节命令。`site` + `framepos` 让控制器能问「末端在哪」，`mj_jac` 让它能问「关节动一点末端会怎么动」。**这是运动规划的基础**。