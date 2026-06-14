# No.7 双摆 LQR 最优控制（含系统线性化）

本节介绍在 No.4–No.6 已有控制器的基础上，引入**最优控制**与**在线性化**技术。核心思路：
1. **线性化**：用有限差分把非线性双摆模型在平衡点附近**线性化**为 `ẋ = Ax + Bu`
2. **LQR 设计**：解连续代数 Riccati 方程 (CARE)，得到**最优反馈增益** `K`
3. **状态反馈**：控制律 `u = -Kx`（这里 K 已含负号，代码里直接 `+K@state`）
4. **鲁棒性测试**：往第一个关节注入高斯噪声，看 LQR 能否镇定

---

## 文件说明

```
mujoco/No_7/
├── doublependulum.xml      # MuJoCo XML 模型文件
└── doublependulum_lqr.py   # 完整脚本：含线性化、LQR 设计、噪声注入
```

> No.7 **没有**最小脚本（`no_7.py`）。要看效果必须跑 `doublependulum_lqr.py`。

---

## 一、doublependulum.xml 详解（对比 No.6）

### No.7 doublependulum.xml 完整代码

```xml
<mujoco>
    <!--
        timestep=0.001（比 No.4/5/6 的 0.0001 粗 10 倍）
        integrator=RK4
    -->
    <option timestep="0.001" integrator="RK4">
        <flag sensornoise="enable" contact="disable" energy="enable"/>
    </option>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

        <!--
            第一连杆 body：高度 2.5（No.4 同款），euler="0 180 0"（朝下）
            【No.7 新增】<inertial> 显式指定惯量
        -->
        <body pos="0 0 2.5" euler="0 180 0">
            <joint name="pin" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
            <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" />
            <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>

            <body pos="0 0 -1" euler="0 0 0">
                <joint name="pin2" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
                <geom type="cylinder" size=".05 .5" rgba="0 0 .9 1" />
                <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- pin 的 motor 被注释掉！只驱动 pin2 -->
        <motor name="torque2" joint="pin2" gear="1" ctrlrange="-1000 1000" ctrllimited="true"/>
    </actuator>
</mujoco>
```

### XML 配置对比表

| 配置项 | No.6 (IK) | No.7 (LQR) |
|--------|-----------|-----------|
| 关节定义 | `pin`, `pin2`（关节在杆底） | `pin`, `pin2`（**关节在杆顶**） |
| 初始 body 位置 | `(0, 0, 1.25)` | `(0, 0, 2.5)` |
| 朝向 | `euler="0 90 0"` | `euler="0 180 0"`（朝下） |
| `<inertial>` 块 | ❌（用默认） | ✅（**显式**指定 `mass=1`，`diaginertia=0.1 0.1 0.1`） |
| timestep | 0.0001 | **0.001**（粗 10 倍） |
| actuator | 4 个（2 motor + 2 servo） | **1 个**（只驱动 pin2） |
| sensor | `framepos` + `framelinvel` | ❌（**没有**） |
| `<flag sensornoise>` | ❌ | **有**（但**该属性已废弃**，见 FAQ） |
| 重力 | `gravity="0 0 0"`（关） | **默认开启** |

### 关键变更说明

#### 1. 关节位置 `pos="0 0 0.5"`（No.6 是 `-0.5`）

No.6 中关节在 body 的**底部**（局部 z=-0.5），第二根 body 接在第一根底部。  
No.7 改成关节在 body **顶部**（局部 z=0.5），第二根 body 接在第一根顶部 `pos="0 0 -1"`。

**为什么改？** 因为 LQR 是基于**线性化模型**的，关节位置的不同会让平衡点附近的 A、B 矩阵**完全不同**。No.7 的设计选择是「杆朝下、关节在顶、像倒挂的钟摆」。

#### 2. 显式 `<inertial>` 块

```xml
<inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
```

`diaginertia="0.1 0.1 0.1"` 是对角惯量张量。No.4/5/6 用默认值（MuJoCo 根据 cylinder 几何自动算），No.7 显式指定**让惯量与几何无关**——便于 LQR 线性化时数值稳定。

#### 3. 只驱动 pin2

```xml
<!-- <motor name="torque1" joint="pin" ... /> -->  <!-- 注释掉 -->
<motor name="torque2" joint="pin2" ... />
```

**只控第二个关节**。第一个关节是**被动**的（受重力、噪声扰动，但无控制输入）。

> **设计哲学**：让 LQR 「**只用一个执行器稳定整个 2 自由度系统**」——这是控制理论的经典挑战。系统是**欠驱动**（underactuated）的。

#### 4. timestep 变粗（0.0001 → 0.001）

**为什么**？LQR 在仿真里**不需要那么小的步长**——它设计时就考虑了「在平衡点附近有界」。粗 10 倍 = 仿真快 10 倍。

---

## 二、核心：状态空间与线性化（最重要的新概念）

### 2.1 状态向量

No.7 第一次明确把系统表示成**状态空间**形式：

```python
state = [q1, dq1, q2, dq2]ᵀ       # 4 维列向量
                              # 两个关节角 + 两个关节角速度
```

控制输入：

```python
u = data.ctrl[0]                # 1 维标量（pin2 的力矩）
```

### 2.2 状态导数函数 `get_dx`

```python
def get_dx(inputs):
    """
    inputs = [q1, dq1, q2, dq2, u]   (5 维)
    outputs = [dq1, ddq1, dq2, ddq2] (4 维)
    """
    data.qpos[0] = inputs[0]
    data.qvel[0] = inputs[1]
    data.qpos[1] = inputs[2]
    data.qvel[1] = inputs[3]
    data.ctrl[0] = inputs[4]

    mj.mj_forward(model, data)

    dq1 = data.qvel[0]
    dq2 = data.qvel[1]

    M = np.zeros((2, 2))
    mj.mj_fullM(model, M, data.qM)

    f = np.array([
        [0 - data.qfrc_bias[0]],
        [data.ctrl[0] - data.qfrc_bias[1]]
    ])

    ddq = inv(M) @ f

    return np.array([dq1, ddq[0, 0], dq2, ddq[1, 0]])
```

**这就是 `ẋ = f(x, u)` 的「真值」**——给定当前 `x` 和 `u`，算出状态导数 `ẋ`。

| 输入分量 | 物理含义 | MuJoCo 字段 |
|---------|---------|------------|
| `inputs[0]` | 关节 1 角 | `data.qpos[0]` |
| `inputs[1]` | 关节 1 角速度 | `data.qvel[0]` |
| `inputs[2]` | 关节 2 角 | `data.qpos[1]` |
| `inputs[3]` | 关节 2 角速度 | `data.qvel[1]` |
| `inputs[4]` | pin2 力矩 | `data.ctrl[0]` |

| 输出分量 | 物理含义 |
|---------|---------|
| `dq1` | 关节 1 角速度（`= inputs[1]`，恒等） |
| `ddq1` | 关节 1 角加速度（从 `M @ d̈q = f` 解出） |
| `dq2` | 关节 2 角速度 |
| `ddq2` | 关节 2 角加速度 |

> **关键洞察**：`dq/dt` = 角速度本身，`d(dq)/dt` = 角加速度。这俩是**不同的物理量**，但 `get_dx` 把它们一起算出来构成完整的 ẋ。

### 2.3 为什么第一关节的 `f[0]` 是 `0 - qfrc_bias[0]`？

```python
f = [[0 - qfrc_bias[0]],               # pin 没有力矩输入（motor 被注释）
     [data.ctrl[0] - qfrc_bias[1]]]   # pin2 收到 ctrl[0] - 偏置
```

MuJoCo 的动力学方程是：

```
M(q) q̈ = τ_applied + τ_bias_constraint - τ_bias
```

整理成 `M q̈ = f` 形式：

```
f = τ_applied - qfrc_bias
```

| 关节 | τ_applied | f = τ_applied - qfrc_bias |
|------|-----------|--------------------------|
| pin  | 0（无 motor）| `0 - qfrc_bias[0]` |
| pin2 | `data.ctrl[0]` | `data.ctrl[0] - qfrc_bias[1]` |

然后 `q̈ = M⁻¹ f` 解出加速度。

> **注意**：这里的 `qfrc_bias` 含重力，**所以 f 自动「扣除」了重力**——这跟 No.4 的反馈线性化一个思路。

### 2.4 数值线性化 `linearization`

```python
def linearization(pert=0.001):
    f0 = get_dx(np.zeros(5))       # 在 x=0, u=0 处求 ẋ

    Jacobians = []
    for i in range(5):
        inputs_i = np.zeros(5)
        inputs_i[i] = pert         # 扰动第 i 个分量
        jac = (get_dx(inputs_i) - f0) / pert   # 有限差分
        Jacobians.append(jac[:, np.newaxis])

    A = np.concatenate(Jacobians[:4], axis=1)  # 4×4（对 x 导数）
    B = Jacobians[-1]                          # 4×1（对 u 导数）

    return A, B
```

这是**有限差分法求 Jacobian**——把 `get_dx` 当黑箱，**数值**地求 `∂ẋ/∂x` 和 `∂ẋ/∂u`：

```
A[i, j] = (get_dx(x + pert·eⱼ, u) - get_dx(x, u))[i] / pert
B[i, 0] = (get_dx(x, u + pert) - get_dx(x, u))[i] / pert
```

**为什么在 `x=0, u=0` 处线性化？** 因为双摆**自然下垂**（`q=0`）是稳定平衡点。`ẋ₀ = 0`（不动），这就是**线性化的基准点**。

### 2.5 线性化结果：状态空间方程

```
ẋ ≈ A·x + B·u            (4 维)
   = [A]·[q1, dq1, q2, dq2]ᵀ + [B]·u
```

| 矩阵 | 形状 | 含义 |
|------|------|------|
| A | 4×4 | 状态转移：x 变化 → ẋ 变化 |
| B | 4×1 | 控制响应：u 变化 → ẋ 变化 |

**这是 LQR 设计的前提**：把非线性系统「假装」成线性系统，再设计线性控制器。

---

## 三、LQR 设计（核心算法）

### 3.1 LQR 想干什么？

LQR 找**最优增益 K**，使下面这个**二次型代价函数最小**：

```
J = ∫₀^∞ ( xᵀ Q x + uᵀ R u ) dt
       \_______________/  \_______/
            状态误差代价     控制代价
```

- **Q 大** → 重视状态误差（快速收敛，但控制信号大）
- **R 大** → 重视控制代价（节能，但响应慢）
- Q、R 选得不同 → 不同 K，**没有"正确"答案**

### 3.2 代码实现

```python
Q = np.diag([10, 10, 10, 10])    # 状态权重（q1, dq1, q2, dq2 各 10）
R = np.diag([0.1])                # 控制权重（u 一个分量 0.1）

P = solve_continuous_are(A, B, Q, R)  # ① 解 Riccati 方程
K = -inv(B.T @ P @ B + R) @ B.T @ P @ A   # ② 计算最优增益
```

#### 步骤 ①：解连续代数 Riccati 方程 (CARE)

```
Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0
```

这是关于矩阵 P 的**非线性矩阵方程**。`scipy.linalg.solve_continuous_are` 用迭代法（schur decomposition）解。

#### 步骤 ②：LQR 增益公式

```
K = -R⁻¹ Bᵀ P          （连续 LQR 公式）
  = -(Bᵀ P B + R)⁻¹ Bᵀ P A     （代码里的等价形式，避免求 R⁻¹）
```

**注意负号**：`K` 已经是「带负号」的形式了，所以控制律是 `u = K @ x`（不是 `u = -K @ x`）。

### 3.3 控制律

```python
def controller(model, data):
    state = np.array([
        [data.qpos[0]],
        [data.qvel[0]],
        [data.qpos[1]],
        [data.qvel[1]],
    ])
    data.ctrl[0] = (K @ state)[0, 0]   # u = K @ x

    # 噪声注入（鲁棒性测试）
    noise = mj.mju_standardNormal(0.0)
    data.qfrc_applied[0] = noise
```

**线性状态反馈**：把 `x` 直接乘以 `K` 矩阵就得到控制量。

### 3.4 噪声注入：鲁棒性测试

```python
noise = mj.mju_standardNormal(0.0)   # 标准正态 N(0, 1)
data.qfrc_applied[0] = noise          # 注入到 pin 关节
```

`mj.mju_standardNormal` 是 MuJoCo 自带的伪随机数生成器（基于 Box-Muller）。注入到**第一个关节**（pin，是被动的）——模拟**外部扰动**。

**为什么这样测？** 因为 LQR 在「无扰动、模型精确」下一定稳。**有扰动还稳**才证明控制器**鲁棒**。这是控制理论的标配实验。

> **特别说明**：噪声注入 `qfrc_applied[0]` 是**叠加**在系统上的，跟 LQR 控制无关。LQR 不需要知道这个噪声，靠反馈把它「压」回去。

---

## 四、与 No.4–No.6 的本质区别

| 维度 | No.4 反馈线性化 | No.5 PD+FSM | No.6 IK | **No.7 LQR** |
|------|----------------|------------|---------|--------------|
| **控制律** | `τ = M·v + f` | FSM 切换的 PD | `Δq = J⁻¹·Δx` | `u = K·x` |
| **设计方法** | 手算 + 调参 | 手算 + 调参 | 手算 + IK 公式 | **算法求解**（CARE） |
| **需要模型吗** | ✅ M, qfrc_bias | ❌ | ✅ J | ✅ A, B（要**线性化**） |
| **需要全状态吗** | 部分（q, q̇） | 部分 | 是 | **是**（x = [q, q̇, q, q̇]） |
| **稳定性保证** | 局部（精确模型时） | 无 | 局部（远离奇异点） | **全局**（线性化点附近，理论上） |
| **最优性** | ❌ | ❌ | ❌ | ✅（最小化二次代价） |
| **鲁棒性** | 差（依赖 M） | 中（大增益） | 中 | **可调**（Q/R 调节） |
| **计算复杂度** | O(n²) | O(1) | O(n³) | O(n³)（一次性离线） |

### 控制思想的演进

```
No.4: 已知模型 → 用模型「抵消」非线性
No.5: 不知道模型 → 用大增益「硬追」
No.6: 不知道关节该怎么动 → 问 Jacobian（局部线性映射）
No.7: 知道模型 → 把模型「线性化」→ 用最优控制理论设计 K
```

---

## 五、controller / get_dx / linearization 三函数协作图

```
线性化（仅在启动时跑一次）                LQR 控制（每物理步）
─────────────────────                ─────────────────
linearization()                       controller(model, data)
  │                                    │
  ├─ get_dx(0,0,0,0,0)                 ├─ 读 qpos, qvel → state (4,1)
  │    └─ 设 qpos/qvel/ctrl            │
  │    └─ mj_forward                   ├─ u = K @ state  ← 一次矩阵乘
  │    └─ 算 M, qfrc_bias              │
  │    └─ q̈ = M⁻¹ f                    └─ data.ctrl[0] = u
  │    └─ return ẋ
  │                                    └─ 注入噪声（鲁棒性测试）
  ├─ 扰动每个输入 → 重复 5 次                qfrc_applied[0] = N(0,1)
  ├─ 有限差分 → A (4×4), B (4×1)
  │
  └─ Q, R 选权重
     P = solve CARE
     K = -R⁻¹ Bᵀ P
```

---

## 六、关键设计参数与调参指南

| 参数 | 当前值 | 含义 | 怎么调 |
|------|--------|------|--------|
| `Q = diag(10, 10, 10, 10)` | 全 10 | 4 个状态分量同等重视 | **增大** → 收敛更快但 u 更大 |
| `R = 0.1` | 0.1 | 控制代价 | **增大** → u 更小但跟踪变慢 |
| `Q/R` 比 | 100 | 关键比例 | **越大越激进**，**越小越保守** |
| `timestep` | 0.001 | 仿真步长 | 线性化**假设连续时间**，dt 不影响 K，但太大会让数字不稳 |
| `pert` | 0.001 | 有限差分步长 | 太小 → 数值误差；太大 → 截断误差；**~1e-3 ~ 1e-5 都行** |

### 调参实验建议

| 想看什么 | 改什么 |
|---------|--------|
| 收敛更快 | `Q` 整体乘 2 |
| 控制更平缓 | `R` 乘 10 |
| 位置精度更高 | 角分量 `Q[0,0]`、`Q[2,2]` 调大 |
| 速度更小 | 角速度分量 `Q[1,1]`、`Q[3,3]` 调大 |
| 测试鲁棒性边界 | 噪声系数放大 10 倍 |

---

## 七、运行方法

```bash
cd mujoco/No_7/
mjpython doublependulum_lqr.py
```

预期效果：
- **无噪声情况下**：双摆稳定在自然下垂（q=0），不摆动。
- **有噪声注入**：双摆**仍然稳定**在 q=0 附近，但会有小幅抖动（被 LQR 抑制）。

> ⚠️ 启动时控制台会打印 `solve_continuous_are` 的解算过程（如果用 verbose 模式）。这是正常的——Riccati 求解是离线一次性计算，**不**影响仿真速度。

---

## 八、跟 No.4–No.6 的学习路径

```
No.1: 基础建模 + viewer 可视化
  ↓
No.2: GLFW + 鼠标交互
  ↓
No.3: 单摆 PD 闭环
  ↓
No.4: 双摆 + 反馈线性化（需 M）
  ↓
No.5: 双摆 + actuator + FSM + 三次多项式轨迹
  ↓
No.6: 双摆 + 任务空间 IK（需 J）
  ↓
No.7: 双摆 + 线性化 + LQR（需 A, B）  ← 当前
  ↓
（未来）No.8: MPC / 强化学习 / 接触 / 抓取
```

### 控制器「知识需求」演进

| 节 | 需要预先知道 |
|---|------------|
| No.4 | 动力学 M、qfrc_bias |
| No.5 | 轨迹生成、FSM 切换 |
| No.6 | 雅可比 J 的几何意义 |
| **No.7** | **状态空间、线性化、Riccati 方程** |

---

## 九、常见问题

### 1. XML 报错 `unrecognized attribute: 'sensornoise'`

**原因**：`sensornoise="enable"` 是老版本 MuJoCo 的写法，新版已移除该属性。

**解决**：删除 `<flag sensornoise="enable" contact="disable" energy="enable"/>` 中的 `sensornoise="enable"`，或整个 flag 元素简化为 `<flag energy="enable"/>`。

### 2. `solve_continuous_are` 报错 / 数值不稳定

**原因**：
- (A, B) 不可控（controllable）→ Riccati 无解
- Q、R 选得不合理（如 R=0）

**检查**：
```python
import numpy as np
controllability = np.concatenate([B, A @ B, A @ A @ B, A @ A @ A @ B], axis=1)
print("rank:", np.linalg.matrix_rank(controllability))  # 应为 4（满秩）
```

### 3. 双摆剧烈震荡不收敛

**可能原因**：
- K 算错了（行/列顺序错）
- 状态向量顺序跟 K 不匹配
- 噪声太大盖过控制

**调试**：
```python
print("K =", K)
print("state =", state.flatten())
print("u =", (K @ state)[0, 0])
```

### 4. 第一关节（pin）完全不动

**原因**：这是**预期行为**。pin 没有 motor，只有 `qfrc_applied` 注入噪声。

**意图**：让 LQR 「**只用 1 个执行器稳定 2 自由度欠驱动系统**」——经典控制难题。

### 5. 噪声注入在哪一行生效？

```python
data.qfrc_applied[0] = noise   # 注入到 pin
data.ctrl[0] = (K @ state)[0]  # LQR 控 pin2
```

`qfrc_applied` 是**叠加**到系统上的力，不经过 actuator。**LQR 不需要知道这个力**，靠反馈自然抑制。

### 6. 为什么 `get_dx` 里第一行写 `0 - qfrc_bias[0]`？

因为 pin 没有 motor（τ_applied = 0）。代入 `M q̈ = τ_applied - qfrc_bias`：
```
f[0] = 0 - qfrc_bias[0]
```

### 7. 跟 No.4 反馈线性化的本质区别

| | No.4 反馈线性化 | No.7 LQR |
|---|---|---|
| 取消非线性的方式 | 实时算 `M(q)` 并用它做补偿 | 离线线性化成 A, B，不再算 M |
| 控制律 | `τ = M·v + f`（实时）| `u = K·x`（**一次矩阵乘**）|
| 计算成本 | 每步 O(n³) | 每步 **O(n²)** |
| 全局最优？ | ❌ | ✅（线性化点附近） |

### 8. LQR 适用范围

LQR **只在线性化点附近**有效。**远离平衡点**性能会急剧下降（因为 `A·x + B·u` 不再近似 `f(x, u)`）。

**解决办法**：
- 在多个平衡点设计 LQR，**切换**（类似 No.5 的 FSM）
- 用 **LTV-LQR**（时变 LQR，每步重新线性化）
- 用 **MPC**（模型预测控制，No.8 可能涉及）

---

## 十、整体公式对应

```
────────────── 系统（物理）────────────────
M(q) q̈ + C(q, q̇) + g(q) = τ
                                       ← 真实非线性动力学
────────────── 线性化（get_dx + 数值 Jacobian）────
ẋ = A x + B u                          ← 在 x=0, u=0 处的线性近似
       x = [q1, q̇1, q2, q̇2]ᵀ (4×1)
       u = pin2 力矩 (1×1)
       A: 4×4, B: 4×1
────────────── LQR 设计（离线）─────────────
min ∫(xᵀQx + uᵀRu)dt
  s.t. ẋ = Ax + Bu
  → 解 CARE: AᵀP + PA − PBR⁻¹BᵀP + Q = 0
  → 增益 K = -R⁻¹BᵀP
────────────── 实时控制（每步）────────────
u = K @ x
data.ctrl[0] = u[0]
data.qfrc_applied[0] = noise   # 鲁棒性测试
────────────── 物理执行（mj_step）─────────
更新 qpos, qvel, x → 回到第一步
```

---

## 十一、一句话总结

> **No.7 = 「离线线性化 + LQR 最优控制 + 噪声鲁棒性测试」**。跟前几节比，最大跃迁是**把控制器设计从「手算调参」升级到「算法求解」**——Q、R 选好，自动算出最优 K。理论上的保证更强了（全局最优、闭环稳定），代价是**只在线性化点附近有效**。