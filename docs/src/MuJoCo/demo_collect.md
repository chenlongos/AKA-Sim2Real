# 移动操作机器人 — 统一 Demo + 数据采集

`demo_collect.py` 把 No.1–No.13 全部概念融合到一个文件里：含 4-DOF 机械臂的小车自动完成"捡地上的盒子 → 运回来 → 放下"，并采集训练数据。

---

## 文件说明

```
mujoco/Chenlong_Robot/
├── car.xml           # MuJoCo 模型：小车 + 2× 缩放臂 + 双指夹爪 + 28 路传感器
├── demo_collect.py   # 统一 Demo + 数据采集（No.1–No.13）
└── episodes/         # 采集的 .npz 数据 + .png 总结图
```

---

## 运行

```bash
# GUI 窗口模式（默认 5 轮）
python3 demo_collect.py

# 无头采集（指定轮数）
python3 demo_collect.py --headless --episodes 10

# 单轮
python3 demo_collect.py --episodes 1
```

### 按键

| 按键 | 功能 |
|------|------|
| 鼠标拖拽 | 旋转 / 平移 / 缩放 |
| Backspace | 重置任务 |
| S | 立即保存数据 |
| Q / Esc | 退出 |

---

## No.1–No.13 概念覆盖

| 编号 | 概念 | 文件中的实现 |
|------|------|-------------|
| No.1 | 基础仿真循环 | `MjModel.from_xml_path` + `mj_step` |
| No.2 | GLFW 渲染 | 完整窗口 + `MjvCamera` + `MjvScene` + 鼠标回调 |
| No.3 | 位置伺服 (PD) | `<position>` actuator 控制 6 个臂关节 |
| No.4 | 动力学提取 | `mj_fullM` + `qfrc_bias`，每次 FSM 切换时打印质量矩阵 |
| No.5 | FSM + 三次轨迹 | 9 状态 FSM + cubic polynomial (`_cubic_coeffs`) |
| No.6 | Jacobian IK | `mj_jac` 计算末端雅可比 + 伪逆求解 |
| No.7 | 状态反馈控制 | 车用比例速度控制，臂用当前 qpos 线性化 |
| No.8 | 约束管理 | 激活/停用 `grasp_weld` equality 实现抓取 |
| No.9 | 位置触发转换 | FSM 由 car_x、EE 距离、timeout 触发 |
| No.11 | 数值优化 IK | Jacobian 阻尼伪逆迭代 |
| No.12 | 独立 FK + 绘图 | 独立 `MjData` 做 FK + matplotlib 总结图 |
| No.13 | 姿态估计 | `xquat` 提取车身四元数 + 相机跟踪 |

---

## FSM 任务流程

```
DRIVE → REACH → LOWER → GRASP → LIFT → DRIVE_BACK → PLACE → RELEASE → DONE
 开车       伸手     下放      夹      抬      开回来      放       松       结束
```

每个状态有时限，超时自动跳下一状态。GRASP 时激活 weld 约束固定盒子，RELEASE 时释放。

---

## 数据采集格式 (`.npz`)

| 键 | 形状 | 内容 |
|----|------|------|
| `joint_states` | (T, 13) | 车 7D(位姿+四元数) + 臂 6D(pan/lift/elbow/wrist/finger_l/finger_r) |
| `ee_position` | (T, 3) | 末端执行器世界坐标 |
| `target_position` | (T, 3) | 盒子世界坐标 |
| `actions` | (T, 10) | 控制信号：4 轮 ctrl + 6 臂 ctrl |
| `sensordata` | (T, 28) | 全部传感器原始读数 |
| `fsm_state` | (T,) | FSM 阶段 (0–8) |
| `timestamps` | (T,) | 仿真时间 (s) |

采样 10 Hz。

---

## 传感器清单（28 维）

| 传感器 | 数量 | 说明 |
|--------|------|------|
| `arm_*_pos` | 6 | 臂关节角度 (pan/lift/elbow/wrist/finger_l/finger_r) |
| `arm_*_vel` | 6 | 臂关节角速度 |
| `wheel_*_pos` | 4 | 轮子角度 |
| `wheel_*_vel` | 4 | 轮子角速度 |
| `ee_pos` | 3 | 末端世界坐标 (xyz) |
| `ee_vel` | 3 | 末端世界速度 |
| `finger_l_touch` | 1 | 左指触觉力 |
| `finger_r_touch` | 1 | 右指触觉力 |

---

## 整体架构

```
常量(臂长/FSM枚举/控制增益)
  → 模型ID缓存 (_cache_ids: body/site/joint地址)
  → 状态读取 (_car_pos, _arm_qpos, _ee_pos, ...)
  → IK求解 (解析2D → Jacobian数值细调)
  → 三次轨迹生成 (_cubic_coeffs / _eval_cubic)
  → 动力学诊断 (_log_mass_matrix_diag)
  → 抓取控制 (_grasp: activate/deactivate weld)

DataCollector  →  10Hz 采样 →  .npz 保存
SimState       →  9 状态 FSM  + 轨迹缓存 + IK 缓存
_check_transitions  →  位置/时间驱动的状态切换
controller          →  每步调用: 车比例驱动 + 臂轨迹跟踪 + 数据记录 + FSM 转换
main                →  GLFW 窗口 + 渲染循环 + 多轮自动重置
```

---

## 臂模型 (car.xml)

```
arm_base (转台, shoulder_pan ±360°)
  └── upper_arm (0.50m, shoulder_lift ±360°)
        └── forearm (0.40m, elbow ±360°)
              └── wrist (0.16m, wrist_pitch ±360°)
                    └── gripper_palm (掌)
                          ├── finger_l (slide 0→0.04)
                          └── finger_r (slide 0→0.04, 镜像)
```

- 总臂展: ~1.06m
- 夹爪: 张开 14cm, 闭合 6cm（挤压 8cm 方块形成摩擦抓取）
- 运输时 weld 约束保证不掉
