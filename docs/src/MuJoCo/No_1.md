# No.1 第一个 MuJoCo 仿真

本节通过一个最小示例，帮助你快速上手 MuJoCo XML 建模与 Python 仿真脚本。

---

## 文件说明

本节的示例文件位于 `mujoco/No_1/` 目录下：

```
mujoco/No_1/
├── hello.xml   # MuJoCo XML 场景描述文件
└── no_1.py     # Python 仿真脚本
```

---

## hello.xml 详解

### 完整内容

```xml
<mujoco>
    <!-- 全局仿真选项：重力加速度 -->
    <option gravity="0 0 -9.81"/>

    <!-- 编译器设置：角度单位使用弧度 -->
    <compiler angle="radian"/>

    <!-- 视觉渲染设置 -->
    <visual>
        <headlight ambient="0.1 0.1 0.1"/>
    </visual>

    <!-- 资产定义：可复用的材质 -->
    <asset>
        <material name="white" rgba="1 1 1 1"/>
    </asset>

    <!-- worldbody: 物理世界中的所有物体 -->
    <worldbody>
        <!-- 场景光源 -->
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

        <!-- 地面（红色平面） -->
        <geom type="plane" size="1 1 0.5" rgba=".9 0 0 1"/>

        <!-- 物体 1：白色盒子，z=1 -->
        <body pos="0 0 1" euler="0 0 0">
            <joint type="free"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="box" size=".1 .2 .3" material="white"/>
        </body>

        <!-- 物体 2：青色盒子，z=2，pitch=90°（侧立） -->
        <body pos="0 0 2" euler="0 90 0">
            <joint type="free"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="box" size=".1 .2 .3" rgba="0 .9 .9 1"/>
        </body>

        <!-- 物体 3：灰色球体，z=3 -->
        <body pos="0 0 3" euler="0 0 0">
            <joint type="free"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="sphere" size=".1" rgba=".5 .5 .5 1"/>
        </body>
    </worldbody>
</mujoco>
```

### 元素说明

#### `<option>`
全局仿真选项。
- `gravity="0 0 -9.81"`：标准重力加速度 9.81 m/s²（向下）

#### `<compiler>`
编译器设置。
- `angle="radian"`：角度单位使用弧度（默认是角度），这样 euler 值可以直接写数值如 `90` 表示 90°

#### `<visual>`
渲染视觉设置。
- `headlight`：场景主光源的亮度

#### `<asset>`
可复用的资产定义。
- `material`：定义可复用的材质（name="white"，rgba 白色），在 geom 中通过 `material="white"` 引用

#### `<worldbody>`
物理世界的根容器，包含所有物体。

#### `<light>`
场景光源。
- `diffuse`：漫反射颜色（灰白色）
- `pos`：光源位置（上方 3 米）
- `dir`：光照方向（沿 -z，即向下）

#### `<geom>` 地面
- `type="plane"`：平面
- `size="1 1 0.5"`：半尺寸（x=1, y=1, z=0.5 → 全尺寸 2×2×1 米）
- `rgba=".9 0 0 1"`：红色

#### `<body>`
刚体，可内嵌关节、惯性、几何体。

| 属性 | 说明 |
|------|------|
| `pos` | 初始位置（世界坐标系） |
| `euler` | 初始欧拉角（roll pitch yaw，弧度） |

三个刚体的配置：

| 物体 | pos | euler | geom 类型 | 颜色 |
|------|-----|-------|-----------|------|
| 物体 1 | z=1 | 0 0 0（水平） | box | 白色（material） |
| 物体 2 | z=2 | 0 90 0（侧立） | box | 青色 |
| 物体 3 | z=3 | 0 0 0 | sphere | 灰色 |

#### `<joint>`
关节，连接 body 与父级（或世界）。
- `type="free"`：自由关节，物体不受任何约束，可在 6 个自由度上自由运动（平移 + 旋转）

#### `<inertial>`
刚体的质量分布特性。

| 属性 | 说明 |
|------|------|
| `pos` | 质心在 body 本地坐标系中的位置 |
| `mass` | 质量（kg） |
| `diaginertia` | 惯性张量的三个主对角元素（Ixx, Iyy, Izz） |

> **提示**：如果不手动指定 `inertial`，MuJoCo 会根据 geom 的形状和默认密度自动计算。

#### `<geom>` 物体几何体
- `type`：形状类型，支持 `box`、`sphere`、`cylinder`、`capsule`、`plane`、`ellipsoid` 等
- `size`：半尺寸向量
  - box：`size="0.1 0.2 0.3"` → 全尺寸 0.2×0.4×0.6 米
  - sphere：`size="0.1"` → 半径 0.1 米
- `rgba`：颜色 + 透明度（RGBA，各通道 0~1）
- `material`：引用 `<asset>` 中定义的材质（优先于 rgba）

---

## no_1.py 详解

```python
import mujoco
import mujoco.viewer

# 1. 从 XML 文件加载模型
model = mujoco.MjModel.from_xml_path('hello.xml')

# 2. 创建仿真数据容器
data = mujoco.MjData(model)

# 3. 启动交互式查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 4. 主循环：每帧推进一次仿真
    while viewer.is_running():
        mujoco.mj_step(model, data)   # 执行一步仿真（默认 dt=0.002s）
        viewer.sync()                  # 同步查看器显示
```

| 步骤 | 说明 |
|------|------|
| `MjModel.from_xml_path()` | 解析 XML 构建物理模型 |
| `MjData` | 存储仿真运行时数据（位置、速度、力等） |
| `mj_step()` | 推进一个仿真时间步 |
| `viewer.sync()` | 将仿真状态同步到可视化窗口 |

---

## 运行方法

在 `mujoco/No_1/` 目录下执行：

```bash
mjpython no_1.py
```

运行效果：三个物体同时从不同高度自由下落，落到红色地面上后弹起/静止。

---

## 常见错误

### 1. `ValueError: XML Error: Schema violation: unrecognized element`
**原因**：XML 中有拼写错误的标签名，如 `<gemo>`、`<intertial>`、`<muhoco>`。

**解决**：检查并修正标签拼写，确认是 `<geom>`、`<inertial>`、`<mujoco>`。

### 2. 物体没有出现或直接穿透地面
**原因**：`inertial` 未指定且 `geom` 没有定义时，MuJoCo 可能使用了零质量。

**解决**：确保每个 body 下都有 `<inertial mass="..."/>` 或让 geom 的密度足够大。

### 3. `mjpython: command not found`
**原因**：`mujoco` 包未正确安装，或 `mjpython` 不在 PATH 中。

**解决**：
```bash
pip install mujoco
# 或直接用 python 运行
python no_1.py
```
