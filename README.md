# AKA-Sim 模拟器

基于前视视角的自动驾驶数据采集与ACT模型训练系统。

## 功能

- 前视视角模拟器（俯视图 + 第一人称伪3D视角）
- 键盘控制（WASD/方向键）
- 数据采集用于ACT训练
- 模型训练与推理

## 快速启动

### 1. 安装依赖

```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../ui
npm install
```

### 2. 启动后端

```bash
cd backend
python main.py
```

后端运行在 http://localhost:8000

### 3. 启动前端

```bash
cd ui
npm run dev
```

前端运行在 http://localhost:5173

## 使用流程

### 数据采集

1. 在模拟器中使用 WASD 或方向键控制小车
2. 点击"开始采集"按钮开始记录数据
3. 采集完成后点击"停止采集"（自动导出到 `output/dataset/`）

### 模型训练

1. 确保已采集足够数据（至少100个样本）
2. 点击"开始训练"按钮
3. 训练完成后模型保存在 `output/train/model.pt`

### 模型推理

1. 点击"加载模型"加载训练好的模型
2. 点击"自动推理"让小车自主行动

## ACT 结构

当前 ACT 相关代码已经按职责拆分为几层：

- `policies/models/act/`
  - 纯模型定义、配置、默认参数与训练脚本
- `backend/services/act_checkpoint.py`
  - checkpoint 解析、模型实例化、stats 加载
- `backend/services/act_preprocess.py`
  - 图像预处理、状态归一化、动作反归一化
- `backend/services/act_execution.py`
  - 在线 temporal ensembling / action chunk 融合策略
- `backend/services/act_model.py`
  - `ACTInferenceRuntime` 运行时装配层
- `backend/api/__init__.py` 与 `backend/sio_handlers/__init__.py`
  - 通过显式 runtime 接口调用推理，不再直接依赖隐式模块状态

## ACT 检查

统一检查入口：

```bash
python3 backend/run_act_checks.py
```

该入口当前会执行：

```bash
python3 -m pytest tests/act -q
```

当前会顺序运行：

- `tests/act/test_model_minimal.py`
- `tests/act/test_runtime_backend.py`
- `tests/act/test_data_export_stats.py`
- `tests/act/test_end_to_end_smoke.py`

如果这一入口通过，说明当前 ACT 主链路至少在最小工作流上是闭合的。

## 目录结构

```
├── backend/           # 后端代码
│   ├── main.py        # 入口
│   ├── api/           # REST API
│   ├── sio_handlers/  # Socket.IO 事件处理
│   └── services/      # ACT runtime / 训练 / 导出
├── tests/
│   └── act/           # ACT 最小回归与 smoke tests
├── ui/               # 前端代码
├── output/           # 输出目录
│   ├── dataset/     # 采集的数据
│   └── train/       # 训练模型
│       └── model.pt
└── README.md
```

## 技术栈

- **后端**: FastAPI, Socket.IO, PyTorch
- **前端**: React, TypeScript, Socket.IO Client
