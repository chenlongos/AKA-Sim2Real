# 快速开始

本节介绍如何在本地搭建并运行 AKA-Sim2Real 系统。

---

## 环境要求

| 依赖 | 版本建议 |
|------|----------|
| Python | 3.10+ |
| Node.js | 18+ |
| PyTorch | 2.0+（支持 CUDA/MPS/CPU） |

---

## 1. 克隆仓库

```bash
git clone <仓库地址>
cd AKA-Sim2Real
```

---

## 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

> **提示**：推荐使用 conda 或 venv 创建独立 Python 环境，避免依赖冲突。

---

## 3. 安装前端依赖

```bash
cd ui
npm install
```

---

## 4. 启动后端服务

```bash
cd backend
python main.py
```

后端将运行在 **http://localhost:8000**

启动成功后，终端会输出类似以下信息：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> **注意**：后端启动时会尝试自动加载 `output/train/model.pt`，如果模型文件不存在，则以无模型状态运行（推理功能不可用，但数据采集和训练功能正常）。

---

## 5. 启动前端服务

新开一个终端窗口：

```bash
cd ui
npm run dev
```

前端将运行在 **http://localhost:5173**

---

## 6. 访问界面

在浏览器中打开 **http://localhost:5173**，可以看到：

- **Sim 页面**：模拟器视角，用于数据采集与模拟推理
- **Real 页面**：真实小车接口视图

---

## 目录结构说明

```
AKA-Sim2Real/
├── backend/          # Python 后端（FastAPI + Socket.IO）
│   ├── main.py       # 服务入口
│   ├── api/          # REST API 路由
│   ├── sio_handlers/ # Socket.IO 事件处理
│   └── services/     # 核心业务逻辑（ACT、训练、采集）
├── ui/               # React 前端
├── policies/         # ACT 模型定义与训练脚本
├── output/           # 运行时输出目录
│   ├── dataset/      # 采集的数据集
│   └── train/        # 训练输出（model.pt）
└── tests/act/        # ACT 最小回归测试
```

---

## 验证安装

运行 ACT 单元测试，验证核心链路正常：

```bash
python3 backend/run_act_checks.py
```

全部通过则说明 ACT 主链路（模型定义、推理、数据导出）运行正常。

---

## 下一步

- [数据采集](data-collection.md) — 开始录制驾驶演示数据
- [模型训练](training.md) — 训练 ACT 策略模型
- [模型推理](inference.md) — 运行自动驾驶推理
