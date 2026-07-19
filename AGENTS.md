# AKA-Sim2Real

基于前视视角模拟器的数据采集与 ACT (Action Chunking Transformer) 模型训练系统。

## 🛠️ 技术栈

- **后端**: FastAPI, Socket.IO, PyTorch
- **前端**: React, TypeScript, Socket.IO Client
- **文档**: mdbook

## 🚫 核心红线

- **Python 环境**: 使用 `pip`，确保 `requirements.txt` 依赖完整，引入新依赖更新`requirements.txt`
- **模型层**: `policies/models/act/` 为纯模型定义，禁止在此目录外修改模型结构
- **API 调用**: 通过显式 runtime 接口调用推理，禁止直接依赖隐式模块状态
- **测试**: 修改 ACT 相关代码后必须运行 `python3 backend/run_act_checks.py`

## ⚙️ 常用命令

```bash
# 启动后端
cd backend && python main.py

# 启动前端
cd ui && npm run dev

# 运行测试
coverage erase && python3 -m pytest tests/ --cov=backend --cov-report=html

# 打开报告网页
open htmlcov/index.html
```

## 🏗️ 架构与规范

### 目录结构

```
├── backend/                 # Python 后端 (FastAPI + Socket.IO)
│   ├── api/                # REST API 路由 (domains/inference, training, episode)
│   ├── sio_handlers/       # Socket.IO 事件处理 (sim/real 命名空间)
│   └── services/
│       ├── inference/      # checkpoint.py, preprocess.py, execution.py, runtime.py
│       ├── training/       # 训练编排
│       └── episode/        # 数据采集/导出
├── policies/models/act/    # ACT 模型 (modeling_act, configuration_act, defaults, train_act)
├── ui/                    # React 前端
├── tests/act/             # ACT 测试
└── docs/                  # mdbook 文档
```

### 代码规范

## 🧹 代码整洁规范 (Clean Code Principles)

在编写任何代码时，必须严格遵守以下 Clean Code 原则：

### 1. 命名即文档
- **意图清晰**: 变量、函数、类的命名必须揭示其意图。避免使用魔术数字和模糊缩写（如 `d` 代表 `data`，`val` 代表 `value`）。
- **可搜索性**: 避免使用魔法数字。如果必须使用，请定义为常量（如 `const DAYS_IN_WEEK = 7`）。
- **类型命名**: 类名和对象名必须是名词（`Customer`, `AccountStatus`），函数名必须是动词（`getUser`, `calculateTotal`）。

### 2. 函数设计
- **单一职责**: 一个函数只做一件事。如果一个函数超过 **20 行**（或屏幕一屏），请尝试拆分。
- **无副作用**: 尽量避免函数产生副作用（如修改全局变量、隐式修改传入的对象）。如果必须修改，请在命名中体现（如 `updateUser` vs `getUser`）。
- **命令与查询分离**: 函数要么执行操作（Command），要么返回数据（Query），不要同时做这两件事。

### 3. 注释与文档
- **代码自解释**: 优先通过重构代码（如提取函数、优化变量名）来消除对注释的需求，而不是写注释来解释烂代码。
- **禁止废话**: 不要写“获取用户 ID”这种显而易见的注释。
- **解释“为什么”**: 只有在解释复杂的业务逻辑、权衡取舍或解决特定 Bug 时才写注释（解释“为什么这么做”，而不是“做了什么”）。
- **TODO 规范**: 遗留代码必须标记 `// TODO: [描述] - [作者/日期]`。

### 4. 错误处理
- **使用异常而非返回码**: 不要通过返回 `null` 或 `-1` 来表示错误。使用 `try/catch` 或 Result/Either 模式。
- **不要返回/传递 null**: 除非是特定语言特性（如 Optional），否则不要返回 `null`。返回空列表 `[]` 或空对象。
- **精确捕获**: 不要捕获通用的 `Exception`，只捕获你明确知道如何处理的特定错误类型。

### 5. 结构与格式
- **垂直格式**: 相关的代码行应该放在一起（变量声明靠近使用处）。
- **依赖倒置**: 高层模块不应依赖低层模块，二者都应依赖抽象（接口）。
- **DRY**: 杜绝重复代码。如果你发现自己在复制粘贴代码，请立即提取为公共函数或组件。

---