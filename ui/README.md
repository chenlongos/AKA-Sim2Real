# AKA-Sim2Real Frontend

AKA-Sim2Real 系统的前端，基于 React + TypeScript + Vite。

## 技术栈

- **框架**: React 19 + TypeScript
- **路由**: React Router DOM 7
- **状态管理**: Zustand
- **样式**: Tailwind CSS 4
- **HTTP 客户端**: Ky
- **实时通信**: Socket.IO Client
- **构建工具**: Vite 7

## 快速开始

```bash
# 安装依赖
npm install
```
```shell
# 启动开发服务器
npm run dev
```

```shell
# 构建生产版本
npm run build
```

前端运行在 **http://localhost:5173**，会自动代理 API 请求到后端 `http://localhost:8000`。

## 页面

| 路由 | 页面 | 说明 |
|------|------|------|
| `/` | SimPage | 模拟器视角，用于数据采集与推理 |
| `/real` | RealPage | 真实小车控制接口 |

## 目录结构

```
src/
├── main.tsx              # 应用入口
├── App.tsx               # 根组件，路由配置
├── index.css             # Tailwind 入口
├── api/
│   ├── api.ts            # REST API 客户端
│   ├── socket.ts         # Socket.IO 客户端工厂
│   └── realCar.ts        # 真实小车 HTTP API
├── models/
│   └── types.ts          # 共享 TypeScript 类型
├── stores/
│   └── simCarStore.ts    # Zustand 状态管理
└── pages/
    ├── SimPage/          # 模拟器页面
    ├── RealPage/         # 真实小车页面
    └── NotFound.tsx      # 404 页面
```

## Socket.IO 命名空间

| 命名空间 | 用途 |
|----------|------|
| `/sim` | 模拟器状态同步 |
| `/real` | 真实小车控制 |

## 键盘控制（SimPage）

| 按键 | 动作 |
|------|------|
| W / ↑ | 前进 |
| S / ↓ | 后退 |
| A / ← | 左转 |
| D / → | 右转 |
| Q | 左前 |
| E | 右前 |
| Z | 左后 |
| C | 右后 |
| Space | 停止 |

## API 代理

Vite 开发服务器配置了以下代理：

- `/api` → `http://localhost:8000/api`
- `/socket.io` → `http://localhost:8000/socket.io` (WebSocket)
