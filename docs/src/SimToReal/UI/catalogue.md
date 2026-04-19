# UI 文档

本节介绍 AKA-Sim2Real 前端的架构、页面和组件。

---

## 页面概览

| 页面 | 路由 | 用途 |
|------|------|------|
| [SimPage](pages.md#simpage) | `/` | 模拟器视角，用于数据采集与模拟推理 |
| [RealPage](pages.md#realpage) | `/real` | 真实小车控制接口 |

---

## 技术栈

- **框架**: React 19 + TypeScript
- **路由**: React Router DOM 7
- **状态管理**: Zustand
- **样式**: Tailwind CSS 4
- **HTTP 客户端**: Ky
- **实时通信**: Socket.IO Client
- **构建工具**: Vite 7

---

## 目录结构

```
ui/src/
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

---

## 核心概念

- [页面详解](pages.md)
- [组件详解](components.md)
- [状态管理](state-management.md)
- [API 通信](api-communication.md)
