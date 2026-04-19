# API 通信

---

## REST API

使用 Ky HTTP 客户端与后端通信。

**基础配置**: `api/api.ts`

```typescript
const api = ky.create({
  prefixUrl: '/api',
  headers: { 'Content-Type': 'application/json' },
});
```

### 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `dataset/collect` | 采集训练图片数据 |
| POST | `train` | 启动模型训练 |
| POST | `train/stop` | 停止训练 |
| POST | `act/load_trained` | 加载训练好的模型 |

---

## Socket.IO

实时双向通信，用于模拟器状态同步和日志推送。

### Socket 工厂

**文件**: `api/socket.ts`

```typescript
createSocket(namespace: string): Socket
```

创建命名空间的 Socket 实例：

| 实例 | 命名空间 | 用途 |
|------|----------|------|
| `simSocket` | `/sim` | 模拟器状态同步 |
| `realSocket` | `/real` | 真实小车控制 |

### 事件列表

#### Emit 事件 (前端 → 后端)

| 事件 | 数据格式 | 说明 |
|------|----------|------|
| `action` | `{ action: string }` | 发送动作指令 |
| `reset_car_state` | - | 重置小车状态 |
| `get_car_state` | - | 请求当前状态 |
| `collect_data` | `{ image: string, state: object }` | 采集数据 |
| `start_episode` | `{ episodeId: number }` | 开始 Episode |
| `end_episode` | `{ episodeId: number }` | 结束 Episode |
| `finalize_episode` | `{ episodeId: number }` | 保存 Episode |
| `act_infer` | `{ image: string, state: object }` | 推理请求 |

#### Listen 事件 (后端 → 前端)

| 事件 | 数据格式 | 说明 |
|------|----------|------|
| `connected` | - | 连接成功 |
| `car_state_update` | `CarState` | 小车状态更新 |
| `collection_count` | `{ count: number }` | 采集数量 |
| `episode_info` | `EpisodeInfo` | Episode 信息 |
| `training_progress` | `{ epoch: number, loss: number }` | 训练进度 |
| `act_infer_result` | `{ action: string }` | 推理结果 |
| `log_message` | `{ level: string, message: string }` | 日志消息 |

---

## 真实小车 HTTP API

**文件**: `api/realCar.ts`

直接向小车 IP 发送 HTTP 请求。

### 接口列表

| 函数 | HTTP 方法 | 路径 | 说明 |
|------|-----------|------|------|
| `carHeartbeat` | GET | `/api/heartbeat` | 发送心跳 |
| `motorStatusAt` | GET | `/api/motor/status` | 获取电机状态 |
| `motorDirect` | POST | `/api/motor/direct` | 直接控制电机 |
| `carControl` | POST | `/api/car/control` | 小车整体控制 |
| `carTimeSync` | GET | `/api/time/sync` | 时间同步 |

### 使用示例

```typescript
import { carControl, motorDirect } from '@/api/realCar';

// 控制小车前进
await carControl(carIP, 'forward');

// 直接控制电机速度
await motorDirect(carIP, 100, 100);
```
