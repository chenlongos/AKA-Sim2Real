# iframe 教学平台事件上报设计

## 背景

线上教学平台以 iframe 形式嵌入 SimPage (`/`) 和 RealPage (`/real`)，需要采集学员在页面中的操作行为、数据采集过程和训练结果，以便教学平台后台进行课程概览、学员学习情况统计和实验详情查看。

## 通信方式

**单向 postMessage**：iframe (子页面) → 父页面 (教学平台)。

- 子页面负责上报事件，不接收父页面下发数据
- 父页面负责监听 `message` 事件，关联 userId/lessonId/token，存入数据库
- 不在 iframe 中时，上报模块自动静默，不影响独立使用

## 新增文件

```
ui/src/services/
└── reportService.ts    # 上报服务
```

## 事件体系

所有消息通过 `window.parent.postMessage()` 发送，统一信封格式：

```typescript
interface LessonEvent {
  namespace: "aka-sim-lesson";  // 父页面过滤标识
  type: LessonEventType;        // 事件类型
  timestamp: number;            // 事件发生时间 (ms)
  page: "sim" | "real";        // 来源页面
  data?: Record<string, unknown>; // 事件负载
}
```

不在 iframe 中时，`send()` 为空操作。

### 事件类型清单

#### 操作事件 (action.*)

| 事件 | 触发时机 | 携带数据 |
|------|---------|---------|
| `action.enter` | 页面挂载 useEffect | page |
| `action.leave` | 页面卸载 cleanup | page |
| `action.start_collection` | 点击"开始采集" | episode_id, dataset_name, fps |
| `action.end_collection` | 点击"结束采集" | episode_id, frame_count, duration_ms |
| `action.start_training` | 点击"开始训练" | dataset_name, epochs, batch_size, lr, resume |
| `action.stop_training` | 点击"停止训练" | current_epoch, current_loss |
| `action.load_model` | 模型加载成功 | model_name |
| `action.switch_episode` | 切换采集轮次 | from_episode, to_episode |
| `action.reset_scene` | 点击"复位场景" | — |
| `action.connect_car` | 小车 IP 连接成功 (仅 RealPage) | car_ip |

#### 采集详情 (collection.*)

通过 Socket.IO 事件自动触发，反映后端确认的权威状态。

| 事件 | Socket.IO 事件 | 携带数据 |
|------|---------------|---------|
| `collection.episode_started` | `episode_started` | episode_id, task_name, dataset_name |
| `collection.episode_ended` | `episode_ended` | episode_id, frame_count, duration_ms |
| `collection.episode_finalized` | `episode_finalized` | episode_id, frame_count, output_path |

#### 训练详情 (training.*)

通过 `training_progress` Socket.IO 事件触发。

| 事件 | 触发条件 | 携带数据 |
|------|---------|---------|
| `training.epoch_progress` | `is_running=true` 且 progress < 1 | epoch, total_epochs, loss, progress |
| `training.completed` | `is_running=false` 且 progress >= 1 | total_epochs, final_loss, dataset_path, model_path |
| `training.stopped` | 用户点击"停止训练" | ended_epoch, last_loss |

### 不包含的内容

- 推理相关事件（单次推理、自动推理的动作输出等）
- 每一帧的采集图像数据
- 键盘实时操控流

## reportService 接口

```typescript
// 初始化：传入 page 标识，检测是否在 iframe 中
function init(page: "sim" | "real"): void

// 上报事件，不在 iframe 中时静默忽略
function send(type: LessonEventType, data?: Record<string, unknown>): void

// 内部维护 page 和 iframe 检测结果，无需调用方关注
```

## 页面注入点

### SimPage (`src/pages/SimPage/index.tsx`)

在每个操作回调中插入上报调用：

| 位置 | 调用 |
|------|------|
| useEffect 初始化 | `reportService.send("action.enter")` |
| useEffect cleanup | `reportService.send("action.leave")` |
| `handleStartEpisode` | `reportService.send("action.start_collection", { episode_id, dataset_name, fps })` |
| `handleEndEpisode` | `reportService.send("action.end_collection", { episode_id, frame_count })` |
| `handleStartTraining` | `reportService.send("action.start_training", { dataset_name, epochs, ... })` |
| `handleStopTraining` | `reportService.send("action.stop_training", { epoch, loss })` |
| `handleLoadModel` (成功后) | `reportService.send("action.load_model", { model_name })` |
| `handleSetEpisode` | `reportService.send("action.switch_episode", { from, to })` |
| 复位按钮回调 | `reportService.send("action.reset_scene")` |
| Socket `episode_started` 监听 | `reportService.send("collection.episode_started", { ... })` |
| Socket `episode_ended` 监听 | `reportService.send("collection.episode_ended", { ... })` |
| Socket `episode_finalized` 监听 | `reportService.send("collection.episode_finalized", { ... })` |
| Socket `training_progress` 监听 | 根据状态区分 `training.epoch_progress` / `training.completed` |

`training.completed` 中的 `dataset_path` 和 `model_path` 在 `handleStartTraining` 中通过 `getDatasetPath()` / `getModelPath()` 计算，用 useRef 暂存。

### RealPage (`src/pages/RealPage/index.tsx`)

与 SimPage 对称，注入点相同。差异：

- `page` 字段值为 `"real"`
- 额外事件：小车 IP 连接成功时上报 `action.connect_car`

## 父页面接收协议

```javascript
window.addEventListener("message", (event) => {
  if (event.data?.namespace !== "aka-sim-lesson") return;

  const { type, timestamp, page, data } = event.data;

  // 父页面自行关联 userId / lessonId / token
  // 存入数据库，按 type 分组做展示
});
```

## 不影响现有功能

- `reportService.send()` 内部通过 `window.self !== window.top` 检测 iframe 环境
- 独立打开页面时所有 `send()` 调用静默忽略
- 不修改现有 Socket.IO、REST API、Zustand store 的任何逻辑
- `TrainingControl` / `InferenceControl` 组件不需要改动（上报在页面层完成）
