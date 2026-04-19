# 状态管理

---

## Zustand Store

使用 Zustand 进行轻量级状态管理。

### simCarStore

**文件**: `stores/simCarStore.ts`

管理模拟器小车状态。

```typescript
interface CarState {
  x: number;        // 小车 X 坐标
  y: number;        // 小车 Y 坐标
  angle: number;   // 小车角度 (弧度)
  vel_left: number;  // 左轮速度
  vel_right: number; // 右轮速度
}

const initialState: CarState = {
  x: 400,
  y: 300,
  angle: -Math.PI / 2,
  vel_left: 0,
  vel_right: 0,
};
```

### Actions

| Action | 说明 |
|--------|------|
| `setCarState(state)` | 更新小车状态 |
| `resetCarState()` | 重置为初始状态 |

---

## 页面级 State

除全局 Store 外，各页面使用 `useState` 管理本地状态。

### SimPage

| State | 类型 | 说明 |
|-------|------|------|
| `isRecording` | boolean | 是否正在录制 |
| `episodeCount` | number | Episode 数量 |
| `trainingStatus` | string | 训练状态 |
| `inferenceResult` | object | 推理结果 |

### RealPage

| State | 类型 | 说明 |
|-------|------|------|
| `carIP` | string | 小车 IP 地址 |
| `isConnected` | boolean | 连接状态 |
| `cameraDevices` | MediaDeviceInfo[] | 摄像头设备列表 |
| `motorStatus` | object | 电机状态 |
