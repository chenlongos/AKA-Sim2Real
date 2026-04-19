# 组件详解

---

## SimPage 组件

### TopDownView

俯视画布组件，展示小车、障碍物和地图。

**文件**: `pages/SimPage/TopDownView.tsx`

**功能**:
- Canvas 绑定的 2D 地图渲染
- 键盘事件监听（WASD/方向键）
- 小车位置和角度显示
- 障碍物绘制

**状态依赖**: 监听 `simCarStore` 获取小车状态

---

### RightPanel

右侧面板容器，包含第一视角视图和日志控制台。

**文件**: `pages/SimPage/RightPanel.tsx`

**子组件**:
- FirstPersonView（第一视角渲染）
- LogConsole（日志显示）

---

### InferenceControl

推理控制面板。

**文件**: `pages/SimPage/InferenceControl.tsx`

**功能**:
- 加载训练好的模型
- 单步推理测试
- 自动推理模式

---

### TrainingControl

训练和数据采集控制面板。

**文件**: `pages/SimPage/TrainingControl.tsx`

**功能**:
- Episode 管理（开始/结束/保存）
- 数据采集开关
- FPS 配置
- 训练启动/停止

---

### LogConsole

实时日志显示组件。

**文件**: `pages/SimPage/LogConsole.tsx`

**功能**:
- Socket.IO `log_message` 事件监听
- 自动滚动到底部
- 日志级别过滤（可选）

---

## RealPage 组件

### RealCameraView

浏览器摄像头访问组件。

**文件**: `pages/RealPage/RealCameraView.tsx`

**功能**:
- `navigator.mediaDevices.getUserMedia` 获取视频流
- 设备选择下拉框
- 视频预览显示

---

### RealRightPanel

真实小车右侧状态面板。

**文件**: `pages/RealPage/RealRightPanel.tsx`

**功能**:
- 小车 IP 输入
- 连接状态指示
- 电机状态显示

---

## 共享组件

### actionMapping

键盘按键到动作的映射工具。

**文件**: `pages/SimPage/actionMapping.ts`

**映射关系**:

| 按键 | 动作 |
|------|------|
| W / ↑ | forward |
| S / ↓ | backward |
| A / ← | turn_left |
| D / → | turn_right |
| Q | forward_left |
| E | forward_right |
| Z | backward_left |
| C | backward_right |
| Space | stop |
