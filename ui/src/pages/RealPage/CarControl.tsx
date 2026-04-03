import type {CarState} from "../../models/types.ts";
import {RealCameraView, type CameraDeviceOption, type RealCameraViewRef} from "./RealCameraView.tsx";

interface CarControlProps {
    carState: CarState;
    isRecording: boolean;
    getCurrentActions: () => string[];
    carIP: string;
    onCarIPChange: (ip: string) => void;
    carConnected: boolean;
    fpvCameraRef?: RefObject<RealCameraViewRef | null>;
    cameraDevices: CameraDeviceOption[];
    fpvCameraId: string;
    onFpvCameraChange: (deviceId: string) => void;
    fpvCameraError?: string;
}

export const CarControl = ({
    carState,
    isRecording,
    getCurrentActions,
    carIP,
    onCarIPChange,
    carConnected,
    fpvCameraRef,
    cameraDevices,
    fpvCameraId,
    onFpvCameraChange,
    fpvCameraError,
}: CarControlProps) => {
    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-3 h-full text-gray-800 overflow-y-auto min-h-0">
            <div className="font-semibold">小车控制 & 状态</div>

            {/* 小车IP输入 */}
            <div className="border border-gray-300 rounded p-2">
                <div className="text-xs font-semibold mb-2">小车IP地址</div>
                <input
                    type="text"
                    value={carIP}
                    onChange={(e) => onCarIPChange(e.target.value)}
                    placeholder="例如: 192.168.1.100"
                    className="w-full px-2 py-1 border border-gray-300 rounded text-xs font-mono"
                />
                {carIP && (
                    <div className={`text-xs mt-1 ${carConnected ? 'text-green-600' : 'text-red-500'}`}>
                        {carConnected ? `已连接: http://${carIP}` : `连接中... (http://${carIP})`}
                    </div>
                )}
            </div>

            <div className="shrink-0">
                <RealCameraView
                    ref={fpvCameraRef}
                    title="右侧摄像头 / 第一人称"
                    description="数据采集与后续推理输入统一使用这一路画面。"
                    devices={cameraDevices}
                    selectedDeviceId={fpvCameraId}
                    onDeviceChange={onFpvCameraChange}
                    cameraError={fpvCameraError}
                    isRecording={isRecording}
                    collectTarget
                />
            </div>
            {/* 速度显示 */}
            <div className="border border-gray-300 rounded p-2">
                <div className="text-xs font-semibold mb-2">实时速度</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-blue-100 rounded p-1.5 text-center">
                        <div className="text-gray-500">左轮</div>
                        <div className="font-mono font-bold text-blue-700">{carState.vel_left.toFixed(3)}</div>
                        <div className="text-gray-400">m/s</div>
                    </div>
                    <div className="bg-green-100 rounded p-1.5 text-center">
                        <div className="text-gray-500">右轮</div>
                        <div className="font-mono font-bold text-green-700">{carState.vel_right.toFixed(3)}</div>
                        <div className="text-gray-400">m/s</div>
                    </div>
                </div>
            </div>

            {/* 控制说明 */}
            <div className="border border-gray-300 rounded p-2">
                <div className="text-xs font-semibold mb-2">控制说明</div>
                <div className="text-xs text-gray-600 space-y-1">
                    <div>↑ / W - 前进</div>
                    <div>↓ / S - 后退</div>
                    <div>← / A - 左转</div>
                    <div>→ / D - 右转</div>
                    <div className="pt-1 border-t border-gray-200">
                        当前动作: {getCurrentActions().join(', ') || '无'}
                    </div>
                </div>
            </div>

            {/* 录制状态 */}
            <div className="text-xs text-gray-500">
                {isRecording ? '录制中: 图像与轮速会一起采集' : '状态来源: 后端实时同步'}
            </div>
        </div>
    )
}
