import {forwardRef, useImperativeHandle, useRef} from "react";
import type {CarState} from "../../models/types.ts";
import {LogConsole} from "../SimPage/LogConsole.tsx";
import {RealCameraView, type CameraDeviceOption, type RealCameraViewRef} from "./RealCameraView.tsx";
import {motorStatus} from "../../api/api";

export interface RealRightPanelRef {
    getImageData: () => string | undefined;
}

interface RealRightPanelProps {
    carState: CarState;
    isRecording: boolean;
    getCurrentActions: () => string[];
    carIP: string;
    onCarIPChange: (ip: string) => void;
    carConnected: boolean;
    fpvCameraRef?: React.RefObject<RealCameraViewRef | null>;
    cameraDevices: CameraDeviceOption[];
    fpvCameraId: string;
    onFpvCameraChange: (deviceId: string) => void;
    fpvCameraError?: string;
}

export const RealRightPanel = forwardRef<RealRightPanelRef, RealRightPanelProps>(({
    carState,
    isRecording,
    getCurrentActions,
    carIP,
    onCarIPChange,
    carConnected,
    cameraDevices,
    fpvCameraId,
    onFpvCameraChange,
    fpvCameraError,
}, ref) => {
    const fpvCameraViewRef = useRef<RealCameraViewRef | null>(null);

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            return fpvCameraViewRef.current?.getImageData();
        }
    }), []);

    return (
        <div className="flex flex-col h-full gap-3 min-h-0 overflow-hidden">
            {/* 自定义滚动条样式 */}
            <style>{`
                .panel-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .panel-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                .panel-scrollbar::-webkit-scrollbar-thumb {
                    background-color: rgba(156, 163, 175, 0.4);
                    border-radius: 3px;
                }
                .panel-scrollbar::-webkit-scrollbar-thumb:hover {
                    background-color: rgba(156, 163, 175, 0.6);
                }
            `}</style>

            {/* 小车控制模块 - 固定高度，可滚动 */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 overflow-y-auto shrink-0 panel-scrollbar" style={{maxHeight: '55%'}}>
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                        <span className="w-2 h-2 bg-blue-500 rounded-full"/>
                        小车控制 & 状态
                    </h3>
                </div>

                {/* 小车IP输入 */}
                <div className="mb-3">
                    <label className="text-xs text-slate-400 mb-1.5 block">小车IP地址</label>
                    <input
                        type="text"
                        value={carIP}
                        onChange={(e) => onCarIPChange(e.target.value)}
                        placeholder="例如: 192.168.1.100"
                        className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-slate-500"
                    />
                    {carIP && (
                        <div className={`text-xs mt-1.5 flex items-center gap-1.5 ${carConnected ? 'text-emerald-400' : 'text-amber-400'}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${carConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}/>
                            {carConnected ? `已连接: http://${carIP}` : `连接中... (http://${carIP})`}
                        </div>
                    )}
                </div>

                {/* 摄像头视图 */}
                <div className="mb-3">
                    <RealCameraView
                        ref={fpvCameraViewRef}
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
                <div className="mb-3">
                    <label className="text-xs text-slate-400 mb-1.5 block">实时速度</label>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="bg-slate-900/50 rounded-lg p-3 text-center border border-slate-700">
                            <div className="text-xs text-slate-500 mb-1">左轮</div>
                            <div className="font-mono font-bold text-lg text-blue-400">
                                {carState.vel_left.toFixed(3)}
                            </div>
                            <div className="text-xs text-slate-500">m/s</div>
                        </div>
                        <div className="bg-slate-900/50 rounded-lg p-3 text-center border border-slate-700">
                            <div className="text-xs text-slate-500 mb-1">右轮</div>
                            <div className="font-mono font-bold text-lg text-emerald-400">
                                {carState.vel_right.toFixed(3)}
                            </div>
                            <div className="text-xs text-slate-500">m/s</div>
                        </div>
                    </div>
                </div>

                {/* 控制说明 */}
                <div className="mb-3 bg-slate-900/50 rounded-lg p-3 border border-slate-700">
                    <div className="text-xs font-semibold text-slate-300 mb-2">控制说明</div>
                    <div className="text-xs text-slate-400 space-y-1">
                        <div>↑ / W - 前进</div>
                        <div>↓ / S - 后退</div>
                        <div>← / A - 左转</div>
                        <div>→ / D - 右转</div>
                        <div className="pt-1.5 mt-1.5 border-t border-slate-700">
                            当前动作: <span className="text-slate-200">{getCurrentActions().join(', ') || '无'}</span>
                        </div>
                    </div>
                </div>

                {/* 获取小车状态按钮 */}
                <button
                    onClick={async () => {
                        if (!carIP) {
                            alert('请先输入小车IP')
                            return
                        }
                        try {
                            const data = await motorStatus(carIP)
                            console.log('motor_status:', data)
                        } catch (e) {
                            console.error('获取电机状态失败:', e)
                        }
                    }}
                    className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-all"
                >
                    获取小车状态
                </button>

                {/* 录制状态 */}
                <div className="mt-3 text-xs text-slate-500">
                    {isRecording ? (
                        <span className="text-red-400 flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"/>
                            录制中: 图像与轮速会一起采集
                        </span>
                    ) : (
                        '状态来源: 后端实时同步'
                    )}
                </div>
            </div>

            {/* 日志控制台 */}
            <div className="flex-1 min-h-0">
                <LogConsole className="h-full"/>
            </div>
        </div>
    );
});

RealRightPanel.displayName = "RealRightPanel";
