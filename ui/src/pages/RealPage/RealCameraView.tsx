import {forwardRef, useEffect, useImperativeHandle, useRef} from "react";

export interface CameraDeviceOption {
    deviceId: string;
    label: string;
}

interface RealCameraViewProps {
    title: string;
    description: string;
    isRecording: boolean;
    devices: CameraDeviceOption[];
    selectedDeviceId: string;
    onDeviceChange: (deviceId: string) => void;
    cameraError?: string;
    collectTarget?: boolean;
}

export interface RealCameraViewRef {
    getImageData: () => string | undefined;
}

export const RealCameraView = forwardRef<RealCameraViewRef, RealCameraViewProps>(({
    title,
    description,
    isRecording,
    devices,
    selectedDeviceId,
    onDeviceChange,
    cameraError,
    collectTarget = false,
}, ref) => {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            const video = videoRef.current;
            const canvas = captureCanvasRef.current;
            if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) return undefined;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            if (!ctx) return undefined;

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', 0.8);
        }
    }), []);

    useEffect(() => {
        const video = videoRef.current;
        if (!video || !selectedDeviceId) return;

        let activeStream: MediaStream | null = null;
        let cancelled = false;

        const startStream = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: {exact: selectedDeviceId},
                        width: {ideal: 1280},
                        height: {ideal: 720},
                    }
                });
                if (cancelled) {
                    stream.getTracks().forEach((track) => track.stop());
                    return;
                }
                activeStream = stream;
                video.srcObject = stream;
                await video.play().catch(() => {});
            } catch {
                if (video.srcObject) {
                    const currentStream = video.srcObject as MediaStream;
                    currentStream.getTracks().forEach((track) => track.stop());
                    video.srcObject = null;
                }
            }
        };
        startStream();

        return () => {
            cancelled = true;
            if (activeStream) {
                activeStream.getTracks().forEach((track) => track.stop());
            }
            if (video.srcObject) {
                const currentStream = video.srcObject as MediaStream;
                currentStream.getTracks().forEach((track) => track.stop());
                video.srcObject = null;
            }
        };
    }, [selectedDeviceId]);

    return (
        <div className="flex flex-col gap-3 h-full">
            <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                    <span className="w-2 h-2 bg-cyan-500 rounded-full"/>
                    {title}
                </h3>
                {collectTarget && (
                    <div className="text-xs font-semibold text-cyan-400 px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30">
                        采集源
                    </div>
                )}
                {isRecording && (
                    <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                        <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"/>
                        <span className="text-xs font-semibold">REC</span>
                    </div>
                )}
            </div>

            <select
                value={selectedDeviceId}
                onChange={(e) => onDeviceChange(e.target.value)}
                className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-slate-500 disabled:opacity-50"
                disabled={devices.length === 0}
            >
                {devices.length === 0 ? (
                    <option value="">未检测到摄像头</option>
                ) : null}
                {devices.map((device) => (
                    <option key={device.deviceId} value={device.deviceId}>
                        {device.label}
                    </option>
                ))}
            </select>

            <div className="flex-1 min-h-0">
                <div className="h-full min-h-[220px] w-full overflow-hidden rounded-lg border border-slate-700 bg-black shadow-lg">
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="block h-full w-full object-contain"
                    />
                </div>
            </div>

            <canvas ref={captureCanvasRef} className="hidden"/>

            <div className="text-xs text-slate-500 space-y-1">
                <div>{description}</div>
                {cameraError && <div className="text-red-400">{cameraError}</div>}
                {!isRecording && <div>Not recording</div>}
            </div>
        </div>
    );
});

RealCameraView.displayName = "RealCameraView";
