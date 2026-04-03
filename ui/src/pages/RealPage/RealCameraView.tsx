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
    const videoRef = useRef<HTMLVideoElement | null>(null)
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            const video = videoRef.current
            const canvas = captureCanvasRef.current
            if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) return undefined

            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            const ctx = canvas.getContext('2d')
            if (!ctx) return undefined

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
            if (!canvas) return undefined
            return canvas.toDataURL('image/jpeg', 0.8)
        }
    }), [])

    useEffect(() => {
        const video = videoRef.current
        if (!video || !selectedDeviceId) return

        let activeStream: MediaStream | null = null
        let cancelled = false

        const startStream = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: {exact: selectedDeviceId},
                        width: {ideal: 1280},
                        height: {ideal: 720},
                    }
                })
                if (cancelled) {
                    stream.getTracks().forEach((track) => track.stop())
                    return
                }
                activeStream = stream
                video.srcObject = stream
                await video.play().catch(() => {})
            } catch {
                if (video.srcObject) {
                    const currentStream = video.srcObject as MediaStream
                    currentStream.getTracks().forEach((track) => track.stop())
                    video.srcObject = null
                }
            }
        }
        startStream()

        return () => {
            cancelled = true
            if (activeStream) {
                activeStream.getTracks().forEach((track) => track.stop())
            }
            if (video.srcObject) {
                const currentStream = video.srcObject as MediaStream
                currentStream.getTracks().forEach((track) => track.stop())
                video.srcObject = null
            }
        }
    }, [selectedDeviceId])

    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2 h-full">
            <div className="flex items-center justify-between gap-3">
                <div className="font-semibold">{title}</div>
                {collectTarget ? (
                    <div className="text-[11px] font-semibold text-blue-700">采集源</div>
                ) : null}
            </div>
            <select
                value={selectedDeviceId}
                onChange={(e) => onDeviceChange(e.target.value)}
                className="w-full px-2 py-1 border border-gray-300 rounded text-xs bg-white"
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
            <div className="flex-1 flex items-center justify-center">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="bg-black border border-gray-800 rounded max-w-full max-h-full w-full h-full object-contain"
                />
            </div>
            <canvas ref={captureCanvasRef} className="hidden"/>
            <div className="text-xs text-gray-500">
                <div>{description}</div>
                {cameraError ? <div className="text-red-500">{cameraError}</div> : null}
                {isRecording ? (
                    <span className="text-red-500 font-semibold">● Recording</span>
                ) : (
                    <span>Not recording</span>
                )}
            </div>
        </div>
    )
})

RealCameraView.displayName = "RealCameraView"
