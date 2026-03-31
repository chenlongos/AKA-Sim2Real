import {useEffect, useRef} from "react";
import {socket} from "../../api/socket.ts";

interface RealCameraViewProps {
    isRecording: boolean;
}

export const RealCameraView = ({isRecording}: RealCameraViewProps) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)

    // 监听后端发送的摄像头画面
    useEffect(() => {
        // 组件挂载时通知后端启动摄像头
        socket.emit("start_camera")

        const handleCameraImage = async (data: ArrayBuffer) => {
            if (!canvasRef.current) return

            const canvas = canvasRef.current
            const ctx = canvas.getContext('2d')
            if (!ctx) return

            const blob = new Blob([data])
            const bitmap = await createImageBitmap(blob)

            ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
        }

        socket.on('camera_image', handleCameraImage)

        return () => {
            socket.emit("stop_camera")
            socket.off('camera_image', handleCameraImage)
        }
    }, [])

    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2 h-full">
            <div className="font-semibold">Real Camera View</div>
            <div className="flex-1 flex items-center justify-center">
                <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    className="bg-black border border-gray-800 rounded max-w-full max-h-full"
                />
            </div>
            <div className="text-xs text-gray-500">
                {isRecording ? (
                    <span className="text-red-500 font-semibold">● Recording</span>
                ) : (
                    <span>Not recording</span>
                )}
            </div>
        </div>
    )
}