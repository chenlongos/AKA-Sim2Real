import {useEffect, useRef, useState} from "react";
import {socket} from "../../api/socket.ts";

interface RealCameraViewProps {
    isRecording: boolean;
}

export const RealCameraView = ({isRecording}: RealCameraViewProps) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const [imageUrl, setImageUrl] = useState<string | null>(null)

    // 监听后端发送的摄像头画面
    useEffect(() => {
        const handleCameraImage = (data: { image: string }) => {
            if (data.image) {
                setImageUrl(data.image)
            }
        }

        socket.on('camera_image', handleCameraImage)

        return () => {
            socket.off('camera_image', handleCameraImage)
        }
    }, [])

    // 渲染摄像头画面
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        if (imageUrl) {
            const img = new Image()
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
            }
            img.src = imageUrl
        } else {
            // 无画面时显示占位符
            ctx.fillStyle = '#1a1a2e'
            ctx.fillRect(0, 0, canvas.width, canvas.height)
            ctx.fillStyle = '#666'
            ctx.font = '16px monospace'
            ctx.textAlign = 'center'
            ctx.fillText('Waiting for camera...', canvas.width / 2, canvas.height / 2)
        }
    }, [imageUrl])

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