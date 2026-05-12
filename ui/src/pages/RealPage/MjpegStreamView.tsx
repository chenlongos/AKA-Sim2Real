import {forwardRef, useImperativeHandle, useRef, useEffect, useState, useCallback} from "react";

export interface MjpegStreamViewRef {
    getImageData: () => string | undefined;
}

interface MjpegStreamViewProps {
    carIP: string;
    title: string;
    description: string;
    isRecording: boolean;
    collectTarget?: boolean;
}

export const MjpegStreamView = forwardRef<MjpegStreamViewRef, MjpegStreamViewProps>(({
    carIP,
    title,
    description,
    isRecording,
    collectTarget = false,
}, ref) => {
    const imgRef = useRef<HTMLImageElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [error, setError] = useState<string>();
    const [ready, setReady] = useState(false);
    const streamUrl = `http://${carIP}/api/camera/stream`;

    const captureFrame = useCallback(() => {
        const img = imgRef.current;
        const canvas = canvasRef.current;
        if (!img || !canvas || img.naturalWidth === 0) return undefined;

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return undefined;

        ctx.drawImage(img, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.8);
    }, []);

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            if (!ready) return undefined;
            return captureFrame();
        }
    }), [ready, captureFrame]);

    useEffect(() => {
        const img = imgRef.current;
        if (!img) return;

        const handleError = () => {
            setError('摄像头连接失败');
            setReady(false);
        };
        const handleLoad = () => {
            setError(undefined);
            setReady(true);
        };

        img.addEventListener('error', handleError);
        img.addEventListener('load', handleLoad);

        // MJPEG 流首次加载时 img.complete 可能已经为 true
        if (img.complete && img.naturalWidth > 0) {
            setReady(true);
        }

        return () => {
            img.removeEventListener('error', handleError);
            img.removeEventListener('load', handleLoad);
        };
    }, []);

    // 定期检查图片是否已加载（MPEG 流不会触发 load 事件）
    useEffect(() => {
        const checkImage = () => {
            const img = imgRef.current;
            if (img && img.naturalWidth > 0 && !ready) {
                setReady(true);
            }
        };
        const interval = setInterval(checkImage, 100);
        return () => clearInterval(interval);
    }, [ready]);

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

            <div className="flex-1 min-h-0">
                <div className="h-full min-h-[220px] w-full overflow-hidden rounded-lg border border-slate-700 bg-black shadow-lg">
                    <img
                        ref={imgRef}
                        src={streamUrl}
                        alt="Camera stream"
                        crossOrigin="anonymous"
                        className="block h-full w-full object-contain"
                    />
                </div>
            </div>

            <canvas ref={canvasRef} className="hidden"/>

            <div className="text-xs text-slate-500 space-y-1">
                <div>{description}</div>
                {error && <div className="text-red-400">{error}</div>}
                {!isRecording && <div>Not recording</div>}
            </div>
        </div>
    );
});

MjpegStreamView.displayName = "MjpegStreamView";
