import {forwardRef, useImperativeHandle, useRef, useEffect, useState} from "react";

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
    const streamUrl = `http://${carIP}/api/camera/stream`;

    useImperativeHandle(ref, () => ({
        getImageData: () => {
            const img = imgRef.current;
            const canvas = canvasRef.current;
            if (!img || !canvas || !img.complete || img.naturalWidth === 0) return undefined;

            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            if (!ctx) return undefined;

            ctx.drawImage(img, 0, 0);
            return canvas.toDataURL('image/jpeg', 0.8);
        }
    }), []);

    useEffect(() => {
        const img = imgRef.current;
        if (!img) return;

        const handleError = () => setError('摄像头连接失败');
        const handleLoad = () => setError(undefined);

        img.addEventListener('error', handleError);
        img.addEventListener('load', handleLoad);

        return () => {
            img.removeEventListener('error', handleError);
            img.removeEventListener('load', handleLoad);
        };
    }, []);

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
