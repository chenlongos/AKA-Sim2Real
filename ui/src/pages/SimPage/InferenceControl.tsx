import {useCallback, useState} from "react"

interface InferenceControlProps {
    isModelLoaded: boolean;
    selectedModel: string;
    isInferring: boolean;
    autoInference: boolean;
    inferenceResult: string[];
    models: string[];
    onLoadModel: (modelName: string) => void;
    onUnloadModel: () => void;
    onInference: () => void;
    onAutoInference: () => void;
    onRefreshModels: () => void;
}

export const InferenceControl = ({
    isModelLoaded,
    selectedModel,
    isInferring,
    autoInference,
    inferenceResult,
    models,
    onLoadModel,
    onUnloadModel,
    onInference,
    onAutoInference,
    onRefreshModels,
}: InferenceControlProps) => {
    const [isRefreshing, setIsRefreshing] = useState(false)

    const handleRefreshModels = useCallback(async () => {
        if (isRefreshing) return
        setIsRefreshing(true)
        try {
            await onRefreshModels()
        } finally {
            setTimeout(() => setIsRefreshing(false), 500)
        }
    }, [isRefreshing, onRefreshModels])

    const inferenceActionLabel = inferenceResult.length > 0
        ? inferenceResult.join(", ")
        : autoInference
            ? "自动推理中"
            : isInferring
                ? "单次推理中"
                : isModelLoaded
                    ? "暂无结果"
                    : "模型未加载"

    return (
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-3">
                <span className="w-2 h-2 bg-cyan-500 rounded-full"/>
                推理控制
                <div className="flex-1"/>
                <button
                    onClick={handleRefreshModels}
                    className={`text-slate-400 hover:text-slate-200 transition-colors ${isRefreshing ? 'animate-spin' : ''}`}
                    title="刷新模型列表"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
                        <path d="M21 3v5h-5"/>
                    </svg>
                </button>
            </h3>

            {/* 模型状态 */}
            <div className="flex items-center gap-2 mb-3">
                <span className="text-xs text-slate-400">模型状态</span>
                <div className="flex-1"/>
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                    isModelLoaded
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-700/50 text-slate-400'
                }`}>
                    {isModelLoaded ? '已加载' : '未加载'}
                </span>
            </div>

            <div className="flex items-center gap-2 mb-3">
                <span className="text-xs text-slate-400">推理动作</span>
                <div className="flex-1"/>
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                    autoInference
                        ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                        : isInferring
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                            : isModelLoaded
                                ? 'bg-slate-700/50 text-slate-300 border border-slate-600'
                                : 'bg-slate-700/50 text-slate-400'
                }`}>
                    {inferenceActionLabel}
                </span>
            </div>

            {!isModelLoaded ? (
                models.length > 0 ? (
                    <div className="space-y-2">
                        <select
                            className="w-full bg-slate-900/50 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-slate-500"
                            id="model-select"
                        >
                            {models.map((m) => (
                                <option key={m} value={m}>{m}</option>
                            ))}
                        </select>
                        <button
                            onClick={() => {
                                const select = document.getElementById("model-select") as HTMLSelectElement;
                                onLoadModel(select.value);
                            }}
                            className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-500 hover:to-cyan-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-cyan-900/20"
                        >
                            加载模型
                        </button>
                    </div>
                ) : (
                    <button
                        disabled
                        className="w-full py-2.5 bg-slate-700 text-slate-500 text-sm font-medium rounded-lg cursor-not-allowed"
                    >
                        暂无可用模型
                    </button>
                )
            ) : (
                <div className="space-y-2">
                    {/* 已加载模型显示 */}
                    <div className="flex items-center justify-between px-3 py-2 bg-slate-900/50 rounded border border-slate-700">
                        <span className="text-xs text-slate-400">当前模型</span>
                        <span className="text-sm text-emerald-400 font-mono">{selectedModel}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={onInference}
                            disabled={isInferring || autoInference}
                            className={`py-2 text-sm font-medium rounded-lg transition-all ${
                                isInferring || autoInference
                                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                            }`}
                        >
                            {isInferring ? '推理中...' : '单次推理'}
                        </button>
                        <button
                            onClick={onAutoInference}
                            disabled={isInferring}
                            className={`py-2 text-sm font-medium rounded-lg transition-all ${
                                autoInference
                                    ? 'bg-red-600 hover:bg-red-500 text-white'
                                    : isInferring
                                        ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                        : 'bg-orange-600 hover:bg-orange-500 text-white'
                            }`}
                        >
                            {autoInference ? '停止自动' : '自动推理'}
                        </button>
                    </div>
                    <button
                        onClick={onUnloadModel}
                        className="w-full py-2 text-xs text-slate-400 hover:text-slate-300 border border-slate-700 rounded-lg transition-all"
                    >
                        切换模型
                    </button>
                </div>
            )}
        </div>
    );
};
