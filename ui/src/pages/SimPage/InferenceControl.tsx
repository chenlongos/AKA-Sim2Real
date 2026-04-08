interface InferenceControlProps {
    isModelLoaded: boolean;
    isInferring: boolean;
    autoInference: boolean;
    inferenceResult: string[];
    onLoadModel: () => void;
    onInference: () => void;
    onAutoInference: () => void;
}

export const InferenceControl = ({
    isModelLoaded,
    isInferring,
    autoInference,
    inferenceResult,
    onLoadModel,
    onInference,
    onAutoInference,
}: InferenceControlProps) => {
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
                <button
                    onClick={onLoadModel}
                    className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-500 hover:to-cyan-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-cyan-900/20"
                >
                    加载模型
                </button>
            ) : (
                <div className="space-y-2">
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
                </div>
            )}
        </div>
    );
};
