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
    return (
        <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2 mt-4">
            <div className="font-semibold">推理控制</div>
            <div className="text-xs text-gray-600">
                模型状态: {isModelLoaded ? '已加载' : '未加载'}
            </div>
            {!isModelLoaded ? (
                <button
                    onClick={onLoadModel}
                    className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600"
                >
                    加载模型
                </button>
            ) : (
                <div className="flex gap-2">
                    <button
                        onClick={onInference}
                        disabled={isInferring || autoInference}
                        className={`px-3 py-1 rounded ${isInferring || autoInference ? 'bg-gray-300' : 'bg-green-500 text-black hover:bg-green-600'}`}
                    >
                        {isInferring ? '推理中...' : '单次推理'}
                    </button>
                    <button
                        onClick={onAutoInference}
                        disabled={isInferring}
                        className={`px-3 py-1 rounded ${autoInference ? 'bg-red-500 text-black' : 'bg-orange-500 text-black hover:bg-orange-600'}`}
                    >
                        {autoInference ? '停止自动' : '自动推理'}
                    </button>
                </div>
            )}
            {inferenceResult.length > 0 && (
                <div className="text-xs text-gray-600 mt-2">
                    推理动作: {inferenceResult.join(', ')}
                </div>
            )}
        </div>
    )
}
