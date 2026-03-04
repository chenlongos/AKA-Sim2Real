interface TrainingControlProps {
    collectedCount: number;
    isTraining: boolean;
    trainingProgress: { epoch: number; total_epochs: number; loss: number; progress: number };
    trainingEpochs: number;
    resumeTraining: boolean;
    episodeCounts: Record<number, number>;
    currentEpisode: number;
    isRecording: boolean;
    onStartTraining: () => void;
    onStopTraining: () => void;
    onSetTrainingEpochs: (epochs: number) => void;
    onSetResumeTraining: (resume: boolean) => void;
    onSetEpisode: (episode: number) => void;
    onEndEpisode: () => void;
    onStartEpisode: () => void;
    onResetCar: () => void;
}

export const TrainingControl = ({
    collectedCount,
    isTraining,
    trainingProgress,
    trainingEpochs,
    resumeTraining,
    episodeCounts,
    currentEpisode,
    isRecording,
    onStartTraining,
    onStopTraining,
    onSetTrainingEpochs,
    onSetResumeTraining,
    onSetEpisode,
    onEndEpisode,
    onStartEpisode,
    onResetCar,
}: TrainingControlProps) => {
    return (
        <>
            <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2">
                <div className="font-semibold">训练控制</div>

                {/* 数据收集流程 */}
                <div className="border border-gray-300 rounded p-2">
                    <div className="text-xs font-semibold mb-2">数据采集</div>

                    {/* 录制状态 */}
                    <div className="flex items-center gap-2 mb-2">
                        <span
                            className={`text-xs px-2 py-0.5 rounded ${isRecording ? 'bg-red-500 text-black' : 'bg-gray-300'}`}>
                            {isRecording ? '收集中' : '未录制'}
                        </span>
                        {collectedCount > 0 && (
                            <span className="text-xs text-gray-600">{collectedCount} 帧</span>
                        )}
                    </div>

                    {/* 操作按钮 */}
                    <div className="flex flex-col gap-2">
                        {isRecording ? (
                            <button
                                onClick={onEndEpisode}
                                className="px-2 py-1.5 text-xs bg-red-500 text-black rounded hover:bg-red-600 font-medium"
                            >
                                结束采集 (第{currentEpisode}轮)
                            </button>
                        ) : (
                            <>
                                <button
                                    onClick={onStartEpisode}
                                    className="px-2 py-1.5 text-xs bg-green-500 text-black rounded hover:bg-green-600 font-medium"
                                >
                                    开始采集 (第{currentEpisode}轮)
                                </button>
                                <button
                                    onClick={onResetCar}
                                    className="px-2 py-1.5 text-xs bg-yellow-500 text-black rounded hover:bg-yellow-600 font-medium"
                                >
                                    复位场景
                                </button>
                            </>
                        )}
                    </div>

                    {!isRecording && currentEpisode > 1 && (
                        <div className="mt-2 pt-2 border-t border-gray-200">
                            <button
                                onClick={() => onSetEpisode(currentEpisode - 1)}
                                className="px-2 py-1 text-xs bg-gray-400 text-black rounded hover:bg-gray-500"
                            >
                                回退到第 {currentEpisode - 1} 轮
                            </button>
                        </div>
                    )}
                </div>

                {!isRecording && (
                    <div className="text-xs text-gray-600">
                        当前轮次: {currentEpisode}
                    </div>
                )}

                <div className="text-xs text-gray-500 mb-1">
                    可修改轮次重新采集
                </div>
                <div className="flex items-center gap-2">
                    <label className="text-xs text-gray-600">采集轮次:</label>
                    <input
                        type="number"
                        value={currentEpisode}
                        onChange={(e) => onSetEpisode(parseInt(e.target.value) || 1)}
                        min={1}
                        disabled={isRecording}
                        className="w-16 px-2 py-1 text-sm border rounded"
                    />
                </div>

                {Object.keys(episodeCounts).length > 0 && (
                    <div className="text-xs text-gray-600 max-h-24 overflow-y-auto border rounded p-1">
                        {Object.entries(episodeCounts).map(([ep, count]) => (
                            <div key={ep} className="flex justify-between">
                                <span>轮次 {ep}:</span>
                                <span>{count} 样本</span>
                            </div>
                        ))}
                    </div>
                )}

                <div className="flex items-center gap-2">
                    <label className="text-xs text-gray-600">训练轮次:</label>
                    <input
                        type="number"
                        value={trainingEpochs}
                        onChange={(e) => onSetTrainingEpochs(parseInt(e.target.value) || 1)}
                        min={1}
                        max={1000}
                        disabled={isTraining}
                        className="w-20 px-2 py-1 text-sm border rounded"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="resumeTraining"
                        checked={resumeTraining}
                        onChange={(e) => onSetResumeTraining(e.target.checked)}
                        disabled={isTraining}
                        className="w-4 h-4"
                    />
                    <label htmlFor="resumeTraining" className="text-xs text-gray-600">
                        从已有模型继续训练
                    </label>
                </div>
                {!isTraining ? (
                    <button
                        onClick={onStartTraining}
                        disabled={collectedCount === 0}
                        className={`px-3 py-1 rounded ${collectedCount > 0 ? 'bg-purple-500 text-black hover:bg-purple-600' : 'bg-gray-300 text-gray-500'}`}
                    >
                        开始训练
                    </button>
                ) : (
                    <button
                        onClick={onStopTraining}
                        className="px-3 py-1 bg-red-500 text-black rounded hover:bg-red-600"
                    >
                        停止训练
                    </button>
                )}
                {isTraining && (
                    <div className="mt-2">
                        <div className="text-xs text-gray-600">
                            Epoch: {trainingProgress.epoch}/{trainingProgress.total_epochs}
                        </div>
                        <div className="text-xs text-gray-600">
                            Loss: {trainingProgress.loss.toFixed(6)}
                        </div>
                        <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                            <div
                                className="bg-purple-500 h-2 rounded-full transition-all"
                                style={{width: `${trainingProgress.progress * 100}%`}}
                            />
                        </div>
                        <div className="text-xs text-gray-600 text-center mt-1">
                            {Math.round(trainingProgress.progress * 100)}%
                        </div>
                    </div>
                )}
            </div>
        </>
    )
}
