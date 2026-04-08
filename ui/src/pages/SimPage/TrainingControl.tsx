interface TrainingControlProps {
    collectedCount: number;
    isTraining: boolean;
    trainingProgress: { epoch: number; total_epochs: number; loss: number; progress: number };
    trainingEpochs: number;
    collectionFps: number;
    resumeTraining: boolean;
    episodeCounts: Record<number, number>;
    currentEpisode: number;
    isRecording: boolean;
    onStartTraining: () => void;
    onStopTraining: () => void;
    onSetTrainingEpochs: (epochs: number) => void;
    onSetCollectionFps: (fps: number) => void;
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
    collectionFps,
    resumeTraining,
    episodeCounts,
    currentEpisode,
    isRecording,
    onStartTraining,
    onStopTraining,
    onSetTrainingEpochs,
    onSetCollectionFps,
    onSetResumeTraining,
    onSetEpisode,
    onEndEpisode,
    onStartEpisode,
    onResetCar,
}: TrainingControlProps) => {
    const collectionFpsPresets = [10, 20, 30]

    return (
        <div className="flex flex-col gap-3">
            {/* 数据采集模块 */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                        数据采集
                    </h3>
                    <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium ${isRecording
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : 'bg-slate-700/50 text-slate-400'
                        }`}>
                        {isRecording && <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />}
                        {isRecording ? '收集中' : '未录制'}
                    </div>
                </div>

                {/* 帧数显示 */}
                {collectedCount > 0 && (
                    <div className="mb-2 bg-slate-900/50 rounded px-3 py-0">
                        <div className="text-[11px] leading-4 text-slate-400">已采集</div>
                        <div className="text-base leading-5 font-mono font-bold text-emerald-400">
                            {collectedCount} <span className="text-xs text-slate-500">帧</span>
                        </div>
                    </div>
                )}

                <div className="mb-2">
                    <div className="flex items-center gap-2 mb-1.5">
                        <label className="text-xs text-slate-400">采样频率</label>
                        <div className="flex-1" />
                        <span className="text-xs font-mono text-slate-300">{collectionFps} FPS</span>
                    </div>
                    <div className="flex gap-2">
                        {collectionFpsPresets.map((fps) => (
                            <button
                                key={fps}
                                type="button"
                                onClick={() => onSetCollectionFps(fps)}
                                className={`flex-1 px-2 py-1 rounded text-xs font-mono border transition-all ${
                                    collectionFps === fps
                                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                        : "bg-slate-900/50 text-slate-300 border-slate-700 hover:bg-slate-800"
                                }`}
                            >
                                {fps} FPS
                            </button>
                        ))}
                    </div>
                </div>

                {/* 采集控制按钮 */}
                <div className="space-y-2">
                    {isRecording ? (
                        <button
                            onClick={onEndEpisode}
                            className="w-full py-2.5 bg-linear-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-red-900/20"
                        >
                            结束采集 (第{currentEpisode}轮)
                        </button>
                    ) : (
                        <>
                            <button
                                onClick={onStartEpisode}
                                className="w-full py-2.5 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-emerald-900/20"
                            >
                                开始采集 (第{currentEpisode}轮)
                            </button>
                            <button
                                onClick={onResetCar}
                                className="w-full py-2.5 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-emerald-900/20"
                            >
                                复位场景
                            </button>
                        </>
                    )}
                </div>

                {/* 轮次控制 */}
                <div className="mt-2 pt-2 border-t border-slate-700/50">
                    <div className="flex items-center gap-2 mb-1.5">
                        <label className="text-xs text-slate-400">采集轮次</label>
                        <div className="flex-1" />
                        <span className="text-xs font-mono text-slate-300">
                            Ep. {currentEpisode}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <input
                            type="number"
                            value={currentEpisode}
                            onChange={(e) => onSetEpisode(parseInt(e.target.value) || 1)}
                            min={1}
                            disabled={isRecording}
                            className="flex-1 bg-slate-900/50 border border-slate-700 rounded px-1 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-slate-500 disabled:opacity-50"
                        />
                        {!isRecording && currentEpisode > 1 && (
                            <button
                                onClick={() => onSetEpisode(currentEpisode - 1)}
                                className="px-3 py-2 cursor-pointer  bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded-lg transition-all"
                            >
                                ← 回退
                            </button>
                        )}
                    </div>
                </div>

                {/* 各轮次数据量 */}
                {Object.keys(episodeCounts).length > 0 && (
                    <div className="mt-2 pt-2 border-t border-slate-700/50">
                        <div className="text-xs text-emerald-400 mb-1">√ 数据集已上传</div>
                        <div className="bg-slate-900/50 rounded-lg p-2 max-h-20 overflow-y-auto space-y-1">
                            {Object.entries(episodeCounts).map(([ep, count]) => (
                                <div key={ep} className="flex items-center justify-between text-xs">
                                    <span className="text-slate-400">轮次 {ep}</span>
                                    <span className="font-mono text-emerald-400">{count} 样本</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* 训练控制模块 */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-3">
                    <span className="w-2 h-2 bg-violet-500 rounded-full" />
                    模型训练
                </h3>

                {/* 训练参数 */}
                <div className="space-y-2.5 mb-3">
                    <div className="flex items-center gap-2">
                        <label className="text-xs text-slate-400 w-20">训练轮次</label>
                        <input
                            type="number"
                            value={trainingEpochs}
                            onChange={(e) => onSetTrainingEpochs(parseInt(e.target.value) || 1)}
                            min={1}
                            max={1000}
                            disabled={isTraining}
                            className="flex-1 bg-slate-900/50 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-slate-500 disabled:opacity-50"
                        />
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={resumeTraining}
                            onChange={(e) => onSetResumeTraining(e.target.checked)}
                            disabled={isTraining}
                            className="w-4 h-4 rounded border-slate-600 bg-slate-900 text-violet-500 focus:ring-violet-500/20"
                        />
                        <span className="text-xs text-slate-400">从已有模型继续</span>
                    </label>
                </div>

                {/* 训练按钮 */}
                {!isTraining ? (
                    <button
                        onClick={onStartTraining}
                        disabled={Object.keys(episodeCounts).length === 0 && collectedCount === 0}
                        className={`w-full py-2.5 text-sm font-medium rounded-lg transition-all ${Object.keys(episodeCounts).length > 0 || collectedCount > 0
                                ? 'bg-linear-to-r from-violet-600 to-violet-700 hover:from-violet-500 hover:to-violet-600 text-white shadow-lg shadow-violet-900/20'
                                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                            }`}
                    >
                        开始训练
                    </button>
                ) : (
                    <>
                        <button
                            onClick={onStopTraining}
                            className="w-full py-2.5 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-red-900/20"
                        >
                            停止训练
                        </button>
                        {/* 训练进度 */}
                        <div className="mt-3 pt-3 border-t border-slate-700/50">
                            <div className="flex items-center justify-between text-xs mb-1.5">
                                <span className="text-slate-400">
                                    Epoch {trainingProgress.epoch}/{trainingProgress.total_epochs}
                                </span>
                                <span className="font-mono text-violet-400">
                                    {Math.round(trainingProgress.progress * 100)}%
                                </span>
                            </div>
                            <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-violet-500 to-violet-400 rounded-full transition-all duration-300"
                                    style={{ width: `${trainingProgress.progress * 100}%` }}
                                />
                            </div>
                            <div className="mt-1.5 text-xs font-mono text-slate-400">
                                Loss: <span className="text-slate-200">{trainingProgress.loss.toFixed(6)}</span>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};
