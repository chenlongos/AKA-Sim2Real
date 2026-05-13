import {useEffect, useState} from "react";
import {api} from "../../api/api";
import {useSimCarStore} from "../../stores/simCarStore.ts";
import {showToast} from "../../lib/toast.ts";

interface DatasetItem {
  name: string;
  path: string;
  episode_count: number;
}

interface ModelItem {
  name: string;
  path: string;
}

const DashboardPage = () => {
  const userId = useSimCarStore((state) => state.userId)
  const [datasets, setDatasets] = useState<DatasetItem[]>([])
  const [selectedDataset, setSelectedDataset] = useState<DatasetItem | null>(null)
  const [models, setModels] = useState<ModelItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set())

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const result = await api.get(`browser/dataset?user_id=${encodeURIComponent(userId)}`).json<DatasetItem[]>()
        setDatasets(result || [])
      } catch {
        showToast.error("加载数据失败")
      } finally {
        setLoading(false)
      }
    }
    loadDatasets()
  }, [userId])

  useEffect(() => {
    if (!selectedDataset) return
    const loadModels = async () => {
      try {
        const result = await api.get(
          `browser/dataset/${encodeURIComponent(selectedDataset.name)}/model?user_id=${encodeURIComponent(userId)}`
        ).json<ModelItem[]>()
        setModels(result || [])
      } catch {
        showToast.error("加载模型失败")
      }
    }
    loadModels()
  }, [selectedDataset, userId])

  const toggleModelSelect = (path: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  const deleteSelected = async () => {
    if (selectedModels.size === 0) return
    if (!confirm(`确定删除 ${selectedModels.size} 个模型？`)) return

    try {
      for (const path of selectedModels) {
        await api.delete(`browser/model?user_id=${encodeURIComponent(userId)}&model_path=${encodeURIComponent(path)}`)
      }
      showToast.success(`已删除 ${selectedModels.size} 个模型`)
      setSelectedModels(new Set())
      // Refresh models
      if (selectedDataset) {
        const result = await api.get(
          `browser/dataset/${encodeURIComponent(selectedDataset.name)}/model?user_id=${encodeURIComponent(userId)}`
        ).json<ModelItem[]>()
        setModels(result || [])
      }
    } catch {
      showToast.error("删除失败")
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg">
            <span className="text-white font-bold text-sm">DATA</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">数据管理中心</h1>
            <p className="text-xs text-slate-500">用户 {userId}</p>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-slate-400">加载中...</div>
          </div>
        ) : selectedDataset ? (
          <div>
            <button
              onClick={() => {
                setSelectedDataset(null)
                setModels([])
                setSelectedModels(new Set())
              }}
              className="flex items-center gap-1 text-slate-400 hover:text-slate-200 mb-4 text-sm"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 12H5M12 19l-7-7 7-7"/>
              </svg>
              返回数据集列表
            </button>

            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                  </svg>
                  <h2 className="text-lg font-semibold text-slate-200">{selectedDataset.name}</h2>
                  <span className="text-xs text-slate-500 ml-2">{models.length} 个模型</span>
                </div>
                {selectedModels.size > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">{selectedModels.size} 已选</span>
                    <button
                      onClick={deleteSelected}
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs rounded-lg"
                    >
                      删除选中
                    </button>
                    <button
                      onClick={() => setSelectedModels(new Set())}
                      className="px-3 py-1.5 text-slate-400 hover:text-slate-200 text-xs"
                    >
                      取消
                    </button>
                  </div>
                )}
              </div>

              {models.length > 0 ? (
                <div className="space-y-2">
                  {models.map((m) => (
                    <div
                      key={m.path}
                      className={`flex items-center gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedModels.has(m.path)
                          ? "bg-red-900/20 border-red-700/50"
                          : "bg-slate-900/50 border-slate-700 hover:border-slate-600"
                      }`}
                      onClick={() => toggleModelSelect(m.path)}
                    >
                      <div className={`w-5 h-5 rounded border flex items-center justify-center ${
                        selectedModels.has(m.path) ? "bg-red-600 border-red-600" : "border-slate-600"
                      }`}>
                        {selectedModels.has(m.path) && (
                          <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                            <path d="M20 6L9 17l-5-5"/>
                          </svg>
                        )}
                      </div>
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                      </svg>
                      <div>
                        <div className="text-sm text-slate-200">{m.name}</div>
                        <div className="text-xs text-slate-500 font-mono">{m.path}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-sm">暂无模型</p>
              )}
            </div>
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-20">
            <p className="text-slate-400">暂无数据</p>
          </div>
        ) : (
          <div className="space-y-3">
            {datasets.map((ds) => (
              <div
                key={ds.path}
                onClick={() => setSelectedDataset(ds)}
                className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 cursor-pointer hover:border-slate-600 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                  </svg>
                  <div className="flex-1">
                    <h2 className="text-sm font-semibold text-slate-200">{ds.name}</h2>
                    <p className="text-xs text-slate-500">{ds.episode_count} episodes</p>
                  </div>
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 18l6-6-6-6"/>
                  </svg>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default DashboardPage