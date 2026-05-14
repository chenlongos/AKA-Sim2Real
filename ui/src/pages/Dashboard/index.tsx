import {useEffect, useState} from "react";
import {api} from "../../api/api";
import {useSimCarStore} from "../../stores/simCarStore.ts";
import {showToast} from "../../lib/toast.ts";

interface DatasetItem {
  name: string;
  path: string;
  episode_count: number;
}

interface ContentItem {
  name: string;
  path: string;
  is_dir: boolean;
}

interface ModelItem {
  name: string;
  dataset: string;
  path: string;
}

const DashboardPage = () => {
  const userId = useSimCarStore((state) => state.userId)

  // Dataset state
  const [datasets, setDatasets] = useState<DatasetItem[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string>("")
  const [datasetContent, setDatasetContent] = useState<ContentItem[]>([])
  const [currentPath, setCurrentPath] = useState<string>("")
  const [breadcrumbs, setBreadcrumbs] = useState<{name: string; path: string}[]>([])

  // Model state
  const [models, setModels] = useState<ModelItem[]>([])
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set())

  // UI state
  const [activeTab, setActiveTab] = useState<"dataset" | "model">("dataset")
  const [loading, setLoading] = useState(true)

  // Load datasets
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

  // Load dataset content
  useEffect(() => {
    if (!selectedDataset) return
    const loadContent = async () => {
      try {
        const result = await api.get(
          `browser/dataset/${encodeURIComponent(selectedDataset)}/content?user_id=${encodeURIComponent(userId)}`
        ).json<{children: ContentItem[]; path: string}>()
        setDatasetContent(result.children || [])
        setCurrentPath(result.path)
        setBreadcrumbs([{name: selectedDataset, path: result.path}])
      } catch {
        showToast.error("加载内容失败")
      }
    }
    loadContent()
  }, [selectedDataset, userId])

  // Load models
  useEffect(() => {
    if (activeTab !== "model") return
    const loadModels = async () => {
      try {
        const result = await api.get(`browser/model?user_id=${encodeURIComponent(userId)}`).json<ModelItem[]>()
        setModels(result || [])
      } catch {
        showToast.error("加载模型失败")
      }
    }
    loadModels()
  }, [activeTab, userId])

  const navigateToPath = async (path: string) => {
    try {
      const result = await api.get(
        `browser/dataset/${encodeURIComponent(selectedDataset)}/content?user_id=${encodeURIComponent(userId)}&path=${encodeURIComponent(path)}`
      ).json<{children: ContentItem[]; path: string}>()
      setDatasetContent(result.children || [])
      setCurrentPath(result.path)
      // Update breadcrumbs
      const parts = path.split("/")
      const newBreadcrumbs = [{name: selectedDataset, path: selectedDataset}]
      let accum = selectedDataset
      for (let i = 1; i < parts.length; i++) {
        if (parts[i]) {
          accum += "/" + parts[i]
          newBreadcrumbs.push({name: parts[i], path: accum})
        }
      }
      setBreadcrumbs(newBreadcrumbs)
    } catch {
      showToast.error("加载失败")
    }
  }

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

  const deleteSelectedModels = async () => {
    if (selectedModels.size === 0) return
    if (!confirm(`确定删除 ${selectedModels.size} 个模型？`)) return

    try {
      for (const path of selectedModels) {
        await api.delete(`browser/model?user_id=${encodeURIComponent(userId)}&model_path=${encodeURIComponent(path)}`)
      }
      showToast.success(`已删除 ${selectedModels.size} 个模型`)
      setSelectedModels(new Set())
      // Refresh models
      const result = await api.get(`browser/model?user_id=${encodeURIComponent(userId)}`).json<ModelItem[]>()
      setModels(result || [])
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

        {/* Tab Switcher */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab("dataset")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "dataset"
                ? "bg-indigo-600 text-white"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            数据集 (Dataset)
          </button>
          <button
            onClick={() => setActiveTab("model")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "model"
                ? "bg-indigo-600 text-white"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            模型 (Model)
          </button>
        </div>

        {activeTab === "dataset" ? (
          <div className="space-y-4">
            {/* Dataset Selector */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="block text-xs text-slate-400">选择数据集</label>
                {selectedDataset && (
                  <button
                    onClick={async () => {
                      if (!confirm(`确定删除数据集 "${selectedDataset}"？此操作不可恢复！`)) return
                      try {
                        await api.delete(`browser/dataset?user_id=${encodeURIComponent(userId)}&dataset_name=${encodeURIComponent(selectedDataset)}`)
                        showToast.success("数据集已删除")
                        setSelectedDataset("")
                        setDatasetContent([])
                        setDatasets(datasets.filter(d => d.name !== selectedDataset))
                      } catch {
                        showToast.error("删除失败")
                      }
                    }}
                    className="px-3 py-1 bg-red-600 hover:bg-red-500 text-white text-xs rounded-lg"
                  >
                    删除当前数据集
                  </button>
                )}
              </div>
              <select
                value={selectedDataset}
                onChange={(e) => {
                  setSelectedDataset(e.target.value)
                  setDatasetContent([])
                }}
                className="w-full bg-slate-900/50 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-slate-500"
              >
                <option value="">-- 选择数据集 --</option>
                {datasets.map((ds) => (
                  <option key={ds.path} value={ds.name}>
                    {ds.name} ({ds.episode_count} episodes)
                  </option>
                ))}
              </select>
            </div>

            {/* Dataset Content */}
            {selectedDataset && (
              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                    </svg>
                    <h2 className="text-sm font-semibold text-slate-200">{selectedDataset}</h2>
                    <span className="text-xs text-slate-500 font-mono">{userId}</span>
                  </div>
                  {currentPath && currentPath !== selectedDataset && (
                    <button
                      onClick={() => {
                        const parentPath = currentPath.split("/").slice(0, -1).join("/")
                        navigateToPath(parentPath || selectedDataset)
                      }}
                      className="text-xs text-slate-400 hover:text-slate-200"
                    >
                      返回上级
                    </button>
                  )}
                </div>

                {/* Breadcrumbs */}
                {breadcrumbs.length > 1 && (
                  <div className="flex items-center gap-1 mb-3 text-xs text-slate-500">
                    {breadcrumbs.map((bc, i) => (
                      <span key={bc.path} className="flex items-center gap-1">
                        {i > 0 && <span>/</span>}
                        <button
                          onClick={() => navigateToPath(bc.path)}
                          className={`hover:text-slate-300 ${i === breadcrumbs.length - 1 ? "text-slate-300" : ""}`}
                        >
                          {bc.name}
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                {datasetContent.length > 0 ? (
                  <div className="space-y-1">
                    {datasetContent.map((item) => (
                      <div
                        key={item.path}
                        onClick={() => {
                          if (item.is_dir) {
                            navigateToPath(item.path)
                          }
                        }}
                        className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                          item.is_dir
                            ? "hover:bg-slate-700/50 text-slate-200"
                            : "text-slate-500"
                        }`}
                      >
                        {item.is_dir ? (
                          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                          </svg>
                        ) : (
                          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                            <polyline points="13,2 13,9 20,9"/>
                          </svg>
                        )}
                        <span className="text-sm">{item.name}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-slate-500 text-sm">暂无内容</p>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-orange-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                </svg>
                <h2 className="text-sm font-semibold text-slate-200">模型列表</h2>
                <span className="text-xs text-slate-500 ml-2">{models.length} 个模型</span>
              </div>
              {selectedModels.size > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">{selectedModels.size} 已选</span>
                  <button
                    onClick={deleteSelectedModels}
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

            {loading ? (
              <div className="text-center py-8 text-slate-400">加载中...</div>
            ) : models.length > 0 ? (
              <div className="space-y-1">
                {models.map((m) => (
                  <div
                    key={m.path}
                    onClick={() => toggleModelSelect(m.path)}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                      selectedModels.has(m.path)
                        ? "bg-red-900/20 border border-red-700/50"
                        : "hover:bg-slate-700/50 border border-transparent"
                    }`}
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
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-orange-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                    </svg>
                    <div className="flex-1">
                      <div className="text-sm text-slate-200">{m.name}</div>
                      <div className="text-xs text-slate-500">{m.dataset}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-sm text-center py-8">暂无模型</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default DashboardPage