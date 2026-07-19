import ky from 'ky';

const BASE_URL = '/api';

export const api = ky.create({
  prefixUrl: BASE_URL,
  timeout: 30000,
});

// ============ 数据集相关 ============

export interface DatasetDirsResponse {
  datasets?: string[];
}

export const listDatasetDirs = (userId: string) =>
  api
    .get(`dataset/dirs?user_id=${encodeURIComponent(userId)}`)
    .json<DatasetDirsResponse>();

export interface DatasetInfoResponse {
  dataset_name: string;
  total_frames: number;
  total_episodes: number;
  exists: boolean;
}

export const getDatasetInfo = (userId: string, datasetName: string) =>
  api
    .get(`dataset/info?user_id=${encodeURIComponent(userId)}&dataset_name=${encodeURIComponent(datasetName)}`)
    .json<DatasetInfoResponse>();

export interface ModelsResponse {
  models?: string[];
}

export const listModels = (userId: string, datasetName: string) =>
  api
    .get(`dataset/models?user_id=${encodeURIComponent(userId)}&dataset_name=${encodeURIComponent(datasetName)}`)
    .json<ModelsResponse>();

// ============ 训练相关 ============

export interface TrainingParams {
  data_dir?: string;
  output_dir?: string;
  epochs?: number;
  batch_size?: number;
  lr?: number;
  episode_ids?: number[];
  resume_from?: string;
}

export interface TrainingResponse {
  success: boolean;
  message?: string;
}

export const startTraining = (userId: string, params: TrainingParams) =>
  api
    .post(`train?user_id=${encodeURIComponent(userId)}`, { json: params })
    .json<TrainingResponse>();

export interface TrainingStatus {
  is_running: boolean;
  epoch: number;
  total_epochs: number;
  loss: number;
  progress: number;
  error?: string | null;
  message?: string;
}

export const getTrainingStatus = (userId: string) =>
  api
    .get(`train/status?user_id=${encodeURIComponent(userId)}`)
    .json<TrainingStatus>();

export interface StopTrainingResponse {
  success?: boolean;
  message?: string;
}

export const stopTraining = (userId: string) =>
  api
    .post(`train/stop?user_id=${encodeURIComponent(userId)}`)
    .json<StopTrainingResponse>();

// ============ 推理相关 ============

export interface LoadModelResponse {
  success: boolean;
  message?: string;
  detail?: string;
}

export const loadTrainedModel = (userId: string, dataDir?: string, modelPath?: string) => {
  let url = `act/load_trained?user_id=${encodeURIComponent(userId)}`;
  const params: string[] = [];
  if (dataDir) params.push(`data_dir=${encodeURIComponent(dataDir)}`);
  if (modelPath) params.push(`model_path=${encodeURIComponent(modelPath)}`);
  if (params.length > 0) url += '&' + params.join('&');
  return api.post(url).json<LoadModelResponse>();
};
