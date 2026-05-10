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

export interface ModelsResponse {
  models?: string[];
}

export const listModels = (userId: string, datasetName: string) =>
  api
    .get(`dataset/models?user_id=${encodeURIComponent(userId)}&dataset_name=${encodeURIComponent(datasetName)}`)
    .json<ModelsResponse>();

// ============ 数据采集 ============

export interface CollectImageRequest {
  image: string;
  user_id?: string;
  dataset_name?: string;
  timestamp?: number;
  state?: {
    vel_left: number;
    vel_right: number;
  };
  action?: [number, number];
}

export interface CollectImageResponse {
  success?: boolean;
  count?: number;
  error?: string;
  detail?: string;
  message?: string;
}

export const collectImage = (data: CollectImageRequest) =>
  api
    .post('dataset/collect', { json: data })
    .json<CollectImageResponse>();

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

export const startTraining = (params: TrainingParams) =>
  api
    .post('train', { json: params })
    .json<TrainingResponse>();

export interface StopTrainingResponse {
  success?: boolean;
  message?: string;
}

export const stopTraining = () =>
  api
    .post('train/stop')
    .json<StopTrainingResponse>();

// ============ 推理相关 ============

export interface LoadModelResponse {
  success: boolean;
  message?: string;
  detail?: string;
  stats?: {
    state_min: number[];
    state_max: number[];
  };
}

export const loadTrainedModel = (dataDir?: string, modelPath?: string) => {
  let url = 'act/load_trained';
  const params: string[] = [];
  if (dataDir) params.push(`data_dir=${encodeURIComponent(dataDir)}`);
  if (modelPath) params.push(`model_path=${encodeURIComponent(modelPath)}`);
  if (params.length > 0) url += '?' + params.join('&');
  return api.post(url).json<LoadModelResponse>();
};
