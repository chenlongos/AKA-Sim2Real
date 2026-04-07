import ky from 'ky';

const BASE_URL = '/api';

export const api = ky.create({
  prefixUrl: BASE_URL,
  timeout: 30000,
});

// ============ 数据采集 ============

export interface CollectImageRequest {
  image: string;
  actions: string[];
  timestamp?: number;
  state?: {
    vel_left: number;
    vel_right: number;
  };
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
}

export const loadTrainedModel = () =>
  api
    .post('act/load_trained')
    .json<LoadModelResponse>();
