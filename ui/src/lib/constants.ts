export const getDatasetPath = (userId: string, datasetName?: string) =>
  datasetName ? `output/dataset/${userId}/${datasetName}` : `output/dataset/${userId}`;
export const getTrainPath = (userId: string, datasetName?: string) =>
  datasetName ? `output/train/${userId}/${datasetName}` : `output/train/${userId}`;
export const getModelPath = (userId: string, datasetName?: string) =>
  datasetName ? `output/train/${userId}/${datasetName}/model.pt` : `output/train/${userId}/model.pt`;

export const PATHS = {
  DATASET: 'output/dataset',
  TRAIN_DIR: 'output/train',
  MODEL: 'output/train/model.pt',
} as const

export const SIMULATION = {
  SEND_INTERVAL_MS: 50,
  DEFAULT_FPS: 30,
  MAX_FPS: 60,
} as const
