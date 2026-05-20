/**
 * 教学平台事件上报服务
 *
 * 通过 postMessage 向 iframe 父页面（教学平台）上报学员操作行为、
 * 数据采集过程和训练结果。不在 iframe 中时自动静默，不影响独立使用。
 */

export type LessonEventType =
  // 操作事件
  | "action.enter"
  | "action.leave"
  | "action.start_collection"
  | "action.end_collection"
  | "action.start_training"
  | "action.stop_training"
  | "action.load_model"
  | "action.switch_episode"
  | "action.reset_scene"
  | "action.connect_car"
  // 采集详情（通过 Socket.IO 事件自动触发）
  | "collection.episode_started"
  | "collection.episode_ended"
  | "collection.episode_finalized"
  // 训练详情（通过 training_progress 事件自动触发）
  | "training.epoch_progress"
  | "training.completed"
  | "training.stopped";

interface LessonEvent {
  namespace: "aka-sim-lesson";
  type: LessonEventType;
  timestamp: number;
  page: "sim" | "real";
  data?: Record<string, unknown>;
}

let currentPage: "sim" | "real" | null = null;
let isInIframe = false;

export function init(page: "sim" | "real"): void {
  currentPage = page;
  isInIframe = window.self !== window.top;
}

export function send(type: LessonEventType, data?: Record<string, unknown>): void {
  if (!currentPage) return;

  const event: LessonEvent = {
    namespace: "aka-sim-lesson",
    type,
    timestamp: Date.now(),
    page: currentPage,
    data,
  };

  console.log(`[ReportService] ${type}`, data ?? {});

  if (!isInIframe) return;

  window.parent.postMessage(event, "*");
}
