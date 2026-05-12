export interface CarHeartbeatResponse {
  ok?: boolean;
  message?: string;
  status?: string;
}

type CarApiSuccessPayload = {
  ok?: boolean;
  status?: string;
};

const fetchCarApi = async <T>(
  carIP: string,
  path: string,
  searchParams: Record<string, string | number>,
  options?: { signal?: AbortSignal },
) => {
  const url = new URL(`http://${carIP}/api/${path}`);
  for (const [key, value] of Object.entries(searchParams)) {
    url.searchParams.set(key, String(value));
  }
  const response = await fetch(url.toString(), { method: 'GET', signal: options?.signal });
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
};

export const carHeartbeat = (carIP: string) =>
  fetchCarApi<CarHeartbeatResponse>(carIP, 'system/heartbeat', {});

export interface CameraAllStatusResponse {
  timestamp?: string;
  left_speed?: number;
  right_speed?: number;
  left_target?: number;
  right_target?: number;
  timestamp_ms?: number;
  image?: string;
  image_format?: string;
  error?: string;
  detail?: string;
  message?: string;
}

export const cameraAllStatus = (carIP: string, timestamp?: number) => {
  const params: Record<string, string | number> = {};
  if (timestamp !== undefined) {
    params.timestamp = timestamp;
  }
  return fetchCarApi<CameraAllStatusResponse>(carIP, 'camera/all_status', params);
};

export interface CarStatusResponse {
  left_speed?: number;
  right_speed?: number;
  left_target?: number;
  right_target?: number;
  error?: string;
  detail?: string;
  message?: string;
}

export const carStatus = (carIP: string) => {
  return fetchCarApi<CarStatusResponse>(carIP, 'camera/speed', {});
};

const MOTOR_ACTION_EPSILON = 1e-3;

export const getActionsFromMotorStatus = (
  motorStatus: Pick<CameraAllStatusResponse, 'left_speed' | 'right_speed'>,
) => {
  const left = motorStatus.left_speed;
  const right = motorStatus.right_speed;

  if (typeof left !== 'number' || typeof right !== 'number') {
    return [] as string[];
  }

  if (Math.abs(left) < MOTOR_ACTION_EPSILON && Math.abs(right) < MOTOR_ACTION_EPSILON) {
    return [] as string[];
  }

  if (left > MOTOR_ACTION_EPSILON && right > MOTOR_ACTION_EPSILON) {
    return ['forward'];
  }

  if (left < -MOTOR_ACTION_EPSILON && right < -MOTOR_ACTION_EPSILON) {
    return ['backward'];
  }

  if (left < -MOTOR_ACTION_EPSILON && right > MOTOR_ACTION_EPSILON) {
    return ['left'];
  }

  if (left > MOTOR_ACTION_EPSILON && right < -MOTOR_ACTION_EPSILON) {
    return ['right'];
  }

  return [];
};

export interface MotorDirectResponse {
  ok?: boolean;
  status?: string;
  left?: number;
  right?: number;
  duration?: number;
  error?: string;
  detail?: string;
  message?: string;
}

export const motorDirect = (
  carIP: string,
  left: number,
  right: number,
  duration: number = 0,
  options?: { signal?: AbortSignal },
) => fetchCarApi<MotorDirectResponse>(carIP, 'motor/direct', { left, right, duration }, options);

export interface CarControlResponse {
  ok?: boolean;
  status?: string;
  error?: string;
  detail?: string;
  message?: string;
}

export const carControl = (carIP: string, action: string, speed: number = 50) =>
  fetchCarApi<CarControlResponse>(carIP, 'control', { action, speed });

export const isCarApiSuccess = (payload?: CarApiSuccessPayload | null) =>
  payload?.ok === true || payload?.status === 'ok' || payload?.status === 'success';
