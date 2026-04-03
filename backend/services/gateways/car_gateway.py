from __future__ import annotations

from typing import Any

import httpx


class CarGateway:
    """HTTP gateway for the physical car APIs."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def heartbeat(self, car_ip: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(f"http://{car_ip}/heartbeat")
            if res.status_code != 200:
                return {"ok": False, "status": res.status_code}
            return {"ok": True}

    async def control(self, car_ip: str, action: str, speed: int = 50) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(
                f"http://{car_ip}/api/control",
                params={"action": action, "speed": speed},
            )
            return {"ok": True, "status": res.status_code}

    async def motor_direct(self, car_ip: str, left: int, right: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(
                f"http://{car_ip}/api/motor_direct",
                params={"left": left, "right": right},
            )
            payload: Any
            try:
                payload = res.json()
            except ValueError:
                payload = {"raw": res.text}
            return {
                "ok": res.status_code == 200,
                "status": res.status_code,
                "left": left,
                "right": right,
                "payload": payload,
            }

    async def get_motor_status(self, car_ip: str, timestamp: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=max(self.timeout, 10.0)) as client:
            res = await client.get(
                f"http://{car_ip}/api/motor_status",
                params={"timestamp": timestamp},
            )
            if res.status_code != 200:
                return {"ok": False, "status": res.status_code}
            payload = res.json()
            if isinstance(payload, dict):
                payload.setdefault("ok", True)
            return payload

    def extract_timestamp(self, payload: dict[str, Any] | None) -> int | None:
        if not isinstance(payload, dict):
            return None

        candidates = ("timestamp", "timestamp_ms", "ts", "time", "time_ms")
        for key in candidates:
            value = payload.get(key)
            parsed = self._parse_timestamp(value)
            if parsed is not None:
                return parsed

        for nested_key in ("data", "motor_status", "status", "result"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                for key in candidates:
                    parsed = self._parse_timestamp(nested.get(key))
                    if parsed is not None:
                        return parsed

        return None

    def extract_wheel_velocity(self, payload: dict[str, Any] | None) -> tuple[float, float] | None:
        if not isinstance(payload, dict):
            return None

        candidates = (
            ("vel_left", "vel_right"),
            ("left", "right"),
            ("left_speed", "right_speed"),
            ("left_velocity", "right_velocity"),
            ("left_wheel", "right_wheel"),
        )
        for left_key, right_key in candidates:
            result = self._read_pair(payload, left_key, right_key)
            if result is not None:
                return result

        for nested_key in ("data", "motor_status", "status", "result"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                for left_key, right_key in candidates:
                    result = self._read_pair(nested, left_key, right_key)
                    if result is not None:
                        return result

        return None

    @staticmethod
    def _read_pair(source: dict[str, Any], left_key: str, right_key: str) -> tuple[float, float] | None:
        if left_key not in source or right_key not in source:
            return None
        try:
            return float(source[left_key]), float(source[right_key])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_timestamp(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None


car_gateway = CarGateway()
