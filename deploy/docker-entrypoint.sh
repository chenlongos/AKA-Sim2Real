#!/bin/bash
# =============================================================================
# AKA-Sim2Real Docker Entrypoint
# 适配 CNB / 自托管双模式：
#   - CNB PaaS:   注入 $PORT，无 Redis → workers=1
#   - 自托管:     默认 80，有 Redis → workers=4
# =============================================================================
set -e

LISTEN_PORT="${PORT:-80}"
NGINX_CONF="/etc/nginx/conf.d/default.conf"
NGINX_TEMPLATE="${NGINX_CONF}.template"
SUPERVISOR_CONF="/etc/supervisor/supervisord.conf"

echo "=== AKA-Sim2Real Entrypoint ==="
echo "Listen port: ${LISTEN_PORT}"
echo "REDIS_URL:   ${REDIS_URL:-not set}"

# Redis 检测：无 Redis 时强制降为 1 worker（避免 Socket.IO 400）
if [ -z "${REDIS_URL}" ]; then
    if grep -q 'workers 4' "${SUPERVISOR_CONF}" 2>/dev/null; then
        sed -i 's/--workers 4/--workers 1/' "${SUPERVISOR_CONF}"
        echo "No Redis detected → workers: 1 (safe mode)"
    fi
else
    echo "Redis detected → workers: 4"
fi

# 从模板生成 nginx 配置
if [ -f "${NGINX_TEMPLATE}" ]; then
    export LISTEN_PORT
    envsubst "\${LISTEN_PORT}" < "${NGINX_TEMPLATE}" > "${NGINX_CONF}"
    echo "Nginx config generated (port: ${LISTEN_PORT})"
fi

mkdir -p /var/log/aka-sim

echo "Starting supervisord..."
exec supervisord -c "${SUPERVISOR_CONF}" -n
