# =============================================================================
# AKA-Sim2Real Production Docker Image
# Multi-stage: Frontend Build → Python Runtime + Nginx + Supervisor
# =============================================================================

# ---- Stage 1: Build Frontend ----
FROM node:22-alpine AS frontend-build

WORKDIR /opt/aka-sim/ui

# Install deps (cache layer)
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

# Build
COPY ui/ ./
RUN npm run build


# ---- Stage 2: Runtime ----
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /opt/aka-sim

# System dependencies: MuJoCo rendering libs + nginx + supervisor + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    patchelf \
    nginx \
    supervisor \
    gettext-base \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# CPU: 使用 PyTorch CPU wheelhouse（体积小，无 CUDA 依赖）
# GPU: docker build --build-arg CUDA=cu124 -t aka-sim2real:gpu .
# 国内加速: docker build --build-arg PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG PIP_MIRROR=""
ARG CUDA="cpu"
COPY backend/requirements.txt ./

# PyTorch 从官方 CDN 装（150MB，一次下载，Docker 层缓存后续秒过）
RUN pip install --no-cache-dir --default-timeout=300 \
        torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA}"

# 其余依赖（可走国内镜像加速）
RUN if [ -n "${PIP_MIRROR}" ]; then \
        pip install --no-cache-dir --default-timeout=300 -r requirements.txt -i "${PIP_MIRROR}"; \
    else \
        pip install --no-cache-dir --default-timeout=300 -r requirements.txt; \
    fi

# ---- Copy application ----
COPY backend/ ./backend/
COPY policies/ ./policies/

# Frontend dist from build stage
COPY --from=frontend-build /opt/aka-sim/ui/dist ./dist/

# ---- Nginx (template — port injected at runtime via envsubst) ----
COPY deploy/nginx.conf.template /etc/nginx/conf.d/default.conf.template
RUN rm -f /etc/nginx/sites-enabled/default

# ---- Supervisor ----
COPY deploy/supervisord.conf /etc/supervisor/supervisord.conf

# ---- Volumes ----
# output/ 结构: output/train/{user_id}/model.pt, output/dataset/{user_id}/
RUN mkdir -p /opt/aka-sim/output /var/log/aka-sim

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost/health || exit 1

# ---- Entrypoint ----
COPY deploy/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["docker-entrypoint.sh"]