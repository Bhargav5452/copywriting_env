# Copywriting Environment — Hugging Face Spaces Dockerfile
# Compatible with the OpenEnv base image pattern.
#
# Build locally:
#   docker build -f server/Dockerfile -t copywriting-env .
#
# Run locally:
#   docker run -p 8000:8000 copywriting-env
#
# Deploy to HF Spaces:
#   Push this Dockerfile (and full env folder) to a HF Space repository.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment source
COPY . /app/env

WORKDIR /app/env

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    fastapi>=0.115.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    requests>=2.31.0 \
    vaderSentiment>=3.3.2 \
    textstat>=0.7.3 \
    fastmcp>=3.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

ENV ENABLE_WEB_INTERFACE=true

# Start server
CMD ["sh", "-c", "cd /app/env && uvicorn src.envs.copywriting_env.server.app:app --host 0.0.0.0 --port 8000"]
