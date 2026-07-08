FROM python:3.10-slim AS env-builder

# Install system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files (including .git for faster HF Spaces dev)
# CACHE BUST: 2026-02-10 - Added setuptools to fix pkg_resources import
COPY pyproject.toml README.md ./
COPY .git/ ./.git/
COPY src/ ./src/
COPY data/*.csv data/metadata/ ./data/

# Create venv and install all dependencies (all deps are public HTTPS)
# IMPORTANT: setuptools must be installed for pkg_resources (required by pytorch_lightning)
RUN uv venv && \
    echo "Installing dependencies (including setuptools for pkg_resources)..." && \
    uv pip install -e .

# Create non-root user for runtime
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

ENV PATH="/app/.venv/bin:$PATH"
# Set OMP_NUM_THREADS to a safe default to avoid libgomp errors
ENV OMP_NUM_THREADS=4
# Prevent PyTorch from calling getpass.getuser() which fails with arbitrary UIDs
# (common in OpenShift/Kubernetes environments)
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache

EXPOSE 7860

# Use mosaic CLI command which runs gradio_app:main (skips runtime pip install)
CMD ["mosaic"]
