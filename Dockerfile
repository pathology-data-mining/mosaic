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

# Copy project files
COPY pyproject.toml README.md app.py ./
COPY src/ ./src/

# Create venv and install with dependencies using GH_TOKEN secret
# Use BuildKit secret mount to securely pass GH_TOKEN
RUN --mount=type=secret,id=GH_TOKEN \
    uv venv && \
    if [ -f /run/secrets/GH_TOKEN ]; then \
        export GH_TOKEN=$(cat /run/secrets/GH_TOKEN) && \
        git config --global url."https://${GH_TOKEN}@github.com/".insteadOf "ssh://git@github.com/"; \
    fi && \
    uv pip install -e .

# Create non-root user for runtime
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

ENV PATH="/app/.venv/bin:$PATH"
# Set OMP_NUM_THREADS to a safe default to avoid libgomp errors
ENV OMP_NUM_THREADS=4

EXPOSE 7860

# Use mosaic CLI command which runs gradio_app:main (skips runtime pip install)
CMD ["mosaic"]
