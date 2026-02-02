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
COPY pyproject.toml README.md ./
COPY .git/ ./.git/
COPY src/ ./src/
COPY data/*.csv data/metadata/ ./data/

# Create venv and install with dependencies using SSH or GH_TOKEN secret
# Prioritize SSH if available (GitHub Actions), fallback to GH_TOKEN (local builds)
RUN --mount=type=ssh \
    --mount=type=secret,id=GH_TOKEN \
    uv venv && \
    mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null && \
    if [ -S "/run/buildkit/ssh_agent.sock" ] || [ -n "$SSH_AUTH_SOCK" ]; then \
        echo "Using SSH authentication for private repos"; \
    elif [ -f /run/secrets/GH_TOKEN ]; then \
        echo "Using GH_TOKEN authentication for private repos" && \
        export GH_TOKEN=$(cat /run/secrets/GH_TOKEN) && \
        git config --global url."https://${GH_TOKEN}@github.com/".insteadOf "ssh://git@github.com/" && \
        git config --global url."https://${GH_TOKEN}@github.com/".insteadOf "git@github.com:"; \
    else \
        echo "Warning: No authentication method found - private repo access may fail"; \
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
