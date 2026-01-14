FROM python:3.10-slim AS env-builder

# Install system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Setup SSH for git (as root)
RUN mkdir -p /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create venv and install with dependencies (as root, before switching users)
RUN --mount=type=ssh \
    uv venv && \
    uv pip install -e .

# Create non-root user for runtime
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7877

ENTRYPOINT ["gradio_app"]
