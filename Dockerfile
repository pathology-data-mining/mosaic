FROM python:3.10-slim AS env-builder

# Install system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git gh \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Clone paladin repo
WORKDIR /deps
RUN --mount=type=secret,id=github_token \
    export GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    git clone --branch dev https://oauth2:$GITHUB_TOKEN@github.com/pathology-data-mining/paladin.git

# Create non-root user for runtime
RUN useradd -m -u 1000 user
RUN chown -R user:user /deps

WORKDIR /app
RUN chown -R user:user /app

USER user
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src/ ./src/

RUN uv sync 
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7877

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["gradio_app"]
