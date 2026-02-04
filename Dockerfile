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
    echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" >> ~/.ssh/known_hosts && \
    echo "github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=" >> ~/.ssh/known_hosts && \
    echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA1lW2R0dG8iB4Mf/p2hEQkMB/6EZHgqHwSLtxAzJPUhZDKkVLCXrF4mKEMEb0Cxl3lqbC3y4B5rBqR8YN/wOvBJlpYmJ9qG0w/xS0w9j8rFr6T8CRvNfQC9VJ7B7sPqrPB0nOmNJ7s9OZQBHVSqDmvTTmCqtQmjWCVvHpq0gCppQprEjL7PpCMdR9PbJtNJ7r9pXNvTQGEvQ1aHrCNSTYkqKhJjQJRBRMWJqDYg8=" >> ~/.ssh/known_hosts && \
    if [ -S "/run/buildkit/ssh_agent.sock" ] || [ -n "$SSH_AUTH_SOCK" ]; then \
        echo "Using SSH authentication for private repos"; \
    elif [ -f /run/secrets/GH_TOKEN ]; then \
        echo "Using GH_TOKEN authentication for private repos" && \
        export GH_TOKEN=$(cat /run/secrets/GH_TOKEN) && \
        TMP_GITCONFIG=$(mktemp) && \
        git config --file "$TMP_GITCONFIG" url."https://${GH_TOKEN}@github.com/".insteadOf "ssh://git@github.com/" && \
        git config --file "$TMP_GITCONFIG" url."https://${GH_TOKEN}@github.com/".insteadOf "git@github.com:"; \
    else \
        echo "Warning: No authentication method found - private repo access may fail"; \
    fi && \
    GIT_CONFIG_GLOBAL="${TMP_GITCONFIG:-}" uv pip install -e . && \
    if [ -n "${TMP_GITCONFIG:-}" ]; then rm -f "$TMP_GITCONFIG"; fi

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
