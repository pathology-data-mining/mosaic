#!/bin/bash
set -e

# Load GH_TOKEN from .gh_env file
if [ -f .gh_env ]; then
    source .gh_env
else
    echo "Error: .gh_env file not found"
    exit 1
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with secret mount for GH_TOKEN
docker build \
    --secret id=GH_TOKEN,env=GH_TOKEN \
    -t mskmind/mosaic:latest \
    .
