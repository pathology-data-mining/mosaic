#!/bin/bash
set -e

# Enable BuildKit
export DOCKER_BUILDKIT=1

docker build -t mskmind/mosaic:latest .
