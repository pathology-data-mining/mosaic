#!/bin/bash
set -e

# Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with SSH forwarding
docker build --ssh default=$SSH_AUTH_SOCK -t mskmind/mosaic:latest .

# Kill SSH agent
eval "$(ssh-agent -k)"
