#!/bin/bash

# Configuration
DOCKER_HUB_USERNAME="aminediro"
VERSION="0.1.8"

# Build CPU version
echo "Building CPU version..."
docker build -f Dockerfile \
  --build-arg RUST_VERSION=stable \
  --build-arg DEBIAN_VERSION=bookworm \
  -t ${DOCKER_HUB_USERNAME}/ferrules-api:${VERSION} \
  -t ${DOCKER_HUB_USERNAME}/ferrules-api:latest \
  .

# Build GPU version
echo "Building GPU version..."
docker build -f Dockerfile.gpu \
  --build-arg CUDA_VERSION=12.3.2 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg RUST_VERSION=stable \
  -t ${DOCKER_HUB_USERNAME}/ferrules-api-gpu:${VERSION} \
  -t ${DOCKER_HUB_USERNAME}/ferrules-api-gpu:latest \
  .

echo "Build complete!"
