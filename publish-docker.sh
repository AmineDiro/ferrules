#!/bin/bash

# Configuration
DOCKER_HUB_USERNAME="aminediro"
VERSION="0.1.8"

# Push CPU version
echo "Pushing CPU version..."
docker push ${DOCKER_HUB_USERNAME}/ferrules-api:${VERSION}
docker push ${DOCKER_HUB_USERNAME}/ferrules-api:latest

# Push GPU version
echo "Pushing GPU version..."
docker push ${DOCKER_HUB_USERNAME}/ferrules-api-gpu:${VERSION}
docker push ${DOCKER_HUB_USERNAME}/ferrules-api-gpu:latest

echo "All images pushed successfully!"
