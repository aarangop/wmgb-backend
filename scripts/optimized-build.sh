#!/bin/bash

# Script to run optimized builds locally
# This script builds Docker images with BuildKit optimizations

echo "🚀 Starting optimized Docker build"

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build specific target
build_target() {
    local target=$1
    local tag="wmgb-backend:$1"
    
    echo "📦 Building $target..."
    
    podman build \
        --target $target \
        --tag $tag \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from $tag \
        --file Dockerfile \
        .
}

# Check which target to build
if [ "$1" == "deps" ]; then
    build_target "dev-dependencies"
elif [ "$1" == "unit-test" ]; then
    build_target "unit-test"
elif [ "$1" == "integration-test" ]; then
    build_target "integration-test"
elif [ "$1" == "dev" ]; then
    build_target "development"
elif [ "$1" == "prod" ]; then
    build_target "production"
else
    echo "Usage: $0 [deps|unit-test|integration-test|dev|prod]"
    echo "Example: $0 unit-test"
    exit 1
fi

echo "✅ Build complete!"
