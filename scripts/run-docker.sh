#!/bin/bash

# Script to build and run the Docker container

# Navigate to the script directory
cd "$(dirname "$0")"

# Build the Docker image
echo "Building Docker image..."
docker-compose build

# Run the container
echo "Starting container..."
docker-compose up

# You can uncomment the line below to run in detached mode instead
# docker-compose up -d
