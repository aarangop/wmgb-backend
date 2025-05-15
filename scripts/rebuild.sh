#!/bin/bash

# Script to force rebuild the Docker container from scratch
echo "Force rebuilding Docker image without cache..."

# Navigate to the script directory
cd "$(dirname "$0")"

# Run docker-compose with the --no-cache flag
docker-compose build --no-cache

# Start the container
echo "Starting container..."
docker-compose up
