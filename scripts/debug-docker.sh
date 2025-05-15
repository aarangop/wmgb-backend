#!/bin/bash

# Script to run Docker container with extra debugging
echo "Running Docker container with more verbose logging..."

# Navigate to the script directory
cd "$(dirname "$0")"

# Set environment variable for more verbose logging
export LOG_LEVEL=DEBUG

# Run docker-compose with run instead of up to attach to the terminal
docker-compose run --service-ports app
