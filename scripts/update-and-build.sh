#!/bin/bash

# Script to update poetry lock file with Python 3.10 and build Docker image
echo "Updating poetry lock file with Python 3.10..."

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed."
    echo "Please install pyenv first: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# Check if Python 3.10.17 is installed via pyenv
if ! pyenv versions | grep -q "3.10.17"; then
    echo "Warning: Python 3.10.17 is not installed via pyenv."
    
    # Check if any Python 3.10.x is available via regular system installation
    if command -v python3.10 &> /dev/null; then
        echo "Found system Python 3.10, will use that instead..."
        # Override the Python path to use system Python 3.10
        USE_SYSTEM_PYTHON=1
    else
        echo "Please install Python 3.10.17 with: pyenv install 3.10.17"
        exit 1
    fi
else
    USE_SYSTEM_PYTHON=0
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed."
    echo "Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Set Python version based on what we have available
if [ "$USE_SYSTEM_PYTHON" -eq 1 ]; then
    echo "Using system Python 3.10..."
    export PYTHON=$(which python3.10)
else
    # Use pyenv to select Python 3.10.17 specifically and ensure we're using it consistently
    echo "Setting Python 3.10.17 as global Python version..."
    pyenv global 3.10.17
    export PYTHON=$(which python)
fi

# Verify Python version
echo "Using Python version:"
python --version
echo "Python path: $PYTHON"

# Make sure poetry is using the right Python version
poetry env use $PYTHON

# Clean existing lock file
echo "Removing existing poetry.lock file..."
rm -f poetry.lock

# Generate a fresh lock file with Python 3.10
echo "Generating new poetry.lock file with Python 3.10..."
poetry lock

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    # Check for new Docker Compose V2 format (docker compose)
    if ! docker compose version &> /dev/null; then
        echo "Error: docker-compose is not installed."
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    else
        # If Docker Compose V2 is available, use that format
        echo "Using Docker Compose V2..."
        COMPOSE_CMD="docker compose"
    fi
else
    COMPOSE_CMD="docker-compose"
fi

# Build and run with docker-compose
echo "Building and running Docker container..."
$COMPOSE_CMD build
$COMPOSE_CMD up
