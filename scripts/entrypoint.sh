#!/bin/bash
set -e

# First download models from S3
python /app/scripts/download_models.py "$@"

# The download_models.py script will execute the command passed to it
# If it fails, this script won't continue due to set -e