import boto3
import os
from loguru import logger

from app.core.config import config


def create_s3_client(region_name=None):
    """
    Create an S3 client that works in both development and production.

    In production (Fargate): Uses the task role credentials automatically
    In development: Uses AWS Toolkit profile or environment variables
    """
    region = region_name or config.AWS_REGION

    # Check if running in AWS environment (ECS/Fargate)
    # This environment variable is automatically set in ECS
    if os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
        logger.info("Running in AWS environment - using task role credentials")
        return boto3.client('s3', region_name=region)

    # Development environment
    profile_name = os.getenv("AWS_PROFILE", "whos-my-good-boy-dev")

    # If using AWS Toolkit or profile exists
    try:
        logger.info(f"Using AWS profile: {profile_name}")
        session = boto3.Session(profile_name=profile_name, region_name=region)
        return session.client('s3')
    except Exception as e:
        logger.warning(f"Failed to use profile {profile_name}: {e}")

    # Fallback to default credential chain (env vars, etc.)
    logger.info("Using default credential chain")
    return boto3.client('s3', region_name=region)
