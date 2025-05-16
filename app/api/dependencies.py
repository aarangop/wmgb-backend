from fastapi import Depends, HTTPException, status
from app.services.general_classifier import GeneralClassifierService
from app.services.apolo_classifier import ApoloClassifierService
from app.core.config import config
from app.utils.inference_models.model_repository import CachingModelRepository, LocalCacheRepository

from app.utils.inference_models.repository_factory import create_model_repository

model_repository = create_model_repository()

# Create instances of our services
general_classifier_service = GeneralClassifierService(model_repository)
apolo_classifier_service = ApoloClassifierService(model_repository)


def get_general_classifier_service():
    return general_classifier_service


def get_apolo_classifier_service():
    return apolo_classifier_service
