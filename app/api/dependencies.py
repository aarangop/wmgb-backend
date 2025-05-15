from fastapi import Depends, HTTPException, status
from app.services.general_classifier import GeneralClassifierService
from app.services.apolo_classifier import ApoloClassifierService

# Create instances of our services
general_classifier_service = GeneralClassifierService()
apolo_classifier_service = ApoloClassifierService()


def get_general_classifier_service():
    return general_classifier_service


def get_apolo_classifier_service():
    return apolo_classifier_service
