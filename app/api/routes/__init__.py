# filepath: /Users/andresap/repos/whos-my-good-boy/backend/app/api/routes/__init__.py

"""
API routes package initialization
"""

from .predictions import router as predictions_router
from .health import router as health_router
from fastapi import APIRouter

router = APIRouter()


router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(predictions_router,
                      prefix="/predictions", tags=["predictions"])
