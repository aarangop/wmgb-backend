from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return HealthResponse(status="ok", version="0.1.0")
