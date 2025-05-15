# # filepath: /Users/andresap/repos/whos-my-good-boy/backend/app/__init__.py

# """
# Main application package initialization
# """

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from .api.routes import health, predictions

# app = FastAPI()

# # CORS settings
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this as needed for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include API routes
# app.include_router(health.router)
# app.include_router(predictions.router)
