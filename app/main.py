from fastapi import FastAPI
from app.api.routes import router
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Context-Aware Surveillance System API"
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": settings.VERSION}
