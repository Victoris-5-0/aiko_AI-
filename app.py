from fastapi import FastAPI

from api.parent_routes import router as parent_router
from api.recommendation_routes import router as recommendation_router

app = FastAPI(
    title="AIKO AI Service",
    version="1.0.0"
)

app.include_router(
    parent_router,
    prefix="/parent",
    tags=["Parent LLM"]
)

app.include_router(
    recommendation_router,
    prefix="/recommendation",
    tags=["Recommendation"]
)


@app.get("/health")
def health():
    return {"status": "ok"}