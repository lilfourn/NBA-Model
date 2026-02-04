from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.snapshots import router as snapshots_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(snapshots_router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {"message": "ok"}
