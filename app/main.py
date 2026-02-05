from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.api.metrics import router as metrics_router
from app.api.picks import router as picks_router
from app.api.snapshots import router as snapshots_router
from app.api.stats import router as stats_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(snapshots_router)
app.include_router(picks_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(stats_router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {"message": "ok"}
