from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import APP_TITLE, APP_VERSION, CORS_ORIGINS
from database.connection import initialize_database, load_cases, load_reports


@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize_database()
    application.state.cases = load_cases()
    application.state.reports = load_reports()
    application.state.evidence_files = {}
    application.state.evidence_tokens = {}
    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="DeepTrace AI media forensics dashboard",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
