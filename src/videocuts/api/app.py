"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from videocuts.api.database import init_db
from videocuts.api.routes import router as projects_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events."""
    logger.info("Starting VideoCuts API...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down VideoCuts API...")


app = FastAPI(
    title="VideoCuts API",
    description="API for generating viral short-form video clips from long-form content",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects_router)

# Mount static files
from videocuts.api.settings import get_settings
settings = get_settings()
os.makedirs(settings.storage_path, exist_ok=True)
app.mount("/storage", StaticFiles(directory=settings.storage_path), name="storage")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
