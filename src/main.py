"""FastAPI entrypoint for Agent Memory System."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.api.ui_routes import router as ui_router
from src.config import get_settings
from src.db.postgres import close_postgres, get_postgres
from src.db.qdrant import close_qdrant, get_qdrant


def configure_logging() -> None:
    """Configure structured logging with structlog."""
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    configure_logging()
    logger = structlog.get_logger()

    # Startup
    logger.info("Starting Agent Memory System")

    try:
        # Initialize database connections
        await get_postgres()
        logger.info("PostgreSQL connected")

        await get_qdrant()
        logger.info("Qdrant connected")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down Agent Memory System")
        await close_postgres()
        await close_qdrant()
        logger.info("All connections closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Agent Memory System",
        description="HINDSIGHT-inspired Memory System for AI Agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)
    app.include_router(ui_router)

    # Mount static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
