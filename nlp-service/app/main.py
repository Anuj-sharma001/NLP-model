"""SAKYTI Multilingual NLP Microservice entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_v1_router
from app.config import Settings, get_settings
from app.schemas.health import HealthResponse
from app.utils.logger import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown routines."""
    settings = get_settings()
    logger = setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
    )
    logger.info(
        "Starting SAKYTI Multilingual NLP Service",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "env": settings.app_env,
            "debug": settings.debug,
        },
    )

    # Load IndicLID models once during startup
    try:
        from app.models.indiclid_model import IndicLIDModelManager

        model_manager = IndicLIDModelManager.get_instance()
        model_manager.load_models(settings)
        logger.info(
            "IndicLID models loaded successfully during startup",
            extra={"models": model_manager.get_status()},
        )
    except Exception as exc:
        logger.error(
            f"Failed to load IndicLID models during startup: {exc}",
            exc_info=True,
        )

    # Load IndicTrans2 models once during startup
    try:
        from app.models.indictrans_model import IndicTransModelManager

        trans_manager = IndicTransModelManager.get_instance()
        trans_manager.initialize(settings)
        logger.info(
            "IndicTrans2 models loaded successfully during startup",
            extra={"translation_models": trans_manager.get_model_info()},
        )
    except Exception as exc:
        logger.error(
            f"Failed to load IndicTrans2 models during startup: {exc}",
            exc_info=True,
        )

    yield

    logger.info("Shutting down SAKYTI Multilingual NLP Service")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for FastAPI instance."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-oriented Multilingual NLP layer for Ayurveda applications.",
        docs_url="/docs" if settings.debug or settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.debug or settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # Enable Request ID correlation tracing
    from app.utils.request_id import RequestIDMiddleware, get_current_request_id
    app.add_middleware(RequestIDMiddleware)

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Structured Validation Exception Handler
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from datetime import datetime, timezone

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        logger = get_logger()
        logger.warning(
            f"Validation error on {request.method} {request.url.path} [{req_id}]: {exc.errors()}",
            extra={"path": str(request.url.path), "method": request.method, "request_id": req_id},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request payload validation failed.",
                    "details": exc.errors(),
                    "request_id": req_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                # Backward compatibility detail field
                "detail": exc.errors(),
            },
            headers={"X-Request-ID": req_id},
        )

    # Structured HTTP Exception Handler
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        logger = get_logger()
        logger.warning(
            f"HTTP {exc.status_code} on {request.method} {request.url.path} [{req_id}]: {exc.detail}",
            extra={"path": str(request.url.path), "method": request.method, "request_id": req_id},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "details": None,
                    "request_id": req_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                # Backward compatibility detail field
                "detail": exc.detail,
            },
            headers={"X-Request-ID": req_id},
        )

    # Global Unhandled Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or get_current_request_id()
        logger = get_logger()
        logger.error(
            f"Unhandled exception during request {request.method} {request.url.path} [{req_id}]: {exc}",
            exc_info=True,
            extra={"path": str(request.url.path), "method": request.method, "request_id": req_id},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing the request.",
                    "details": str(exc) if settings.debug else None,
                    "request_id": req_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "detail": "An unexpected error occurred while processing the request.",
            },
            headers={"X-Request-ID": req_id},
        )

    # Health Check Endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        tags=["Health"],
        status_code=status.HTTP_200_OK,
    )
    async def health_check(
        current_settings: Settings = Depends(get_settings),
    ) -> HealthResponse:
        """Verify service operational status and metadata."""
        from app.models.indiclid_model import IndicLIDModelManager
        from app.models.indictrans_model import IndicTransModelManager

        lid_models_status = IndicLIDModelManager.get_instance().get_status()
        trans_manager = IndicTransModelManager.get_instance()
        trans_status = "loaded" if trans_manager.is_loaded else ("error" if trans_manager._load_error else "not_loaded")

        models_status = {
            **lid_models_status,
            "indictrans2_indic_en": trans_status,
            "indictrans2_en_indic": trans_status,
        }
        service_status = (
            "healthy"
            if any(v == "loaded" for v in lid_models_status.values()) and trans_manager.is_loaded
            else "degraded"
        )
        return HealthResponse(
            status=service_status,
            app_name=current_settings.app_name,
            version=current_settings.app_version,
            environment=current_settings.app_env,
            models=models_status,
        )

    # Include API Routers
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.debug,
    )
