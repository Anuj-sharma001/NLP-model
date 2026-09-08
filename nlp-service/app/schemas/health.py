"""Health check schema definition."""

from datetime import datetime, timezone
from typing import Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for service health check response."""

    status: str = Field(default="healthy", description="Current service health status")
    app_name: str = Field(..., description="Service identifier name")
    version: str = Field(..., description="Application semantic version")
    environment: str = Field(..., description="Active runtime environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of health check execution in UTC",
    )
    models: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Health and readiness status of loaded NLP models",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "app_name": "SAKYTI Multilingual NLP Service",
                "version": "0.1.0",
                "environment": "development",
                "timestamp": "2026-09-07T22:30:00.000000Z",
                "models": {},
            }
        }
    }
