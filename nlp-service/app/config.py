"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SAKYTI NLP service configuration schema."""

    # Application Settings
    app_name: str = "SAKYTI Multilingual NLP Service"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_v1_prefix: str = "/api/v1"

    # Logging Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Model Storage & Paths
    models_dir: Path = Path("./data/models")
    indiclid_ftn_path: Path = Path("./data/models/indiclid-ftn/model_baseline_roman.bin")
    indiclid_ftr_path: Path = Path("./data/models/indiclid-ftr/model_baseline_roman.bin")
    indictrans2_model_path: Path = Path("./data/models/indictrans2")

    # Language Detection Parameters
    language_detection_confidence_threshold: float = 0.40
    language_detection_roman_threshold: float = 0.50

    # IndicTrans2 Translation Settings
    indictrans2_indic_en_model: str = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
    indictrans2_en_indic_model: str = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
    indictrans2_max_input_length: int = 512
    indictrans2_max_output_length: int = 512
    indictrans2_num_beams: int = 5
    translation_timeout_seconds: float = 30.0

    # Hardware & Concurrency Settings
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    num_workers: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
