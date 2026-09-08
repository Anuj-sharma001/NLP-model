"""IndicLID Model Loader and Lifecycle Management.

This module handles loading, holding, and managing AI4Bharat IndicLID model instances
strictly decoupled from inference and request handling business logic.
"""

from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import fasttext

from app.config import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger("sakyti-nlp.models.indiclid")


class IndicLIDModelManager:
    """Manages the lifecycle of AI4Bharat IndicLID models (Native FTN and Roman FTR).

    Models are loaded once during service startup and shared across inference requests.
    """

    _instance: Optional["IndicLIDModelManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.ftn_model: Optional[Any] = None
        self.ftr_model: Optional[Any] = None
        self.ftn_path: Optional[Path] = None
        self.ftr_path: Optional[Path] = None
        self.is_loaded: bool = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "IndicLIDModelManager":
        """Get the singleton instance of IndicLIDModelManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        """Resolve model path relative to project root or current directory."""
        if path.is_absolute() and path.exists():
            return path

        # Try relative to cwd
        if path.exists():
            return path.resolve()

        # Try searching relative to common roots (nlp-service or workspace root)
        project_root = Path(__file__).resolve().parent.parent.parent
        alt_path = project_root / path
        if alt_path.exists():
            return alt_path.resolve()

        # If data path starts with ./data or data, try relative to project_root
        if str(path).startswith("data") or str(path).startswith("./data"):
            clean_part = str(path).lstrip("./")
            candidate = project_root / clean_part
            if candidate.exists():
                return candidate.resolve()

        return path.resolve()

    def load_models(self, settings: Optional[Settings] = None) -> None:
        """Load IndicLID models into memory once at startup.

        Loads:
        1. IndicLID-FTN: FastText model for native Indic scripts (Devanagari, Tamil, etc.)
        2. IndicLID-FTR: FastText model for Romanized Indic text and English
        """
        if self.is_loaded:
            logger.info("IndicLID models are already loaded.")
            return

        with self._load_lock:
            if self.is_loaded:
                return

            if settings is None:
                settings = get_settings()

            ftn_target = self._resolve_path(settings.indiclid_ftn_path)
            ftr_target = self._resolve_path(settings.indiclid_ftr_path)

            logger.info(
                "Initializing IndicLID models",
                extra={
                    "ftn_path": str(ftn_target),
                    "ftr_path": str(ftr_target),
                },
            )

            # 1. Load IndicLID-FTN
            if not ftn_target.exists():
                error_msg = f"IndicLID FTN model file not found at: {ftn_target}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            try:
                self.ftn_model = fasttext.load_model(str(ftn_target))
                self.ftn_path = ftn_target
                logger.info(
                    "IndicLID-FTN model successfully loaded",
                    extra={"path": str(ftn_target)},
                )
            except Exception as exc:
                logger.error(f"Failed to load IndicLID-FTN model: {exc}", exc_info=True)
                raise RuntimeError(f"Could not initialize IndicLID-FTN: {exc}") from exc

            # 2. Load IndicLID-FTR
            if not ftr_target.exists():
                error_msg = f"IndicLID FTR model file not found at: {ftr_target}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            try:
                self.ftr_model = fasttext.load_model(str(ftr_target))
                self.ftr_path = ftr_target
                logger.info(
                    "IndicLID-FTR model successfully loaded",
                    extra={"path": str(ftr_target)},
                )
            except Exception as exc:
                logger.error(f"Failed to load IndicLID-FTR model: {exc}", exc_info=True)
                raise RuntimeError(f"Could not initialize IndicLID-FTR: {exc}") from exc

            self.is_loaded = True
            logger.info("All IndicLID models successfully loaded and ready for inference.")

    def predict_native(self, text: str, k: int = 1) -> Tuple[List[str], List[float]]:
        """Run inference using IndicLID-FTN (native scripts)."""
        if not self.is_loaded or self.ftn_model is None:
            raise RuntimeError("IndicLID-FTN model is not loaded. Ensure service startup succeeded.")
        labels, probs = self.ftn_model.predict(text, k=k)
        return list(labels), [float(p) for p in probs]

    def predict_roman(self, text: str, k: int = 1) -> Tuple[List[str], List[float]]:
        """Run inference using IndicLID-FTR (Romanized Indic text & English)."""
        if not self.is_loaded or self.ftr_model is None:
            raise RuntimeError("IndicLID-FTR model is not loaded. Ensure service startup succeeded.")
        labels, probs = self.ftr_model.predict(text, k=k)
        return list(labels), [float(p) for p in probs]

    def get_status(self) -> Dict[str, str]:
        """Return readiness and diagnostic status of IndicLID models."""
        return {
            "indiclid_ftn": "loaded" if self.ftn_model is not None else "not_loaded",
            "indiclid_ftr": "loaded" if self.ftr_model is not None else "not_loaded",
        }
