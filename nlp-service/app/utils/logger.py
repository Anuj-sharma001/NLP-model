"""Structured logging module for SAKYTI NLP Service."""

import datetime
import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include standard exception information if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include extra custom attributes attached to the LogRecord
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message",
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extras:
            log_payload["extra"] = extras

        return json.dumps(log_payload, default=str)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> logging.Logger:
    """Configures structured logging for the application and third-party libraries."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is invoked multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if log_format.lower() == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress excessive noise from dependencies
    logging.getLogger("uvicorn.access").handlers = [console_handler]
    logging.getLogger("uvicorn.error").handlers = [console_handler]

    logger = logging.getLogger("sakyti-nlp")
    logger.setLevel(numeric_level)
    return logger


def get_logger(name: str = "sakyti-nlp") -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
