"""
Production logging configuration.

Creates three handlers:
  logs/app.log    — JSON-structured, INFO+,  rotating 10 MB × 5 files
  logs/error.log  — JSON-structured, ERROR+, rotating  5 MB × 3 files
  console         — human-readable colour output, level from LOG_LEVEL env var

Usage:
    from utils.logging_config import setup_logging
    setup_logging()     # call once at process start (main.py)
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path

LOG_DIR   = Path("logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record — machine-parseable for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict = {
            "ts"    : round(time.time(), 3),
            "level" : record.levelname,
            "logger": record.name,
            "msg"   : record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            obj["stack"] = record.stack_info
        return json.dumps(obj, ensure_ascii=False)


class _ReadableFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG"   : "\033[36m",
        "INFO"    : "\033[32m",
        "WARNING" : "\033[33m",
        "ERROR"   : "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        ts    = time.strftime("%H:%M:%S", time.localtime(record.created))
        line  = (
            f"{ts}  "
            f"{color}{record.levelname:<8}{self._RESET}  "
            f"{record.name:<30}  "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging() -> None:
    """Configure the root logger for production. Call exactly once at startup."""
    LOG_DIR.mkdir(exist_ok=True)

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    root  = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # ── logs/app.log — INFO+ JSON, rotating 10 MB × 5 ────────────────────────
    app_h = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes    = 10 * 1024 * 1024,
        backupCount = 5,
        encoding    = "utf-8",
    )
    app_h.setLevel(logging.INFO)
    app_h.setFormatter(_JsonFormatter())

    # ── logs/error.log — ERROR+ JSON, rotating 5 MB × 3 ─────────────────────
    err_h = logging.handlers.RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes    = 5 * 1024 * 1024,
        backupCount = 3,
        encoding    = "utf-8",
    )
    err_h.setLevel(logging.ERROR)
    err_h.setFormatter(_JsonFormatter())

    # ── console — human-readable colour ──────────────────────────────────────
    con_h = logging.StreamHandler()
    con_h.setLevel(level)
    con_h.setFormatter(_ReadableFormatter())

    root.addHandler(app_h)
    root.addHandler(err_h)
    root.addHandler(con_h)

    # Suppress noisy third-party libraries
    for noisy in ("aiortc", "aioice", "aiohttp", "urllib3",
                  "ultralytics", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging ready  level=%s  dir=%s", LOG_LEVEL, LOG_DIR.resolve()
    )
