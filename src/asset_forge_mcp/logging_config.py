"""Logging configuration with secret redaction."""

from __future__ import annotations

import logging
import re


class SecretRedactFilter(logging.Filter):
    """Redact patterns that look like API keys from log messages."""

    _pattern = re.compile(r"sk-[A-Za-z0-9_-]{10,}")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._pattern.sub("sk-***REDACTED***", record.msg)
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent format and secret redaction."""
    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
