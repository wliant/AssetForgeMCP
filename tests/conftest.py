"""Shared test fixtures."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from asset_forge_mcp.config import Settings, clear_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test gets fresh settings."""
    clear_settings()
    yield
    clear_settings()


@pytest.fixture()
def settings() -> Settings:
    """Settings with S3 config and dummy API key."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-key-12345",
        "S3_ENDPOINT_URL": "http://localhost:9000",
        "S3_ACCESS_KEY": "testing",
        "S3_SECRET_KEY": "testing",
        "S3_BUCKET": "test-bucket",
        "S3_REGION": "us-east-1",
    }, clear=False):
        clear_settings()
        from asset_forge_mcp.config import get_settings
        return get_settings()


@pytest.fixture()
def mock_storage():
    """Provide a mock S3Storage for all tests that need it."""
    from asset_forge_mcp.s3_client import S3Storage
    from asset_forge_mcp.tools import set_storage

    mock = AsyncMock(spec=S3Storage)
    mock.bucket = "test-bucket"
    mock.key_exists = AsyncMock(return_value=False)
    mock.upload_bytes = AsyncMock()
    mock.upload_json = AsyncMock()
    set_storage(mock)
    yield mock
    set_storage(None)


def make_png_b64(width: int = 4, height: int = 4, mode: str = "RGBA") -> str:
    """Create a tiny valid PNG and return as base64."""
    img = Image.new(mode, (width, height), (255, 0, 0, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_png_bytes(width: int = 4, height: int = 4, mode: str = "RGBA") -> bytes:
    """Create a tiny valid PNG and return raw bytes."""
    img = Image.new(mode, (width, height), (255, 0, 0, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_test_png(path: Path, width: int = 4, height: int = 4, mode: str = "RGBA") -> None:
    """Save a tiny test PNG to disk."""
    img = Image.new(mode, (width, height), (255, 0, 0, 128))
    img.save(path, format="PNG")
