"""Shared test fixtures."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

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
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    d = tmp_path / "assets" / "generated"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def settings(tmp_output_dir: Path) -> Settings:
    """Settings with a temp output dir and dummy API key."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-key-12345",
        "ASSET_OUTPUT_DIR": str(tmp_output_dir),
    }, clear=False):
        clear_settings()
        from asset_forge_mcp.config import get_settings
        return get_settings()


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
