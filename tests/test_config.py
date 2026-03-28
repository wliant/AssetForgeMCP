"""Tests for config loading and validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from asset_forge_mcp.config import Settings, clear_settings, get_settings


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        clear_settings()
        with pytest.raises(Exception):  # ValidationError
            Settings()  # type: ignore[call-arg]


def test_defaults_applied():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        clear_settings()
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_base_url == "https://api.openai.com/v1"
        assert s.openai_image_model == "gpt-image-1"
        assert s.mcp_host == "0.0.0.0"
        assert s.mcp_port == 8080
        assert s.log_level == "INFO"


def test_secret_str_hides_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-value"}, clear=True):
        clear_settings()
        s = Settings()  # type: ignore[call-arg]
        assert "sk-secret-value" not in str(s.openai_api_key)
        assert "sk-secret-value" not in repr(s.openai_api_key)
        assert s.openai_api_key.get_secret_value() == "sk-secret-value"


def test_env_override():
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test",
        "MCP_PORT": "9090",
        "LOG_LEVEL": "DEBUG",
    }, clear=True):
        clear_settings()
        s = Settings()  # type: ignore[call-arg]
        assert s.mcp_port == 9090
        assert s.log_level == "DEBUG"
