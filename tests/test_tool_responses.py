"""Integration-level tests for tool response structure."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import TextContent

from asset_forge_mcp.models import AssetType
from asset_forge_mcp.openai_client import OpenAIImageClient
from asset_forge_mcp.tools import (
    generate_asset_variants,
    generate_game_asset,
    edit_game_asset,
    set_client,
)

from .conftest import make_png_b64


@pytest.fixture(autouse=True)
def mock_openai_client():
    """Provide a mock OpenAI client for all tests."""
    mock = AsyncMock(spec=OpenAIImageClient)
    mock.generate_image = AsyncMock(return_value=[make_png_b64()])
    mock.edit_image = AsyncMock(return_value=make_png_b64())
    set_client(mock)
    yield mock
    set_client(None)  # type: ignore[arg-type]


class TestGenerateGameAsset:
    @pytest.mark.asyncio
    async def test_returns_text_only(self, settings, mock_storage):
        result = await generate_game_asset(name="slime", prompt="a cute slime")
        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        meta = json.loads(result[0].text)
        assert meta["ok"] is True
        assert meta["tool"] == "generate_game_asset"
        assert meta["name"] == "slime"
        assert meta["bucket"] == "test-bucket"
        assert meta["keys"] == ["sprites/slime.png"]

    @pytest.mark.asyncio
    async def test_multiple_images(self, settings, mock_storage, mock_openai_client):
        mock_openai_client.generate_image.return_value = [make_png_b64(), make_png_b64()]
        result = await generate_game_asset(name="slime", prompt="a slime", n=2)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        meta = json.loads(result[0].text)
        assert meta["bucket"] == "test-bucket"
        assert len(meta["keys"]) == 2

    @pytest.mark.asyncio
    async def test_response_has_bucket_and_keys(self, settings, mock_storage):
        result = await generate_game_asset(
            name="slime", prompt="a slime", asset_type=AssetType.ICON,
        )
        meta = json.loads(result[0].text)
        assert "bucket" in meta
        assert "keys" in meta
        assert meta["keys"][0].startswith("icons/")


class TestEditGameAsset:
    @pytest.mark.asyncio
    async def test_returns_text_only(self, settings, mock_storage):
        b64_input = make_png_b64()
        result = await edit_game_asset(
            input_image=b64_input,
            prompt="make it blue",
        )
        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        meta = json.loads(result[0].text)
        assert meta["ok"] is True
        assert meta["tool"] == "edit_game_asset"
        assert meta["bucket"] == "test-bucket"
        assert "key" in meta

    @pytest.mark.asyncio
    async def test_response_has_bucket_and_key(self, settings, mock_storage):
        b64_input = make_png_b64()
        result = await edit_game_asset(input_image=b64_input, prompt="edit it")
        meta = json.loads(result[0].text)
        assert "bucket" in meta
        assert "key" in meta
        assert meta["key"].startswith("sprites/")

    @pytest.mark.asyncio
    async def test_invalid_base64_raises(self, settings, mock_storage):
        from asset_forge_mcp.models import AssetError
        with pytest.raises((AssetError, Exception)):
            await edit_game_asset(input_image="not-valid-base64!!", prompt="edit")


class TestGenerateAssetVariants:
    @pytest.mark.asyncio
    async def test_returns_text_only(self, settings, mock_storage, mock_openai_client):
        mock_openai_client.generate_image.return_value = [
            make_png_b64(), make_png_b64(), make_png_b64(),
        ]
        result = await generate_asset_variants(
            name="poison", prompt="poison icon", variant_count=3,
        )
        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        meta = json.loads(result[0].text)
        assert meta["completed_variants"] == 3
        assert meta["bucket"] == "test-bucket"
        assert len(meta["keys"]) == 3

    @pytest.mark.asyncio
    async def test_partial_failure(self, settings, mock_storage, mock_openai_client):
        from asset_forge_mcp.models import AssetError, ErrorCode
        # Batch fails, falls back to sequential; 1 of 2 sequential calls fails
        mock_openai_client.generate_image.side_effect = [
            AssetError(ErrorCode.OPENAI_BAD_RESPONSE, "batch not supported"),
            [make_png_b64()],
            Exception("timeout"),
        ]
        result = await generate_asset_variants(
            name="test", prompt="test", variant_count=2,
        )
        meta = json.loads(result[0].text)
        assert meta["completed_variants"] == 1
        assert meta["requested_variants"] == 2
        assert len(meta["warnings"]) == 1
        assert meta["bucket"] == "test-bucket"
        assert len(meta["keys"]) == 1

    @pytest.mark.asyncio
    async def test_response_has_bucket_and_keys(self, settings, mock_storage, mock_openai_client):
        mock_openai_client.generate_image.return_value = [make_png_b64(), make_png_b64()]
        result = await generate_asset_variants(
            name="icon", prompt="an icon", variant_count=2,
        )
        meta = json.loads(result[0].text)
        assert "bucket" in meta
        assert "keys" in meta
