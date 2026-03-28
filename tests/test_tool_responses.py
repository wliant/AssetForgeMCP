"""Integration-level tests for tool response structure."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import ImageContent, TextContent

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
    async def test_returns_text_and_image(self, settings):
        result = await generate_game_asset(name="slime", prompt="a cute slime")
        assert len(result) == 2
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)

        meta = json.loads(result[0].text)
        assert meta["ok"] is True
        assert meta["tool"] == "generate_game_asset"
        assert meta["name"] == "slime"
        assert "files" not in meta

    @pytest.mark.asyncio
    async def test_multiple_images(self, settings, mock_openai_client):
        mock_openai_client.generate_image.return_value = [make_png_b64(), make_png_b64()]
        result = await generate_game_asset(name="slime", prompt="a slime", n=2)
        # 1 text + 2 images
        assert len(result) == 3
        assert isinstance(result[1], ImageContent)
        assert isinstance(result[2], ImageContent)

        meta = json.loads(result[0].text)
        assert "files" not in meta

    @pytest.mark.asyncio
    async def test_response_has_no_paths(self, settings):
        result = await generate_game_asset(
            name="slime", prompt="a slime", asset_type=AssetType.ICON,
        )
        meta = json.loads(result[0].text)
        assert "files" not in meta
        assert "image_path" not in json.dumps(meta)
        assert "metadata_path" not in json.dumps(meta)


class TestEditGameAsset:
    @pytest.mark.asyncio
    async def test_returns_text_and_image(self, settings):
        b64_input = make_png_b64()
        result = await edit_game_asset(
            input_image=b64_input,
            prompt="make it blue",
        )
        assert len(result) == 2
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)

        meta = json.loads(result[0].text)
        assert meta["ok"] is True
        assert meta["tool"] == "edit_game_asset"
        assert "files" not in meta
        assert "input_path" not in meta

    @pytest.mark.asyncio
    async def test_response_has_no_paths(self, settings):
        b64_input = make_png_b64()
        result = await edit_game_asset(input_image=b64_input, prompt="edit it")
        text = result[0].text
        assert "path" not in text.lower()

    @pytest.mark.asyncio
    async def test_invalid_base64_raises(self, settings):
        from asset_forge_mcp.models import AssetError
        with pytest.raises((AssetError, Exception)):
            await edit_game_asset(input_image="not-valid-base64!!", prompt="edit")


class TestGenerateAssetVariants:
    @pytest.mark.asyncio
    async def test_returns_multiple_images(self, settings, mock_openai_client):
        mock_openai_client.generate_image.return_value = [
            make_png_b64(), make_png_b64(), make_png_b64(),
        ]
        result = await generate_asset_variants(
            name="poison", prompt="poison icon", variant_count=3,
        )
        # 1 text + 3 images
        assert len(result) == 4
        meta = json.loads(result[0].text)
        assert meta["completed_variants"] == 3
        assert "files" not in meta

    @pytest.mark.asyncio
    async def test_partial_failure(self, settings, mock_openai_client):
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
        assert "files" not in meta

    @pytest.mark.asyncio
    async def test_response_has_no_paths(self, settings, mock_openai_client):
        mock_openai_client.generate_image.return_value = [make_png_b64(), make_png_b64()]
        result = await generate_asset_variants(
            name="icon", prompt="an icon", variant_count=2,
        )
        meta = json.loads(result[0].text)
        assert "files" not in meta
        assert "path" not in json.dumps(meta).lower()
