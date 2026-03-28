"""Integration-level tests for tool response structure."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

from .conftest import make_png_b64, save_test_png


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
        assert len(meta["files"]) == 1

    @pytest.mark.asyncio
    async def test_saves_to_disk(self, settings):
        result = await generate_game_asset(name="slime", prompt="a cute slime")
        meta = json.loads(result[0].text)
        img_path = Path(meta["files"][0]["image_path"])
        meta_path = Path(meta["files"][0]["metadata_path"])
        assert img_path.exists()
        assert meta_path.exists()

    @pytest.mark.asyncio
    async def test_multiple_images(self, settings, mock_openai_client):
        mock_openai_client.generate_image.return_value = [make_png_b64(), make_png_b64()]
        result = await generate_game_asset(name="slime", prompt="a slime", n=2)
        # 1 text + 2 images
        assert len(result) == 3
        assert isinstance(result[1], ImageContent)
        assert isinstance(result[2], ImageContent)

    @pytest.mark.asyncio
    async def test_correct_output_dir(self, settings):
        result = await generate_game_asset(
            name="slime", prompt="a slime", asset_type=AssetType.ICON,
        )
        meta = json.loads(result[0].text)
        assert "icons" in meta["files"][0]["image_path"]


class TestEditGameAsset:
    @pytest.mark.asyncio
    async def test_returns_text_and_image(self, settings, tmp_output_dir):
        src = tmp_output_dir / "source.png"
        save_test_png(src)

        result = await edit_game_asset(
            input_path=str(src),
            prompt="make it blue",
        )
        assert len(result) == 2
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)

        meta = json.loads(result[0].text)
        assert meta["ok"] is True
        assert meta["tool"] == "edit_game_asset"

    @pytest.mark.asyncio
    async def test_output_in_same_dir_as_input(self, settings, tmp_output_dir):
        src = tmp_output_dir / "source.png"
        save_test_png(src)

        result = await edit_game_asset(input_path=str(src), prompt="edit it")
        meta = json.loads(result[0].text)
        out_path = Path(meta["files"][0]["image_path"])
        assert out_path.parent == src.parent

    @pytest.mark.asyncio
    async def test_missing_input_raises(self, settings):
        from asset_forge_mcp.models import AssetError
        with pytest.raises(AssetError):
            await edit_game_asset(input_path="/nonexistent.png", prompt="edit")


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
        assert len(meta["files"]) == 3

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

    @pytest.mark.asyncio
    async def test_variant_naming(self, settings, mock_openai_client):
        mock_openai_client.generate_image.return_value = [make_png_b64(), make_png_b64()]
        result = await generate_asset_variants(
            name="icon", prompt="an icon", variant_count=2,
        )
        meta = json.loads(result[0].text)
        paths = [f["image_path"] for f in meta["files"]]
        assert any("icon_1" in p for p in paths)
        assert any("icon_2" in p for p in paths)
