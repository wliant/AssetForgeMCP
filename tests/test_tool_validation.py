"""Tests for tool input validation."""

from __future__ import annotations

import pytest

from asset_forge_mcp.models import AssetError, AssetType, ImageSize, StyleHint


class TestEnumValues:
    def test_asset_type_values(self):
        assert set(AssetType) == {
            AssetType.SPRITE, AssetType.ICON, AssetType.PORTRAIT,
            AssetType.BACKGROUND, AssetType.TILE, AssetType.UI,
        }

    def test_invalid_asset_type(self):
        with pytest.raises(ValueError):
            AssetType("weapon")

    def test_image_size_values(self):
        assert ImageSize.S_1024x1024.value == "1024x1024"
        assert ImageSize.AUTO.value == "auto"

    def test_style_values(self):
        assert StyleHint.PIXEL_ART.value == "pixel-art"
        assert StyleHint.SEMI_REALISTIC.value == "semi-realistic"


class TestToolBounds:
    @pytest.mark.asyncio
    async def test_n_too_low(self, settings):
        from asset_forge_mcp.tools import generate_game_asset
        with pytest.raises(AssetError, match="n must be between"):
            await generate_game_asset(name="test", prompt="test", n=0)

    @pytest.mark.asyncio
    async def test_n_too_high(self, settings):
        from asset_forge_mcp.tools import generate_game_asset
        with pytest.raises(AssetError, match="n must be between"):
            await generate_game_asset(name="test", prompt="test", n=9)

    @pytest.mark.asyncio
    async def test_variant_count_too_low(self, settings):
        from asset_forge_mcp.tools import generate_asset_variants
        with pytest.raises(AssetError, match="variant_count must be between"):
            await generate_asset_variants(name="test", prompt="test", variant_count=0)

    @pytest.mark.asyncio
    async def test_variant_count_too_high(self, settings):
        from asset_forge_mcp.tools import generate_asset_variants
        with pytest.raises(AssetError, match="variant_count must be between"):
            await generate_asset_variants(name="test", prompt="test", variant_count=9)
