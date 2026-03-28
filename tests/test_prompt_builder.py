"""Tests for prompt construction."""

from __future__ import annotations

from asset_forge_mcp.models import AssetType, BackgroundType, StyleHint
from asset_forge_mcp.prompts import build_edit_prompt, build_generation_prompt


class TestBuildGenerationPrompt:
    def test_basic_structure(self):
        result = build_generation_prompt(
            "cute slime enemy",
            AssetType.SPRITE,
            StyleHint.PIXEL_ART,
            BackgroundType.OPAQUE,
        )
        assert result.startswith("Create a game-ready pixel-art sprite.")
        assert "cute slime enemy" in result
        assert "No text, no watermark, no mockup." in result

    def test_transparent_background(self):
        result = build_generation_prompt(
            "test", AssetType.SPRITE, StyleHint.VECTOR, BackgroundType.TRANSPARENT,
        )
        assert "Transparent background." in result

    def test_opaque_no_transparent_instruction(self):
        result = build_generation_prompt(
            "test", AssetType.BACKGROUND, StyleHint.PAINTERLY, BackgroundType.OPAQUE,
        )
        assert "Transparent background." not in result

    def test_compact_types_get_silhouette(self):
        for at in [AssetType.SPRITE, AssetType.ICON, AssetType.TILE, AssetType.UI]:
            result = build_generation_prompt("test", at, StyleHint.VECTOR, BackgroundType.AUTO)
            assert "Clean silhouette" in result, f"Missing for {at}"

    def test_non_compact_types_no_silhouette(self):
        for at in [AssetType.PORTRAIT, AssetType.BACKGROUND]:
            result = build_generation_prompt("test", at, StyleHint.VECTOR, BackgroundType.AUTO)
            assert "Clean silhouette" not in result, f"Unexpected for {at}"


class TestBuildEditPrompt:
    def test_structure(self):
        result = build_edit_prompt("make it more menacing")
        assert result.startswith("Edit the provided game asset.")
        assert "Preserve overall readability." in result
        assert "Do not add text, watermark, frame, or mockup." in result
        assert "make it more menacing" in result
