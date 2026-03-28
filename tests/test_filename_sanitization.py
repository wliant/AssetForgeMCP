"""Tests for filename sanitization and S3 key resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from asset_forge_mcp.files import resolve_s3_key, sanitize_filename
from asset_forge_mcp.models import AssetError


class TestSanitizeFilename:
    def test_lowercase(self):
        assert sanitize_filename("ForestSlime") == "forestslime"

    def test_spaces_to_underscores(self):
        assert sanitize_filename("forest slime idle") == "forest_slime_idle"

    def test_hyphens_to_underscores(self):
        assert sanitize_filename("forest-slime-idle") == "forest_slime_idle"

    def test_strips_special_chars(self):
        assert sanitize_filename("forest@slime!idle#1") == "forestslimeidle1"

    def test_collapses_multiple_underscores(self):
        assert sanitize_filename("forest___slime") == "forest_slime"

    def test_truncates_long_names(self):
        long_name = "a" * 100
        result = sanitize_filename(long_name)
        assert len(result) <= 64

    def test_empty_after_sanitization_raises(self):
        with pytest.raises(AssetError):
            sanitize_filename("@#$%^&")

    def test_strips_leading_trailing_underscores(self):
        assert sanitize_filename("_forest_") == "forest"


class TestResolveS3Key:
    @pytest.mark.asyncio
    async def test_no_collision(self):
        from asset_forge_mcp.s3_client import S3Storage
        mock = AsyncMock(spec=S3Storage)
        mock.key_exists = AsyncMock(return_value=False)
        result = await resolve_s3_key(mock, "sprites", "test_image")
        assert result == "sprites/test_image.png"

    @pytest.mark.asyncio
    async def test_collision_increments(self):
        from asset_forge_mcp.s3_client import S3Storage
        mock = AsyncMock(spec=S3Storage)
        mock.key_exists = AsyncMock(
            side_effect=lambda k: k == "sprites/test_image.png"
        )
        result = await resolve_s3_key(mock, "sprites", "test_image")
        assert result == "sprites/test_image_v2.png"

    @pytest.mark.asyncio
    async def test_path_traversal_sanitized_away(self):
        from asset_forge_mcp.s3_client import S3Storage
        mock = AsyncMock(spec=S3Storage)
        mock.key_exists = AsyncMock(return_value=False)
        result = await resolve_s3_key(mock, "sprites", "../../etc/passwd")
        assert ".." not in result
        assert result.startswith("sprites/")
