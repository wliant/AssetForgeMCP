"""Tests for filename sanitization and filepath resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from asset_forge_mcp.files import resolve_filepath, sanitize_filename
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


class TestResolveFilepath:
    def test_no_collision(self, tmp_path: Path):
        result = resolve_filepath(tmp_path, "test_image")
        assert result == tmp_path / "test_image.png"

    def test_collision_increments(self, tmp_path: Path):
        (tmp_path / "test_image.png").touch()
        result = resolve_filepath(tmp_path, "test_image")
        assert result == tmp_path / "test_image_v2.png"

    def test_multiple_collisions(self, tmp_path: Path):
        (tmp_path / "test_image.png").touch()
        (tmp_path / "test_image_v2.png").touch()
        result = resolve_filepath(tmp_path, "test_image")
        assert result == tmp_path / "test_image_v3.png"

    def test_path_traversal_sanitized_away(self, tmp_path: Path):
        # Dots and slashes are stripped by sanitize_filename, so traversal
        # is inherently prevented — the result is a safe flat name.
        result = resolve_filepath(tmp_path, "../../etc/passwd")
        assert result.parent == tmp_path
        assert ".." not in str(result.name)
