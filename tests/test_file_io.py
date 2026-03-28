"""Tests for file I/O: save, validate, mask checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from asset_forge_mcp.files import (
    save_asset, save_metadata, validate_input_image, validate_mask,
    validate_image_bytes, validate_mask_bytes,
)
from asset_forge_mcp.models import AssetError, AssetMetadata, ErrorCode

from .conftest import make_png_b64, make_png_bytes, save_test_png


class TestSaveAsset:
    def test_writes_png(self, tmp_path: Path):
        b64 = make_png_b64()
        filepath = tmp_path / "test.png"
        save_asset(b64, filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_save_metadata_writes_json(self, tmp_path: Path):
        from datetime import datetime, timezone
        meta = AssetMetadata(
            name="test",
            tool="generate_game_asset",
            model="gpt-image-1",
            asset_type="sprite",
            prompt="a slime",
            final_prompt="Create a game-ready pixel-art sprite. a slime",
            background="transparent",
            quality="medium",
            size="1024x1024",
            created_at=datetime.now(timezone.utc),
            tags=["test"],
        )
        img_path = tmp_path / "test.png"
        img_path.touch()
        meta_path = save_metadata(meta, img_path)
        assert meta_path.exists()
        assert meta_path.suffix == ".json"
        import json
        data = json.loads(meta_path.read_text())
        assert data["name"] == "test"
        assert data["tool"] == "generate_game_asset"


class TestValidateInputImage:
    def test_valid_png(self, tmp_path: Path):
        path = tmp_path / "test.png"
        save_test_png(path, 32, 32)
        w, h = validate_input_image(path)
        assert w == 32
        assert h == 32

    def test_nonexistent_file(self, tmp_path: Path):
        with pytest.raises(AssetError) as exc_info:
            validate_input_image(tmp_path / "nope.png")
        assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND

    def test_oversized_image(self, tmp_path: Path):
        path = tmp_path / "big.png"
        save_test_png(path, 5000, 5000)
        with pytest.raises(AssetError) as exc_info:
            validate_input_image(path)
        assert exc_info.value.code == ErrorCode.INVALID_IMAGE

    def test_not_an_image(self, tmp_path: Path):
        path = tmp_path / "fake.png"
        path.write_text("not an image")
        with pytest.raises(AssetError) as exc_info:
            validate_input_image(path)
        assert exc_info.value.code == ErrorCode.INVALID_IMAGE


class TestValidateMask:
    def test_valid_mask(self, tmp_path: Path):
        path = tmp_path / "mask.png"
        save_test_png(path, 32, 32, mode="RGBA")
        validate_mask(path, 32, 32)  # should not raise

    def test_dimension_mismatch(self, tmp_path: Path):
        path = tmp_path / "mask.png"
        save_test_png(path, 64, 64, mode="RGBA")
        with pytest.raises(AssetError) as exc_info:
            validate_mask(path, 32, 32)
        assert exc_info.value.code == ErrorCode.MASK_MISMATCH

    def test_no_alpha_channel(self, tmp_path: Path):
        path = tmp_path / "mask.png"
        save_test_png(path, 32, 32, mode="RGB")
        with pytest.raises(AssetError) as exc_info:
            validate_mask(path, 32, 32)
        assert exc_info.value.code == ErrorCode.INVALID_IMAGE

    def test_mask_not_found(self, tmp_path: Path):
        with pytest.raises(AssetError) as exc_info:
            validate_mask(tmp_path / "nope.png", 32, 32)
        assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND


class TestValidateImageBytes:
    def test_valid_png(self):
        data = make_png_bytes(32, 32)
        w, h = validate_image_bytes(data)
        assert w == 32
        assert h == 32

    def test_invalid_data(self):
        with pytest.raises(AssetError) as exc_info:
            validate_image_bytes(b"not an image")
        assert exc_info.value.code == ErrorCode.INVALID_IMAGE

    def test_oversized(self):
        data = make_png_bytes(5000, 5000)
        with pytest.raises(AssetError) as exc_info:
            validate_image_bytes(data)
        assert exc_info.value.code == ErrorCode.INVALID_IMAGE


class TestValidateMaskBytes:
    def test_valid_mask(self):
        data = make_png_bytes(32, 32, mode="RGBA")
        validate_mask_bytes(data, 32, 32)  # should not raise

    def test_dimension_mismatch(self):
        data = make_png_bytes(64, 64, mode="RGBA")
        with pytest.raises(AssetError) as exc_info:
            validate_mask_bytes(data, 32, 32)
        assert exc_info.value.code == ErrorCode.MASK_MISMATCH

    def test_no_alpha(self):
        data = make_png_bytes(32, 32, mode="RGB")
        with pytest.raises(AssetError) as exc_info:
            validate_mask_bytes(data, 32, 32)
        assert exc_info.value.code == ErrorCode.INVALID_IMAGE
