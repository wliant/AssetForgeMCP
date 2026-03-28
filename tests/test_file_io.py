"""Tests for file I/O: S3 upload helpers, validate, mask checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from asset_forge_mcp.files import (
    resolve_s3_key, upload_asset, upload_metadata,
    validate_input_image, validate_mask,
    validate_image_bytes, validate_mask_bytes,
)
from asset_forge_mcp.models import AssetError, AssetMetadata, ErrorCode

from .conftest import make_png_b64, make_png_bytes, save_test_png


def _make_mock_storage(existing_keys: set[str] | None = None):
    """Create a mock S3Storage that knows about existing keys."""
    from asset_forge_mcp.s3_client import S3Storage
    mock = AsyncMock(spec=S3Storage)
    mock.bucket = "test-bucket"
    existing = existing_keys or set()
    mock.key_exists = AsyncMock(side_effect=lambda k: k in existing)
    mock.upload_bytes = AsyncMock()
    mock.upload_json = AsyncMock()
    return mock


class TestResolveS3Key:
    @pytest.mark.asyncio
    async def test_no_collision(self):
        storage = _make_mock_storage()
        key = await resolve_s3_key(storage, "sprites", "test_image")
        assert key == "sprites/test_image.png"

    @pytest.mark.asyncio
    async def test_collision_increments(self):
        storage = _make_mock_storage({"sprites/test_image.png"})
        key = await resolve_s3_key(storage, "sprites", "test_image")
        assert key == "sprites/test_image_v2.png"

    @pytest.mark.asyncio
    async def test_multiple_collisions(self):
        storage = _make_mock_storage({
            "sprites/test_image.png",
            "sprites/test_image_v2.png",
        })
        key = await resolve_s3_key(storage, "sprites", "test_image")
        assert key == "sprites/test_image_v3.png"


class TestUploadAsset:
    @pytest.mark.asyncio
    async def test_uploads_decoded_png(self):
        storage = _make_mock_storage()
        b64 = make_png_b64()
        await upload_asset(storage, b64, "sprites/test.png")
        storage.upload_bytes.assert_called_once()
        args, kwargs = storage.upload_bytes.call_args
        # upload_bytes(raw, key, content_type="image/png")
        assert args[1] == "sprites/test.png"
        assert kwargs.get("content_type") == "image/png"
        assert len(args[0]) > 0  # decoded bytes


class TestUploadMetadata:
    @pytest.mark.asyncio
    async def test_uploads_json_sidecar(self):
        from datetime import datetime, timezone
        storage = _make_mock_storage()
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
        meta_key = await upload_metadata(storage, meta, "sprites/test.png")
        assert meta_key == "sprites/test.json"
        storage.upload_json.assert_called_once()


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
