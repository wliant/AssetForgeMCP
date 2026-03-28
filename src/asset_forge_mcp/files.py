"""File I/O: sanitization, S3 upload, validate, image checks."""

from __future__ import annotations

import base64
import logging
import re
from io import BytesIO

from PIL import Image

from .models import ASSET_TYPE_FOLDERS, AssetError, AssetMetadata, AssetType, ErrorCode
from .s3_client import S3Storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r"[^a-z0-9_]")
_MULTI_UNDERSCORE_RE = re.compile(r"_{2,}")
_MAX_NAME_LEN = 64


def sanitize_filename(name: str) -> str:
    """Lowercase, underscore-safe filename. Raises on empty result."""
    cleaned = name.lower().strip()
    cleaned = cleaned.replace(" ", "_").replace("-", "_")
    cleaned = _UNSAFE_RE.sub("", cleaned)
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned)
    cleaned = cleaned.strip("_")
    cleaned = cleaned[:_MAX_NAME_LEN]
    if not cleaned:
        raise AssetError(ErrorCode.INVALID_INPUT, f"Name '{name}' produces empty filename after sanitization.")
    return cleaned


# ---------------------------------------------------------------------------
# S3 key helpers
# ---------------------------------------------------------------------------

def build_s3_key(folder: str, name: str, ext: str = ".png") -> str:
    """Build an S3 key like 'sprites/forest_slime.png'."""
    safe = sanitize_filename(name)
    return f"{folder}/{safe}{ext}"


async def resolve_s3_key(
    storage: S3Storage, folder: str, name: str, ext: str = ".png"
) -> str:
    """Return a non-colliding S3 key. Checks head_object for existence."""
    safe = sanitize_filename(name)
    candidate = f"{folder}/{safe}{ext}"
    if not await storage.key_exists(candidate):
        return candidate

    for i in range(2, 100):
        candidate = f"{folder}/{safe}_v{i}{ext}"
        if not await storage.key_exists(candidate):
            logger.warning("Key collision: renamed to %s", candidate)
            return candidate

    raise AssetError(ErrorCode.OUTPUT_DIR_ERROR, f"Too many collisions for name '{safe}' in {folder}/.")


async def upload_asset(storage: S3Storage, b64_data: str, key: str) -> None:
    """Decode base64 PNG data and upload to S3."""
    raw = base64.b64decode(b64_data)
    await storage.upload_bytes(raw, key, content_type="image/png")
    logger.info("Uploaded image: s3://%s/%s (%d bytes)", storage.bucket, key, len(raw))


async def upload_metadata(storage: S3Storage, metadata: AssetMetadata, image_key: str) -> str:
    """Upload sidecar JSON to S3 next to the image. Returns metadata key."""
    meta_key = image_key.rsplit(".", 1)[0] + ".json"
    await storage.upload_json(metadata.model_dump(mode="json"), meta_key)
    logger.info("Uploaded metadata: s3://%s/%s", storage.bucket, meta_key)
    return meta_key


# ---------------------------------------------------------------------------
# Image validation (for edits)
# ---------------------------------------------------------------------------

_MAX_DIMENSION = 4096


def validate_input_image(path) -> tuple[int, int]:
    """Verify *path* is a valid PNG/JPEG and return (width, height).

    Raises AssetError on invalid input.
    """
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        raise AssetError(ErrorCode.FILE_NOT_FOUND, f"Input file not found: {path}")
    if not path.is_file():
        raise AssetError(ErrorCode.FILE_NOT_FOUND, f"Input path is not a file: {path}")

    try:
        with Image.open(path) as img:
            fmt = img.format
            if fmt not in ("PNG", "JPEG"):
                raise AssetError(
                    ErrorCode.INVALID_IMAGE,
                    f"Unsupported image format '{fmt}'. Must be PNG or JPEG.",
                )
            w, h = img.size
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(ErrorCode.INVALID_IMAGE, f"Cannot open image: {exc}") from exc

    if w > _MAX_DIMENSION or h > _MAX_DIMENSION:
        raise AssetError(
            ErrorCode.INVALID_IMAGE,
            f"Image dimensions {w}x{h} exceed maximum {_MAX_DIMENSION}x{_MAX_DIMENSION}.",
        )
    return w, h


def validate_mask(mask_path, source_width: int, source_height: int) -> None:
    """Verify mask is a PNG with alpha channel matching source dimensions."""
    from pathlib import Path
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise AssetError(ErrorCode.FILE_NOT_FOUND, f"Mask file not found: {mask_path}")

    try:
        with Image.open(mask_path) as img:
            if img.format != "PNG":
                raise AssetError(ErrorCode.INVALID_IMAGE, "Mask must be a PNG file.")
            if img.mode not in ("RGBA", "LA"):
                raise AssetError(ErrorCode.INVALID_IMAGE, "Mask PNG must have an alpha channel (RGBA).")
            mw, mh = img.size
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(ErrorCode.INVALID_IMAGE, f"Cannot open mask: {exc}") from exc

    if mw != source_width or mh != source_height:
        raise AssetError(
            ErrorCode.MASK_MISMATCH,
            f"Mask dimensions {mw}x{mh} do not match source image {source_width}x{source_height}.",
        )


def read_image_bytes(path) -> bytes:
    """Read an image file as raw bytes."""
    from pathlib import Path
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# In-memory image validation (for base64 inputs)
# ---------------------------------------------------------------------------

def validate_image_bytes(data: bytes) -> tuple[int, int]:
    """Validate raw image bytes are a valid PNG/JPEG. Returns (width, height)."""
    try:
        with Image.open(BytesIO(data)) as img:
            fmt = img.format
            if fmt not in ("PNG", "JPEG"):
                raise AssetError(
                    ErrorCode.INVALID_IMAGE,
                    f"Unsupported image format '{fmt}'. Must be PNG or JPEG.",
                )
            w, h = img.size
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(ErrorCode.INVALID_IMAGE, f"Cannot open image: {exc}") from exc

    if w > _MAX_DIMENSION or h > _MAX_DIMENSION:
        raise AssetError(
            ErrorCode.INVALID_IMAGE,
            f"Image dimensions {w}x{h} exceed maximum {_MAX_DIMENSION}x{_MAX_DIMENSION}.",
        )
    return w, h


def validate_mask_bytes(data: bytes, source_width: int, source_height: int) -> None:
    """Validate raw mask bytes: must be PNG with alpha, matching source dimensions."""
    try:
        with Image.open(BytesIO(data)) as img:
            if img.format != "PNG":
                raise AssetError(ErrorCode.INVALID_IMAGE, "Mask must be a PNG file.")
            if img.mode not in ("RGBA", "LA"):
                raise AssetError(ErrorCode.INVALID_IMAGE, "Mask PNG must have an alpha channel (RGBA).")
            mw, mh = img.size
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(ErrorCode.INVALID_IMAGE, f"Cannot open mask: {exc}") from exc

    if mw != source_width or mh != source_height:
        raise AssetError(
            ErrorCode.MASK_MISMATCH,
            f"Mask dimensions {mw}x{mh} do not match source image {source_width}x{source_height}.",
        )
