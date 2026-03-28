"""File I/O: sanitization, save, validate, path traversal protection."""

from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from pathlib import Path

from PIL import Image

from .models import ASSET_TYPE_FOLDERS, AssetError, AssetMetadata, AssetType, ErrorCode

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
# Directory helpers
# ---------------------------------------------------------------------------

def get_output_dir(base_dir: Path, asset_type: AssetType) -> Path:
    """Return and create the output directory for an asset type."""
    folder = ASSET_TYPE_FOLDERS[asset_type]
    out = base_dir / folder
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AssetError(ErrorCode.OUTPUT_DIR_ERROR, f"Cannot create output directory {out}: {exc}") from exc
    return out


def _assert_inside(filepath: Path, allowed_root: Path) -> None:
    """Raise if *filepath* escapes *allowed_root* (path traversal guard)."""
    try:
        filepath.resolve().relative_to(allowed_root.resolve())
    except ValueError:
        raise AssetError(
            ErrorCode.INVALID_INPUT,
            f"Path '{filepath}' resolves outside allowed directory '{allowed_root}'.",
        )


# ---------------------------------------------------------------------------
# Overwrite-safe filepath resolution
# ---------------------------------------------------------------------------

def resolve_filepath(directory: Path, name: str, ext: str = ".png") -> Path:
    """Return a non-colliding filepath inside *directory*.

    If ``name.ext`` exists, tries ``name_v2.ext``, ``name_v3.ext``, etc.
    """
    safe = sanitize_filename(name)
    _assert_inside(directory / f"{safe}{ext}", directory)

    candidate = directory / f"{safe}{ext}"
    if not candidate.exists():
        return candidate

    for i in range(2, 100):
        candidate = directory / f"{safe}_v{i}{ext}"
        if not candidate.exists():
            logger.warning("File collision: renamed to %s", candidate.name)
            return candidate

    raise AssetError(ErrorCode.OUTPUT_DIR_ERROR, f"Too many collisions for name '{safe}' in {directory}.")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_asset(b64_data: str, filepath: Path) -> None:
    """Decode base64 PNG data and write to *filepath*."""
    raw = base64.b64decode(b64_data)
    filepath.write_bytes(raw)
    logger.info("Saved image: %s (%d bytes)", filepath, len(raw))


def save_metadata(metadata: AssetMetadata, image_path: Path) -> Path:
    """Write sidecar JSON next to the image file. Returns metadata path."""
    meta_path = image_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", meta_path)
    return meta_path


# ---------------------------------------------------------------------------
# Image validation (for edits)
# ---------------------------------------------------------------------------

_MAX_DIMENSION = 4096


def validate_input_image(path: Path) -> tuple[int, int]:
    """Verify *path* is a valid PNG/JPEG and return (width, height).

    Raises AssetError on invalid input.
    """
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


def validate_mask(mask_path: Path, source_width: int, source_height: int) -> None:
    """Verify mask is a PNG with alpha channel matching source dimensions."""
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


def read_image_bytes(path: Path) -> bytes:
    """Read an image file as raw bytes."""
    return path.read_bytes()
