"""MCP tool handler implementations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.types import ImageContent, TextContent

from .config import get_settings
from .files import (
    get_output_dir,
    read_image_bytes,
    resolve_filepath,
    save_asset,
    save_metadata,
    validate_input_image,
    validate_mask,
)
from .models import (
    ASSET_TYPE_FOLDERS,
    AssetError,
    AssetMetadata,
    AssetType,
    BackgroundType,
    ErrorCode,
    ImageQuality,
    ImageSize,
    StyleHint,
)
from .openai_client import OpenAIImageClient
from .prompts import build_edit_prompt, build_generation_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level client singleton (set by server lifespan)
# ---------------------------------------------------------------------------

_client: OpenAIImageClient | None = None


def set_client(client: OpenAIImageClient) -> None:
    global _client
    _client = client


def get_client() -> OpenAIImageClient:
    if _client is None:
        raise AssetError(ErrorCode.MISSING_API_KEY, "OpenAI client not initialised. Is OPENAI_API_KEY set?")
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload_size_bytes(content: list) -> int:
    """Estimate response payload size for logging."""
    total = 0
    for block in content:
        if isinstance(block, ImageContent):
            total += len(block.data) if block.data else 0
        elif isinstance(block, TextContent):
            total += len(block.text) if block.text else 0
    return total


_10MB = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Tool 1: generate_game_asset
# ---------------------------------------------------------------------------

async def generate_game_asset(
    name: str,
    prompt: str,
    asset_type: AssetType = AssetType.SPRITE,
    style: StyleHint = StyleHint.PIXEL_ART,
    size: ImageSize = ImageSize.S_1024x1024,
    background: BackgroundType = BackgroundType.TRANSPARENT,
    quality: ImageQuality = ImageQuality.AUTO,
    n: int = 1,
    tags: list[str] | None = None,
) -> list[TextContent | ImageContent]:
    """Generate a new game image asset."""
    if n < 1 or n > 8:
        raise AssetError(ErrorCode.INVALID_INPUT, "n must be between 1 and 8.")

    settings = get_settings()
    client = get_client()
    tags = tags or []

    final_prompt = build_generation_prompt(prompt, asset_type, style, background)
    out_dir = get_output_dir(settings.asset_output_dir, asset_type)

    start = time.monotonic()
    b64_images = await client.generate_image(
        prompt=final_prompt,
        model=settings.openai_image_model,
        size=size.value,
        quality=quality.value,
        background=background.value,
        n=n,
    )
    elapsed = time.monotonic() - start
    logger.info("generate_game_asset '%s': %d image(s) in %.1fs", name, len(b64_images), elapsed)

    files_info: list[dict[str, str]] = []
    image_blocks: list[ImageContent] = []

    for idx, b64 in enumerate(b64_images):
        img_name = name if n == 1 else f"{name}_{idx + 1}"
        filepath = resolve_filepath(out_dir, img_name)
        save_asset(b64, filepath)

        meta = AssetMetadata(
            name=img_name,
            tool="generate_game_asset",
            model=settings.openai_image_model,
            asset_type=asset_type.value,
            style=style.value,
            prompt=prompt,
            final_prompt=final_prompt,
            background=background.value,
            quality=quality.value,
            size=size.value,
            created_at=datetime.now(timezone.utc),
            tags=tags,
        )
        meta_path = save_metadata(meta, filepath)

        files_info.append({
            "image_path": str(filepath),
            "metadata_path": str(meta_path),
        })
        image_blocks.append(ImageContent(type="image", data=b64, mimeType="image/png"))

    text_payload = json.dumps({
        "ok": True,
        "tool": "generate_game_asset",
        "name": name,
        "asset_type": asset_type.value,
        "files": files_info,
    })

    content: list[TextContent | ImageContent] = [TextContent(type="text", text=text_payload)]
    content.extend(image_blocks)

    payload_bytes = _payload_size_bytes(content)
    if payload_bytes > _10MB:
        logger.warning("Response payload ~%d bytes exceeds 10 MB.", payload_bytes)

    return content


# ---------------------------------------------------------------------------
# Tool 2: edit_game_asset
# ---------------------------------------------------------------------------

async def edit_game_asset(
    input_path: str,
    prompt: str,
    output_name: str | None = None,
    mask_path: str | None = None,
    background: BackgroundType = BackgroundType.AUTO,
    quality: ImageQuality = ImageQuality.AUTO,
    size: ImageSize = ImageSize.S_1024x1024,
) -> list[TextContent | ImageContent]:
    """Edit an existing image asset."""
    settings = get_settings()
    client = get_client()

    src = Path(input_path)
    src_w, src_h = validate_input_image(src)

    mask_bytes: bytes | None = None
    if mask_path is not None:
        mp = Path(mask_path)
        validate_mask(mp, src_w, src_h)
        mask_bytes = read_image_bytes(mp)

    image_bytes = read_image_bytes(src)
    final_prompt = build_edit_prompt(prompt)

    start = time.monotonic()
    b64_result = await client.edit_image(
        image_bytes=image_bytes,
        prompt=final_prompt,
        model=settings.openai_image_model,
        size=size.value,
        quality=quality.value,
        background=background.value,
        mask_bytes=mask_bytes,
    )
    elapsed = time.monotonic() - start
    logger.info("edit_game_asset '%s': completed in %.1fs", input_path, elapsed)

    # Output goes in the same directory as the input file
    out_dir = src.parent
    out_name = output_name or f"{src.stem}_edited"
    filepath = resolve_filepath(out_dir, out_name)
    save_asset(b64_result, filepath)

    meta = AssetMetadata(
        name=out_name,
        tool="edit_game_asset",
        model=settings.openai_image_model,
        asset_type="",  # derived from input context
        prompt=prompt,
        final_prompt=final_prompt,
        background=background.value,
        quality=quality.value,
        size=size.value,
        source_image=str(src),
        mask_image=mask_path,
        created_at=datetime.now(timezone.utc),
    )
    meta_path = save_metadata(meta, filepath)

    text_payload = json.dumps({
        "ok": True,
        "tool": "edit_game_asset",
        "input_path": str(src),
        "files": [{
            "image_path": str(filepath),
            "metadata_path": str(meta_path),
        }],
    })

    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=text_payload),
        ImageContent(type="image", data=b64_result, mimeType="image/png"),
    ]

    payload_bytes = _payload_size_bytes(content)
    if payload_bytes > _10MB:
        logger.warning("Response payload ~%d bytes exceeds 10 MB.", payload_bytes)

    return content


# ---------------------------------------------------------------------------
# Tool 3: generate_asset_variants
# ---------------------------------------------------------------------------

async def generate_asset_variants(
    name: str,
    prompt: str,
    asset_type: AssetType = AssetType.SPRITE,
    style: StyleHint = StyleHint.PIXEL_ART,
    size: ImageSize = ImageSize.S_1024x1024,
    background: BackgroundType = BackgroundType.TRANSPARENT,
    quality: ImageQuality = ImageQuality.AUTO,
    variant_count: int = 4,
    tags: list[str] | None = None,
) -> list[TextContent | ImageContent]:
    """Generate multiple variants of a game asset."""
    if variant_count < 1 or variant_count > 8:
        raise AssetError(ErrorCode.INVALID_INPUT, "variant_count must be between 1 and 8.")

    settings = get_settings()
    client = get_client()
    tags = tags or []

    final_prompt = build_generation_prompt(prompt, asset_type, style, background)
    out_dir = get_output_dir(settings.asset_output_dir, asset_type)

    # Try batch first; fall back to sequential calls if the API limits n
    start = time.monotonic()
    try:
        b64_images = await client.generate_image(
            prompt=final_prompt,
            model=settings.openai_image_model,
            size=size.value,
            quality=quality.value,
            background=background.value,
            n=variant_count,
        )
        results: list[tuple[int, str | None, str | None]] = [
            (i, img, None) for i, img in enumerate(b64_images)
        ]
    except AssetError:
        # Fall back to sequential calls
        logger.info("Batch generation unavailable, falling back to sequential calls.")
        tasks = []
        for _ in range(variant_count):
            tasks.append(
                client.generate_image(
                    prompt=final_prompt,
                    model=settings.openai_image_model,
                    size=size.value,
                    quality=quality.value,
                    background=background.value,
                    n=1,
                )
            )
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                results.append((i, None, str(res)))
            else:
                results.append((i, res[0], None))

    elapsed = time.monotonic() - start

    files_info: list[dict[str, str]] = []
    image_blocks: list[ImageContent] = []
    warnings: list[str] = []
    completed = 0

    for idx, b64, error in results:
        variant_num = idx + 1
        if error is not None:
            warnings.append(f"Variant {variant_num} failed: {error}")
            logger.warning("Variant %d failed: %s", variant_num, error)
            continue

        assert b64 is not None
        img_name = f"{name}_{variant_num}"
        filepath = resolve_filepath(out_dir, img_name)
        save_asset(b64, filepath)

        meta = AssetMetadata(
            name=img_name,
            tool="generate_asset_variants",
            model=settings.openai_image_model,
            asset_type=asset_type.value,
            style=style.value,
            prompt=prompt,
            final_prompt=final_prompt,
            background=background.value,
            quality=quality.value,
            size=size.value,
            created_at=datetime.now(timezone.utc),
            tags=tags,
        )
        meta_path = save_metadata(meta, filepath)

        files_info.append({
            "image_path": str(filepath),
            "metadata_path": str(meta_path),
        })
        image_blocks.append(ImageContent(type="image", data=b64, mimeType="image/png"))
        completed += 1

    logger.info(
        "generate_asset_variants '%s': %d/%d variants in %.1fs",
        name, completed, variant_count, elapsed,
    )

    if completed == 0:
        raise AssetError(
            ErrorCode.OPENAI_SERVER_ERROR,
            f"All {variant_count} variants failed. Errors: {'; '.join(warnings)}",
        )

    text_data: dict[str, Any] = {
        "ok": True,
        "tool": "generate_asset_variants",
        "name": name,
        "asset_type": asset_type.value,
        "requested_variants": variant_count,
        "completed_variants": completed,
        "files": files_info,
    }
    if warnings:
        text_data["warnings"] = warnings

    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(text_data))]
    content.extend(image_blocks)

    payload_bytes = _payload_size_bytes(content)
    if payload_bytes > _10MB:
        logger.warning("Response payload ~%d bytes exceeds 10 MB.", payload_bytes)

    return content
