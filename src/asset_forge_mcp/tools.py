"""MCP tool handler implementations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from mcp.types import TextContent

from .config import get_settings
from .files import (
    resolve_s3_key,
    upload_asset,
    upload_metadata,
    validate_image_bytes,
    validate_mask_bytes,
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
from .s3_client import S3Storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (set by server lifespan)
# ---------------------------------------------------------------------------

_client: OpenAIImageClient | None = None
_storage: S3Storage | None = None


def set_client(client: OpenAIImageClient) -> None:
    global _client
    _client = client


def get_client() -> OpenAIImageClient:
    if _client is None:
        raise AssetError(ErrorCode.MISSING_API_KEY, "OpenAI client not initialised. Is OPENAI_API_KEY set?")
    return _client


def set_storage(storage: S3Storage | None) -> None:
    global _storage
    _storage = storage


def get_storage() -> S3Storage:
    if _storage is None:
        raise AssetError(ErrorCode.S3_ERROR, "S3 storage not initialised.")
    return _storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
) -> list[TextContent]:
    """Generate a new game image asset."""
    if n < 1 or n > 8:
        raise AssetError(ErrorCode.INVALID_INPUT, "n must be between 1 and 8.")

    settings = get_settings()
    client = get_client()
    storage = get_storage()
    tags = tags or []

    final_prompt = build_generation_prompt(prompt, asset_type, style, background)
    folder = ASSET_TYPE_FOLDERS[asset_type]

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

    stored_keys: list[str] = []

    for idx, b64 in enumerate(b64_images):
        img_name = name if n == 1 else f"{name}_{idx + 1}"
        key = await resolve_s3_key(storage, folder, img_name)
        await upload_asset(storage, b64, key)
        stored_keys.append(key)

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
        await upload_metadata(storage, meta, key)

    text_payload = json.dumps({
        "ok": True,
        "tool": "generate_game_asset",
        "name": name,
        "asset_type": asset_type.value,
        "bucket": storage.bucket,
        "keys": stored_keys,
    })

    return [TextContent(type="text", text=text_payload)]


# ---------------------------------------------------------------------------
# Tool 2: edit_game_asset
# ---------------------------------------------------------------------------

async def edit_game_asset(
    input_image: str,
    prompt: str,
    output_name: str | None = None,
    mask_image: str | None = None,
    background: BackgroundType = BackgroundType.AUTO,
    quality: ImageQuality = ImageQuality.AUTO,
    size: ImageSize = ImageSize.S_1024x1024,
) -> list[TextContent]:
    """Edit an existing image asset."""
    settings = get_settings()
    client = get_client()
    storage = get_storage()

    image_bytes = base64.b64decode(input_image)
    src_w, src_h = validate_image_bytes(image_bytes)

    mask_bytes: bytes | None = None
    if mask_image is not None:
        mask_bytes = base64.b64decode(mask_image)
        validate_mask_bytes(mask_bytes, src_w, src_h)

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
    logger.info("edit_game_asset: completed in %.1fs", elapsed)

    # Upload to S3
    folder = ASSET_TYPE_FOLDERS[AssetType.SPRITE]
    out_name = output_name or "edited_asset"
    key = await resolve_s3_key(storage, folder, out_name)
    await upload_asset(storage, b64_result, key)

    meta = AssetMetadata(
        name=out_name,
        tool="edit_game_asset",
        model=settings.openai_image_model,
        asset_type="",
        prompt=prompt,
        final_prompt=final_prompt,
        background=background.value,
        quality=quality.value,
        size=size.value,
        created_at=datetime.now(timezone.utc),
    )
    await upload_metadata(storage, meta, key)

    text_payload = json.dumps({
        "ok": True,
        "tool": "edit_game_asset",
        "name": out_name,
        "bucket": storage.bucket,
        "key": key,
    })

    return [TextContent(type="text", text=text_payload)]


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
) -> list[TextContent]:
    """Generate multiple variants of a game asset."""
    if variant_count < 1 or variant_count > 8:
        raise AssetError(ErrorCode.INVALID_INPUT, "variant_count must be between 1 and 8.")

    settings = get_settings()
    client = get_client()
    storage = get_storage()
    tags = tags or []

    final_prompt = build_generation_prompt(prompt, asset_type, style, background)
    folder = ASSET_TYPE_FOLDERS[asset_type]

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

    stored_keys: list[str] = []
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
        key = await resolve_s3_key(storage, folder, img_name)
        await upload_asset(storage, b64, key)
        stored_keys.append(key)

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
        await upload_metadata(storage, meta, key)
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
        "bucket": storage.bucket,
        "keys": stored_keys,
    }
    if warnings:
        text_data["warnings"] = warnings

    return [TextContent(type="text", text=json.dumps(text_data))]
