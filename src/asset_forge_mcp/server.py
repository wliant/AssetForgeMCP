"""Asset Forge MCP server — entry point and tool registration."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .logging_config import setup_logging
from .models import (
    AssetError,
    AssetType,
    BackgroundType,
    ImageQuality,
    ImageSize,
    StyleHint,
)
from .openai_client import OpenAIImageClient
from .s3_client import S3Storage
from .tools import (
    edit_game_asset as _edit_game_asset,
    generate_asset_variants as _generate_asset_variants,
    generate_game_asset as _generate_game_asset,
    set_client,
    set_storage,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Initialise resources on startup, clean up on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Asset Forge MCP starting")
    logger.info("  Host: %s:%d", settings.mcp_host, settings.mcp_port)
    logger.info("  Image model: %s", settings.openai_image_model)
    logger.info("  Base URL: %s", settings.openai_base_url)
    logger.info("  S3 endpoint: %s", settings.s3_endpoint_url)
    logger.info("  S3 bucket: %s", settings.s3_bucket)
    logger.info("  Docker: %s", "yes" if os.path.exists("/.dockerenv") else "no")

    client = OpenAIImageClient(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
    )
    set_client(client)

    storage = S3Storage(
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key.get_secret_value(),
        region=settings.s3_region,
        bucket=settings.s3_bucket,
    )
    await storage.ensure_bucket()
    set_storage(storage)

    try:
        yield
    finally:
        await storage.close()
        await client.close()
        logger.info("Asset Forge MCP stopped")


_startup_settings = get_settings()

mcp = FastMCP(
    "Asset Forge MCP",
    instructions=(
        "Game asset generation and editing server. "
        "Use generate_game_asset for new assets, "
        "edit_game_asset to modify existing images, "
        "and generate_asset_variants for multiple variations."
    ),
    lifespan=lifespan,
    host=_startup_settings.mcp_host,
    port=_startup_settings.mcp_port,
)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


@mcp.tool()
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
) -> list:
    """Generate a new game image asset.

    Returns a text block with JSON metadata including the S3 bucket and keys.

    Args:
        name: Asset name (used as filename).
        prompt: Description of the game asset to generate.
        asset_type: Type of game asset (sprite, icon, portrait, background, tile, ui).
        style: Visual style (pixel-art, painterly, vector, semi-realistic).
        size: Image dimensions (1024x1024, 1536x1024, 1024x1536, auto).
        background: Background type (transparent, opaque, auto).
        quality: Image quality (low, medium, high, auto).
        n: Number of images to generate (1-8).
        tags: Optional tags for metadata.
    """
    try:
        return await _generate_game_asset(
            name=name, prompt=prompt, asset_type=asset_type, style=style,
            size=size, background=background, quality=quality, n=n, tags=tags,
        )
    except AssetError as exc:
        from mcp.types import TextContent
        return [TextContent(type="text", text=str(exc.to_dict()))]


@mcp.tool()
async def edit_game_asset(
    input_image: str,
    prompt: str,
    output_name: str | None = None,
    mask_image: str | None = None,
    background: BackgroundType = BackgroundType.AUTO,
    quality: ImageQuality = ImageQuality.AUTO,
    size: ImageSize = ImageSize.S_1024x1024,
) -> list:
    """Edit an existing image asset.

    Returns a text block with JSON metadata including the S3 bucket and key.

    Args:
        input_image: Base64-encoded source image (PNG or JPEG).
        prompt: Description of the edits to make.
        output_name: Output filename for internal storage.
        mask_image: Optional base64-encoded mask image (PNG with alpha channel).
        background: Background type (transparent, opaque, auto).
        quality: Image quality (low, medium, high, auto).
        size: Output dimensions (1024x1024, 1536x1024, 1024x1536, auto).
    """
    try:
        return await _edit_game_asset(
            input_image=input_image, prompt=prompt, output_name=output_name,
            mask_image=mask_image, background=background, quality=quality, size=size,
        )
    except AssetError as exc:
        from mcp.types import TextContent
        return [TextContent(type="text", text=str(exc.to_dict()))]


@mcp.tool()
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
) -> list:
    """Generate multiple variants of a game asset concept.

    Returns a text block with JSON metadata including completed/requested
    variant counts, S3 bucket and keys, and any warnings.

    Args:
        name: Base asset name (variants get _1, _2, ... suffixes).
        prompt: Core concept description for variant generation.
        asset_type: Type of game asset (sprite, icon, portrait, background, tile, ui).
        style: Visual style (pixel-art, painterly, vector, semi-realistic).
        size: Image dimensions (1024x1024, 1536x1024, 1024x1536, auto).
        background: Background type (transparent, opaque, auto).
        quality: Image quality (low, medium, high, auto).
        variant_count: Number of variants to generate (1-8).
        tags: Optional tags for metadata.
    """
    try:
        return await _generate_asset_variants(
            name=name, prompt=prompt, asset_type=asset_type, style=style,
            size=size, background=background, quality=quality,
            variant_count=variant_count, tags=tags,
        )
    except AssetError as exc:
        from mcp.types import TextContent
        return [TextContent(type="text", text=str(exc.to_dict()))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
