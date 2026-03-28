"""Prompt construction for generation and editing tools."""

from __future__ import annotations

from .models import AssetType, BackgroundType, StyleHint


# Asset types that get the "clean silhouette" suffix
_COMPACT_TYPES = {AssetType.ICON, AssetType.SPRITE, AssetType.TILE, AssetType.UI}


def build_generation_prompt(
    user_prompt: str,
    asset_type: AssetType,
    style: StyleHint,
    background: BackgroundType,
) -> str:
    """Build the final prompt for image generation (spec section 7)."""
    parts: list[str] = []

    parts.append(f"Create a game-ready {style.value} {asset_type.value}.")
    parts.append(user_prompt.strip())
    parts.append("No text, no watermark, no mockup.")

    if background == BackgroundType.TRANSPARENT:
        parts.append("Transparent background.")

    if asset_type in _COMPACT_TYPES:
        parts.append("Clean silhouette, readable composition, production-friendly.")

    return " ".join(parts)


def build_edit_prompt(user_prompt: str) -> str:
    """Build the final prompt for image editing (spec section 7)."""
    parts: list[str] = [
        "Edit the provided game asset.",
        "Preserve overall readability.",
        "Do not add text, watermark, frame, or mockup.",
        user_prompt.strip(),
    ]
    return " ".join(parts)
