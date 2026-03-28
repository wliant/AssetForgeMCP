"""Domain models, enums, and metadata schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums (values match spec exactly)
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    SPRITE = "sprite"
    ICON = "icon"
    PORTRAIT = "portrait"
    BACKGROUND = "background"
    TILE = "tile"
    UI = "ui"


# Map asset type -> plural folder name
ASSET_TYPE_FOLDERS: dict[AssetType, str] = {
    AssetType.SPRITE: "sprites",
    AssetType.ICON: "icons",
    AssetType.PORTRAIT: "portraits",
    AssetType.BACKGROUND: "backgrounds",
    AssetType.TILE: "tiles",
    AssetType.UI: "ui",
}


class StyleHint(str, Enum):
    PIXEL_ART = "pixel-art"
    PAINTERLY = "painterly"
    VECTOR = "vector"
    SEMI_REALISTIC = "semi-realistic"


class ImageSize(str, Enum):
    S_1024x1024 = "1024x1024"
    S_1536x1024 = "1536x1024"
    S_1024x1536 = "1024x1536"
    AUTO = "auto"


class ImageQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class BackgroundType(str, Enum):
    TRANSPARENT = "transparent"
    OPAQUE = "opaque"
    AUTO = "auto"


class ErrorCode(str, Enum):
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_INPUT = "INVALID_INPUT"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_IMAGE = "INVALID_IMAGE"
    MASK_MISMATCH = "MASK_MISMATCH"
    OUTPUT_DIR_ERROR = "OUTPUT_DIR_ERROR"
    OPENAI_AUTH_ERROR = "OPENAI_AUTH_ERROR"
    OPENAI_RATE_LIMIT = "OPENAI_RATE_LIMIT"
    OPENAI_SERVER_ERROR = "OPENAI_SERVER_ERROR"
    OPENAI_TIMEOUT = "OPENAI_TIMEOUT"
    OPENAI_BAD_RESPONSE = "OPENAI_BAD_RESPONSE"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    S3_ERROR = "S3_ERROR"


# ---------------------------------------------------------------------------
# Sidecar metadata (written alongside every generated image)
# ---------------------------------------------------------------------------

class AssetMetadata(BaseModel):
    name: str
    tool: str
    model: str
    asset_type: str
    style: str | None = None
    prompt: str
    final_prompt: str
    background: str
    quality: str
    size: str
    source_image: str | None = None
    mask_image: str | None = None
    created_at: datetime
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Structured error detail
# ---------------------------------------------------------------------------

class AssetError(Exception):
    """Raised by tool handlers to surface structured errors."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "error": {"code": self.code.value, "message": self.message},
        }
