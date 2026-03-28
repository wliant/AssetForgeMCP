"""OpenAI Image API client with retry logic."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .models import AssetError, ErrorCode

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_429_RETRIES = 3
_429_BACKOFF_SECONDS = [1, 2, 4]
_5XX_RETRY_DELAY = 2
_DEFAULT_TIMEOUT = 60.0


class OpenAIImageClient:
    """Async HTTP client for the OpenAI Images API."""

    def __init__(self, api_key: str, base_url: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIImageClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        background: str,
        n: int = 1,
    ) -> list[str]:
        """Generate image(s) and return list of base64-encoded strings."""
        body: dict = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "background": background,
            "response_format": "b64_json",
        }
        # Remove 'auto' values — let the API use its defaults
        body = {k: v for k, v in body.items() if v != "auto"}

        data = await self._request_json("POST", "/images/generations", json=body)
        return self._extract_b64_list(data)

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        background: str,
        mask_bytes: bytes | None = None,
    ) -> str:
        """Edit an image and return base64-encoded result."""
        files: dict = {
            "image": ("image.png", image_bytes, "image/png"),
        }
        if mask_bytes is not None:
            files["mask"] = ("mask.png", mask_bytes, "image/png")

        form_data: dict = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "response_format": "b64_json",
        }
        # Remove 'auto' values
        form_data = {k: v for k, v in form_data.items() if v != "auto"}

        data = await self._request_json("POST", "/images/edits", data=form_data, files=files)
        results = self._extract_b64_list(data)
        if not results:
            raise AssetError(ErrorCode.OPENAI_BAD_RESPONSE, "No image data in edit response.")
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_json(self, method: str, path: str, **kwargs: object) -> dict:
        """Send an HTTP request with retry logic. Returns parsed JSON."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        # Determine max attempts: first try + retries
        max_attempts = 1 + _MAX_429_RETRIES  # covers 429 retries; 5xx gets 1 extra

        for attempt in range(max_attempts):
            start = time.monotonic()
            try:
                resp = await self._client.request(method, url, **kwargs)  # type: ignore[arg-type]
                elapsed = time.monotonic() - start
                logger.debug("OpenAI %s %s -> %s (%.2fs)", method, path, resp.status_code, elapsed)

                if resp.is_success:
                    return resp.json()

                # --- error handling ---
                status = resp.status_code
                body_text = resp.text[:500]

                if status == 429:
                    if attempt < _MAX_429_RETRIES:
                        delay = _429_BACKOFF_SECONDS[min(attempt, len(_429_BACKOFF_SECONDS) - 1)]
                        logger.warning("Rate limited (429). Retry %d/%d in %ds.", attempt + 1, _MAX_429_RETRIES, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise AssetError(ErrorCode.OPENAI_RATE_LIMIT, f"Rate limited after {_MAX_429_RETRIES} retries.")

                if 500 <= status < 600:
                    if attempt == 0:
                        logger.warning("Server error (%d). Retrying once in %ds.", status, _5XX_RETRY_DELAY)
                        await asyncio.sleep(_5XX_RETRY_DELAY)
                        continue
                    raise AssetError(ErrorCode.OPENAI_SERVER_ERROR, f"OpenAI server error {status}: {body_text}")

                if status in (401, 403):
                    raise AssetError(ErrorCode.OPENAI_AUTH_ERROR, f"OpenAI request failed with {status}. Check OPENAI_API_KEY.")

                # Other 4xx
                raise AssetError(ErrorCode.OPENAI_BAD_RESPONSE, f"OpenAI request failed with {status}: {body_text}")

            except httpx.TimeoutException as exc:
                raise AssetError(ErrorCode.OPENAI_TIMEOUT, f"Request timed out after {_DEFAULT_TIMEOUT}s.") from exc
            except AssetError:
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                raise AssetError(ErrorCode.OPENAI_SERVER_ERROR, f"HTTP error: {exc}") from exc

        # Should not reach here, but just in case
        raise AssetError(ErrorCode.OPENAI_SERVER_ERROR, "Request failed after all retries.")

    @staticmethod
    def _extract_b64_list(data: dict) -> list[str]:
        """Pull base64 strings from the OpenAI response."""
        try:
            return [item["b64_json"] for item in data["data"]]
        except (KeyError, TypeError, IndexError) as exc:
            raise AssetError(ErrorCode.OPENAI_BAD_RESPONSE, f"Unexpected response format: {exc}") from exc
