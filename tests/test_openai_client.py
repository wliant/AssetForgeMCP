"""Tests for OpenAI client with retry logic."""

from __future__ import annotations

import pytest
import httpx
import respx

from asset_forge_mcp.models import AssetError, ErrorCode
from asset_forge_mcp.openai_client import OpenAIImageClient


@pytest.fixture()
def client() -> OpenAIImageClient:
    return OpenAIImageClient(
        api_key="sk-test-key",
        base_url="https://api.openai.com/v1",
        timeout=5.0,
    )


class TestGenerateImage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_success(self, client: OpenAIImageClient):
        respx.post("https://api.openai.com/v1/images/generations").mock(
            return_value=httpx.Response(200, json={
                "data": [{"b64_json": "abc123"}],
            })
        )
        result = await client.generate_image("a slime", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert result == ["abc123"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_multiple_images(self, client: OpenAIImageClient):
        respx.post("https://api.openai.com/v1/images/generations").mock(
            return_value=httpx.Response(200, json={
                "data": [{"b64_json": "img1"}, {"b64_json": "img2"}],
            })
        )
        result = await client.generate_image("a slime", "gpt-image-1", "1024x1024", "medium", "transparent", n=2)
        assert result == ["img1", "img2"]


class TestRetryLogic:
    @respx.mock
    @pytest.mark.asyncio
    async def test_429_retries_then_succeeds(self, client: OpenAIImageClient):
        route = respx.post("https://api.openai.com/v1/images/generations")
        route.side_effect = [
            httpx.Response(429, text="rate limited"),
            httpx.Response(200, json={"data": [{"b64_json": "ok"}]}),
        ]
        result = await client.generate_image("test", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert result == ["ok"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_exhausted(self, client: OpenAIImageClient):
        respx.post("https://api.openai.com/v1/images/generations").mock(
            return_value=httpx.Response(429, text="rate limited"),
        )
        with pytest.raises(AssetError) as exc_info:
            await client.generate_image("test", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert exc_info.value.code == ErrorCode.OPENAI_RATE_LIMIT

    @respx.mock
    @pytest.mark.asyncio
    async def test_5xx_retries_once(self, client: OpenAIImageClient):
        route = respx.post("https://api.openai.com/v1/images/generations")
        route.side_effect = [
            httpx.Response(500, text="internal error"),
            httpx.Response(200, json={"data": [{"b64_json": "ok"}]}),
        ]
        result = await client.generate_image("test", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert result == ["ok"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_400_no_retry(self, client: OpenAIImageClient):
        respx.post("https://api.openai.com/v1/images/generations").mock(
            return_value=httpx.Response(400, text="bad request"),
        )
        with pytest.raises(AssetError) as exc_info:
            await client.generate_image("test", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert exc_info.value.code == ErrorCode.OPENAI_BAD_RESPONSE

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_auth_error(self, client: OpenAIImageClient):
        respx.post("https://api.openai.com/v1/images/generations").mock(
            return_value=httpx.Response(401, text="unauthorized"),
        )
        with pytest.raises(AssetError) as exc_info:
            await client.generate_image("test", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert exc_info.value.code == ErrorCode.OPENAI_AUTH_ERROR


class TestBadResponse:
    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_json(self, client: OpenAIImageClient):
        respx.post("https://api.openai.com/v1/images/generations").mock(
            return_value=httpx.Response(200, json={"unexpected": "format"}),
        )
        with pytest.raises(AssetError) as exc_info:
            await client.generate_image("test", "gpt-image-1", "1024x1024", "medium", "transparent")
        assert exc_info.value.code == ErrorCode.OPENAI_BAD_RESPONSE
