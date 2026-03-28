"""Tests for S3Storage client using a real MinIO-like mock approach.

Since moto has compatibility issues with aiobotocore's async response handling,
these tests use AsyncMock-based unit tests for the S3Storage wrapper logic.
Integration testing with a real MinIO instance should be done via docker-compose.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asset_forge_mcp.models import AssetError, ErrorCode
from asset_forge_mcp.s3_client import S3Storage


def _make_storage(bucket: str = "test-bucket") -> S3Storage:
    return S3Storage(
        endpoint_url="http://localhost:9000",
        access_key="testing",
        secret_key="testing",
        region="us-east-1",
        bucket=bucket,
    )


class TestS3Storage:
    @pytest.mark.asyncio
    async def test_ensure_bucket_exists(self):
        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.head_bucket = AsyncMock()  # succeeds = bucket exists
        storage._client = mock_client

        await storage.ensure_bucket()
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @pytest.mark.asyncio
    async def test_ensure_bucket_creates_when_missing(self):
        from botocore.exceptions import ClientError

        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.head_bucket = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadBucket",
            )
        )
        mock_client.create_bucket = AsyncMock()
        storage._client = mock_client

        await storage.ensure_bucket()
        mock_client.create_bucket.assert_called_once_with(Bucket="test-bucket")

    @pytest.mark.asyncio
    async def test_ensure_bucket_access_denied_warns(self):
        from botocore.exceptions import ClientError

        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.head_bucket = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}},
                "HeadBucket",
            )
        )
        storage._client = mock_client

        # Should not raise — just logs a warning
        await storage.ensure_bucket()

    @pytest.mark.asyncio
    async def test_key_exists_true(self):
        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.head_object = AsyncMock()
        storage._client = mock_client

        assert await storage.key_exists("sprites/test.png") is True

    @pytest.mark.asyncio
    async def test_key_exists_false(self):
        from botocore.exceptions import ClientError

        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.head_object = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        )
        storage._client = mock_client

        assert await storage.key_exists("nonexistent.png") is False

    @pytest.mark.asyncio
    async def test_upload_bytes(self):
        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock()
        storage._client = mock_client

        await storage.upload_bytes(b"png-data", "sprites/test.png", "image/png")
        mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="sprites/test.png",
            Body=b"png-data",
            ContentType="image/png",
        )

    @pytest.mark.asyncio
    async def test_upload_bytes_error_raises(self):
        from botocore.exceptions import ClientError

        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "500", "Message": "Internal"}},
                "PutObject",
            )
        )
        storage._client = mock_client

        with pytest.raises(AssetError) as exc_info:
            await storage.upload_bytes(b"data", "key", "text/plain")
        assert exc_info.value.code == ErrorCode.S3_ERROR

    @pytest.mark.asyncio
    async def test_upload_json(self):
        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock()
        storage._client = mock_client

        await storage.upload_json({"name": "test"}, "meta.json")
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["Key"] == "meta.json"
        assert call_args.kwargs["ContentType"] == "application/json"
        body = json.loads(call_args.kwargs["Body"])
        assert body["name"] == "test"

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        storage = _make_storage()
        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        storage._client = mock_client

        await storage.close()
        assert storage._client is None
