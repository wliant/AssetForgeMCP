"""S3-compatible storage client (works with MinIO and AWS S3)."""

from __future__ import annotations

import json
import logging

import aioboto3
from botocore.exceptions import ClientError

from .models import AssetError, ErrorCode

logger = logging.getLogger(__name__)


class S3Storage:
    """Async S3 storage wrapper using aioboto3."""

    def __init__(
        self,
        endpoint_url: str | None,
        access_key: str,
        secret_key: str,
        region: str,
        bucket: str,
    ) -> None:
        self.bucket = bucket
        self._session = aioboto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._endpoint_url = endpoint_url
        self._client = None

    async def _get_client(self):
        if self._client is None:
            self._client = await self._session.client(
                "s3", endpoint_url=self._endpoint_url
            ).__aenter__()
        return self._client

    async def ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist. Safe to call in prod."""
        client = await self._get_client()
        try:
            await client.head_bucket(Bucket=self.bucket)
            logger.info("S3 bucket '%s' exists", self.bucket)
        except ClientError as exc:
            code = exc.response["Error"].get("Code", "")
            if code in ("404", "NoSuchBucket"):
                try:
                    await client.create_bucket(Bucket=self.bucket)
                    logger.info("Created S3 bucket '%s'", self.bucket)
                except ClientError as create_exc:
                    error_code = create_exc.response["Error"].get("Code", "")
                    if error_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                        logger.info("S3 bucket '%s' already exists", self.bucket)
                    else:
                        raise AssetError(
                            ErrorCode.S3_ERROR,
                            f"Cannot create bucket '{self.bucket}': {create_exc}",
                        ) from create_exc
            elif code in ("403", "AccessDenied"):
                logger.warning(
                    "Cannot verify bucket '%s' (AccessDenied) — assuming it exists",
                    self.bucket,
                )
            else:
                raise AssetError(
                    ErrorCode.S3_ERROR,
                    f"Cannot access bucket '{self.bucket}': {exc}",
                ) from exc

    async def key_exists(self, key: str) -> bool:
        """Check if an object exists in the bucket."""
        client = await self._get_client()
        try:
            await client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    async def upload_bytes(self, data: bytes, key: str, content_type: str) -> None:
        """Upload raw bytes to S3."""
        client = await self._get_client()
        try:
            await client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        except ClientError as exc:
            raise AssetError(
                ErrorCode.S3_ERROR, f"Failed to upload '{key}': {exc}"
            ) from exc

    async def upload_json(self, obj: dict, key: str) -> None:
        """Upload a JSON-serializable dict as a .json object."""
        data = json.dumps(obj, indent=2, default=str).encode("utf-8")
        await self.upload_bytes(data, key, content_type="application/json")

    async def close(self) -> None:
        """Clean up the S3 client."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
