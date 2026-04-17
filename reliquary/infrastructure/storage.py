"""R2/S3 object storage for window rollout files."""

import asyncio
import gzip
import hashlib
import hmac
import json
import logging
import os
from typing import Any

from aiobotocore.session import get_session

from reliquary.constants import MAX_ROLLOUT_FILE_SIZE_BYTES
from botocore.config import Config

logger = logging.getLogger(__name__)

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = get_session()
    return _SESSION


def get_s3_client(
    account_id: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    bucket_name: str | None = None,
):
    """Create S3 client context for R2."""
    account_id = account_id or os.getenv("R2_ACCOUNT_ID", "")
    access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID", "")
    secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY", "")
    endpoint = os.getenv("R2_ENDPOINT_URL") or f"https://{account_id}.r2.cloudflarestorage.com"
    region = os.getenv("R2_REGION", "us-east-1")

    config = Config(
        connect_timeout=3,
        read_timeout=30,
        retries={"max_attempts": 2},
    )
    return _get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )


async def upload_json(key: str, data: Any, **client_kwargs) -> bool:
    """Upload JSON data to S3."""
    payload = json.dumps(data, separators=(",", ":")).encode()
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=payload)
    return True


async def download_json(key: str, **client_kwargs) -> dict | None:
    """Download and parse JSON from S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            return json.loads(body)
    except Exception as e:
        logger.debug("download_json failed for %s: %s", key, e)
        return None


async def file_exists(key: str, **client_kwargs) -> bool:
    """Check if file exists in S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
            await client.head_object(Bucket=bucket, Key=key)
            return True
    except Exception:
        return False


def _state_hmac_key() -> bytes:
    """Derive HMAC key for state file integrity from validator's secret.

    Uses R2_SECRET_ACCESS_KEY as the base secret — only the validator who
    owns the bucket can produce or verify the HMAC.
    """
    secret = os.getenv("R2_SECRET_ACCESS_KEY", "").encode()
    return hashlib.sha256(b"reliquary-state-hmac|" + secret).digest()


def _compute_state_hmac(data: bytes) -> str:
    return hmac.new(_state_hmac_key(), data, hashlib.sha256).hexdigest()


async def load_used_indices(validator_id: str = "", **client_kwargs) -> dict[int, str]:
    """Load the validator's used-indices map from S3.

    Each validator uses its own state file to prevent concurrent validators
    from overwriting each other's state.

    Returns:
        {dataset_index: hotkey} for every index ever credited.
        Empty dict if file doesn't exist yet or integrity check fails.
    """
    suffix = f"-{validator_id}" if validator_id else ""
    key = f"reliquary/state/used_indices{suffix}.json.gz"
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            body = gzip.decompress(body)
            raw = json.loads(body)

            # SECURITY: Verify HMAC to detect tampering with the state file.
            stored_hmac = raw.get("_hmac")
            if stored_hmac:
                data_without_hmac = {k: v for k, v in raw.items() if k != "_hmac"}
                payload = json.dumps(data_without_hmac, sort_keys=True, separators=(",", ":")).encode()
                expected_hmac = _compute_state_hmac(payload)
                if not hmac.compare_digest(stored_hmac, expected_hmac):
                    logger.error(
                        "SECURITY: used_indices HMAC mismatch — state file may be tampered!"
                    )
                    return {}
            else:
                logger.warning("used_indices has no HMAC — first run or legacy file")

            # JSON keys are strings — convert back to int
            return {int(k): v for k, v in raw.items() if k != "_hmac"}
    except Exception as e:
        logger.info("No existing used_indices found (starting fresh): %s", e)
        return {}


async def save_used_indices(used: dict[int, str], validator_id: str = "", **client_kwargs) -> bool:
    """Save the validator's used-indices map to S3 with HMAC integrity."""
    suffix = f"-{validator_id}" if validator_id else ""
    key = f"reliquary/state/used_indices{suffix}.json.gz"
    # JSON keys must be strings
    data = {str(k): v for k, v in used.items()}
    # Compute HMAC over the data before adding the HMAC field
    payload_for_hmac = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["_hmac"] = _compute_state_hmac(payload_for_hmac)

    payload = json.dumps(data, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info("Saved %d used indices (%d bytes, HMAC protected)", len(used), len(compressed))
    return True


async def save_window_results(
    window_start: int, results: dict, **client_kwargs
) -> bool:
    """Save validation results for a window to S3."""
    key = f"reliquary/results/window-{window_start}.json.gz"
    payload = json.dumps(results, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info("Saved results for window %d (%d bytes)", window_start, len(compressed))
    return True


async def upload_window_rollouts(
    hotkey: str, window_start: int, rollouts: list[dict], **client_kwargs
) -> bool:
    """Upload window rollouts as gzipped JSON."""
    key = f"reliquary/windows/{hotkey}-window-{window_start}.json.gz"
    payload = json.dumps(rollouts, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info(
        "Uploaded %d rollouts for window %d (%d bytes)",
        len(rollouts), window_start, len(compressed),
    )
    return True


async def upload_window_dataset(
    window_start: int, data: dict, **client_kwargs
) -> bool:
    """Upload the settled GRPO dataset (prompt + 32 completions + rewards) for a window.

    The output of this is the actual deliverable of the network: a stream of
    {prompt, completions, rewards} bundles ready to feed a training pipeline.
    Stored under `reliquary/dataset/window-{n}.json.gz`.
    """
    key = f"reliquary/dataset/window-{window_start}.json.gz"
    payload = json.dumps(data, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info(
        "Uploaded GRPO dataset for window %d (%d slots, %d bytes)",
        window_start, len(data.get("slots", [])), len(compressed),
    )
    return True


async def download_window_rollouts(
    hotkey: str, window_start: int, **client_kwargs
) -> tuple[list[dict] | None, float | None]:
    """Download window rollouts and return S3 LastModified timestamp.

    Returns:
        Tuple of (rollouts, upload_time_unix). upload_time is from S3
        LastModified header. Both are None if file not found.
    """
    key = f"reliquary/windows/{hotkey}-window-{window_start}.json.gz"
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")

            # SECURITY: Check file size before downloading to prevent
            # memory exhaustion from oversized or zip-bomb uploads.
            head = await client.head_object(Bucket=bucket, Key=key)
            content_length = head.get("ContentLength", 0)
            if content_length > MAX_ROLLOUT_FILE_SIZE_BYTES:
                logger.warning(
                    "Rollout file %s too large: %d bytes (max %d)",
                    key, content_length, MAX_ROLLOUT_FILE_SIZE_BYTES,
                )
                return None, None

            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()

            if key.endswith(".gz"):
                # Limit decompressed size to prevent zip bombs.
                # A 10:1 ratio is generous for JSON; anything beyond is suspicious.
                max_decompressed = min(
                    MAX_ROLLOUT_FILE_SIZE_BYTES,
                    content_length * 20,
                )
                decompressed = gzip.decompress(body)
                if len(decompressed) > max_decompressed:
                    logger.warning(
                        "Decompressed rollout %s too large: %d bytes "
                        "(compressed %d, limit %d)",
                        key, len(decompressed), content_length, max_decompressed,
                    )
                    return None, None
                body = decompressed

            data = json.loads(body)

            upload_time = None
            last_modified = head.get("LastModified")
            if last_modified is not None:
                upload_time = last_modified.timestamp()

            if isinstance(data, list):
                return data, upload_time
            return None, None
    except Exception as e:
        logger.debug("download_window_rollouts failed for %s: %s", key, e)
        return None, None
