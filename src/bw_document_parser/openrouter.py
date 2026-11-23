"""OpenRouter transport for the opt-in hosted commands."""

import base64
import json
import math
import os
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_URL = "https://openrouter.ai/api/v1"
_RETRYABLE = {408, 425, 429, 500, 502, 503, 504, 529}


class OpenRouterError(ValueError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def blocks_run(self) -> bool:
        return self.status in {401, 402, 403, 404, 408, 425, 429, 500, 502, 503, 504, 529}


def image_part(data: bytes, mime: str) -> dict:
    uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": uri}}


def cost_of(response: dict) -> float | None:
    """Read the per-request cost OpenRouter includes in usage accounting."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        return None
    return float(cost)


def _retry_after(headers) -> float | None:
    value = (headers.get("Retry-After") or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def post(endpoint: str, payload: dict) -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("set OPENROUTER_API_KEY for hosted processing", status=401)
    request = Request(
        f"{_URL}/{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=90) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise OpenRouterError("OpenRouter returned an invalid response")
            if error := result.get("error"):
                code = error.get("code") if isinstance(error, dict) else None
                if isinstance(code, str):
                    code = int(code) if code.isdecimal() else None
                raise OpenRouterError(f"OpenRouter returned an error ({code or 'unknown'})", status=code)
            return result
        except HTTPError as exc:
            try:
                delay = _retry_after(exc.headers)
                # A short throttle can be retried once. A long Retry-After means
                # the quota is exhausted, so return control and let the caller
                # save completed work instead of hammering the endpoint.
                if (
                    attempt == 0
                    and exc.code in _RETRYABLE
                    and (delay is None or delay <= 5)
                ):
                    sleep(max(1.0, delay or 0.0))
                    continue
                message = f"OpenRouter returned HTTP {exc.code}"
                if exc.code == 429:
                    message += ". Rate limit reached. Rerun to resume after the limit resets"
                raise OpenRouterError(message, status=exc.code) from None
            finally:
                exc.close()
        except (URLError, TimeoutError):
            if attempt == 0:
                sleep(1)
                continue
            raise OpenRouterError("OpenRouter request timed out or could not connect", status=503) from None
        except (UnicodeError, json.JSONDecodeError):
            raise OpenRouterError("OpenRouter returned invalid JSON") from None
    raise AssertionError("unreachable")
