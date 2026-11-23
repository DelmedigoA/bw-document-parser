"""OpenRouter transport for the two opt-in hosted commands."""

import base64
import json
import os
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(ValueError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def blocks_run(self) -> bool:
        return self.status in {401, 402, 403, 404, 429, 500, 502, 503, 504}


def image_part(data: bytes, mime: str) -> dict:
    uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": uri}}


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
                retry_after = exc.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdecimal() else 1
                if attempt == 0 and exc.code in {429, 500, 502, 503, 504} and delay <= 5:
                    sleep(max(1, delay))
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
