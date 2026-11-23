"""Shared limits for the optional hosted stages."""

import os

DEFAULT_WORKERS = 4
MAX_WORKERS = 32


def resolve_workers(
    value: int | None,
    environment: str,
    *,
    default: int = DEFAULT_WORKERS,
) -> int:
    if value is None:
        value = os.getenv(environment, default)
    try:
        workers = int(value)
    except (TypeError, ValueError):
        raise ValueError("worker count must be an integer") from None
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"worker count must be between 1 and {MAX_WORKERS}")
    return workers
