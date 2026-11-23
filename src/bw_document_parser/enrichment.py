"""Describe previously parsed figures without reconverting the PDF."""

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import Timeout

from . import openrouter
from .bundle import (
    JsonWriter,
    api_image,
    asset_paths,
    document_lock,
    fingerprint,
    linked_text,
    pieces_with_context,
    read_document,
    read_image,
)
from .concurrency import resolve_workers
from .document import FigureEnrichment
from .prompts import FIGURE_DESCRIPTION, FIGURE_DESCRIPTION_VERSION

MODEL = "tencent/hy4-flash"
_SCHEMA = {
    "name": "figure_enrichment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decorative": {"type": "boolean"},
            "description": {"type": ["string", "null"]},
            "visible_text": {"type": ["string", "null"]},
        },
        "required": ["decorative", "description", "visible_text"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Description:
    description: str | None
    visible_text: str | None
    decorative: bool
    cost: float | None = None


def _description(
    context: dict, images: list[tuple[bytes, str]], *, with_cost: bool
):
    response = openrouter.post("chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": FIGURE_DESCRIPTION},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
                *(openrouter.image_part(data, mime) for data, mime in images),
            ]},
        ],
        "provider": {"require_parameters": True},
        "response_format": {"type": "json_schema", "json_schema": _SCHEMA},
        "reasoning": {"effort": "none"},
        "max_tokens": 4096,
    })
    try:
        choice = response["choices"][0]
        status = choice["finish_reason"]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise openrouter.OpenRouterError("OpenRouter returned a malformed completion") from None
    if status == "length":
        raise openrouter.OpenRouterError("Hy4 Flash reached its output token limit")
    if status == "content_filter":
        raise openrouter.OpenRouterError("OpenRouter filtered the figure description")
    if status != "stop":
        raise openrouter.OpenRouterError("Hy4 Flash did not finish the figure description")
    if not isinstance(content, str) or not content.strip():
        raise openrouter.OpenRouterError("Hy4 Flash returned an empty figure description")

    content = content.strip()
    if content.startswith(("```json\n", "```\n")) and content.endswith("```"):
        content = content.split("\n", 1)[1][:-3].strip()
    try:
        values = json.loads(content)
    except json.JSONDecodeError:
        raise openrouter.OpenRouterError("Hy4 Flash returned invalid figure-description JSON") from None
    if not isinstance(values, dict) or values.keys() - {"decorative", "description", "visible_text"}:
        raise openrouter.OpenRouterError("Hy4 Flash returned unexpected figure-description fields")
    decorative = values.get("decorative")
    if type(decorative) is not bool:
        raise openrouter.OpenRouterError("Hy4 Flash returned an invalid decorative flag")
    if decorative:
        result = Description(None, None, True, openrouter.cost_of(response))
        return result if with_cost else (None, None, True)
    description, visible_text = values.get("description"), values.get("visible_text")
    if not isinstance(description, str) or not description.strip():
        raise openrouter.OpenRouterError("Hy4 Flash returned no usable description text")
    if visible_text is not None and not isinstance(visible_text, str):
        raise openrouter.OpenRouterError("Hy4 Flash returned an invalid visible_text field")
    result = Description(
        description.strip(),
        (visible_text or "").strip() or None,
        False,
        openrouter.cost_of(response),
    )
    return result if with_cost else (result.description, result.visible_text, False)


def describe(
    context: dict, images: list[tuple[bytes, str]]
) -> tuple[str | None, str | None, bool]:
    return _description(context, images, with_cost=False)


@dataclass(frozen=True)
class _Job:
    piece: dict
    context: dict
    image_data: list[bytes]
    identity: str


@dataclass(frozen=True)
class _Result:
    job: _Job | None
    description: Description | None = None
    error: Exception | None = None


def enrich_bundle(bundle: Path, *, workers: int | None = None) -> bool:
    workers = resolve_workers(workers, "BW_ENRICH_WORKERS")
    with document_lock(bundle):
        document = read_document(bundle)
        writer = JsonWriter(bundle / "document.json", indent=2)
        enriched = omitted = reused = missing = failed = 0
        jobs: list[_Job] = []
        for piece, headings, by_id in pieces_with_context(document):
            if piece["content"]["kind"] != "figure":
                continue
            paths = asset_paths(piece)
            if not paths:
                missing += 1
                continue
            try:
                raw = [read_image(bundle, path) for path in paths]
                context = {
                    "section": " / ".join(headings)[-2000:],
                    "published_caption": "\n".join(linked_text(piece, by_id, "caption_ids"))[:4000],
                }
                identity = fingerprint(
                    MODEL, FIGURE_DESCRIPTION_VERSION, context,
                    [hashlib.sha256(data).hexdigest() for data in raw],
                )
                previous = piece.get("enrichment") or {}
                if previous.get("fingerprint") == identity and (
                    previous.get("description") or previous.get("decorative") is True
                ):
                    reused += 1
                    continue
                jobs.append(_Job(piece, context, raw, identity))
            except (OSError, ValueError) as exc:
                print(f"{piece['id']}: {exc}", file=sys.stderr)
                failed += 1

        stop = threading.Event()

        def run(job: _Job) -> _Result:
            if stop.is_set():
                return _Result(None)
            try:
                description = _description(
                    job.context,
                    [api_image(data, 2048) for data in job.image_data],
                    with_cost=True,
                )
            except (OSError, ValueError) as exc:
                return _Result(job, error=exc)
            return _Result(job, description=description)

        if jobs:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run, job): job for job in jobs}
                remaining = set(futures)
                interrupted = None
                while remaining:
                    try:
                        future = next(as_completed(remaining))
                        remaining.remove(future)
                        try:
                            result = future.result()
                        except CancelledError:
                            continue
                        job = result.job
                        if job is None:
                            continue
                        if result.error is not None:
                            print(f"{job.piece['id']}: {result.error}", file=sys.stderr)
                            failed += 1
                            if (
                                isinstance(result.error, openrouter.OpenRouterError)
                                and result.error.blocks_run
                            ):
                                stop.set()
                                for pending in futures:
                                    pending.cancel()
                            continue
                        description = result.description
                        job.piece["enrichment"] = asdict(FigureEnrichment(
                            description.description,
                            description.visible_text,
                            MODEL,
                            FIGURE_DESCRIPTION_VERSION,
                            job.identity,
                            asset_paths(job.piece),
                            description.decorative,
                            description.cost,
                        ))
                        omitted += int(description.decorative)
                        enriched += int(not description.decorative)
                        # Persist each completed request so a later quota or
                        # interrupt can resume without repeating paid work.
                        writer.write(document)
                    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- drain paid work before reraising
                        if interrupted is None:
                            interrupted = exc
                        stop.set()
                        for pending in remaining:
                            pending.cancel()
                if interrupted is not None:
                    raise interrupted

        print(
            f"Figures enriched: {enriched}, decorative: {omitted}, reused: {reused}, "
            f"no image: {missing}, failed: {failed} "
            f"(workers: {workers})"
        )
        return failed == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="concurrent Hy4 Flash requests (default: 4, env BW_ENRICH_WORKERS)",
    )
    args = parser.parse_args()
    try:
        workers = resolve_workers(args.workers, "BW_ENRICH_WORKERS")
        return 0 if enrich_bundle(args.bundle, workers=workers) else 1
    except (OSError, ValueError, Timeout) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
