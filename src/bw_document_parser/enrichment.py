"""Describe previously parsed figures without reconverting the PDF."""

import argparse
import hashlib
import json
import sys
import threading
from collections.abc import Callable, Collection
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from copy import deepcopy
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
from .document import (
    EnrichedDoc,
    FigureEnrichment,
    ResolvedDoc,
    apply_visual_decision,
    set_visual_enrichment,
    upgrade_resolved,
)
from .figure_decision import STRATEGY as DECISION_STRATEGY
from .figure_decision import Decision, decide_figure
from .prompts import FIGURE_DESCRIPTION, FIGURE_DESCRIPTION_VERSION
from .resolving import block_document, resolved_pieces

MODEL = "tencent/hy4-flash"
ENRICH_STRATEGY = "figure-description-v2"
_SCHEMA = {
    "name": "figure_enrichment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "description": {"type": ["string", "null"]},
            "visible_text": {"type": ["string", "null"]},
        },
        "required": ["description", "visible_text"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Description:
    description: str
    visible_text: str | None
    cost: float | None = None


def _description(context: dict, images: list[tuple[bytes, str]]) -> Description:
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
    if not isinstance(values, dict) or values.keys() - {"description", "visible_text"}:
        raise openrouter.OpenRouterError("Hy4 Flash returned unexpected figure-description fields")
    description, visible_text = values.get("description"), values.get("visible_text")
    if not isinstance(description, str) or not description.strip():
        raise openrouter.OpenRouterError("Hy4 Flash returned no usable description text")
    if visible_text is not None and not isinstance(visible_text, str):
        raise openrouter.OpenRouterError("Hy4 Flash returned an invalid visible_text field")
    return Description(
        description.strip(),
        (visible_text or "").strip() or None,
        openrouter.cost_of(response),
    )


def describe(
    context: dict, images: list[tuple[bytes, str]]
) -> tuple[str, str | None]:
    result = _description(context, images)
    return result.description, result.visible_text


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
    decision: Decision | None = None


def enrich_document(
    resolved: ResolvedDoc, bundle: Path, *, block_ids: Collection[str] | None = None,
    workers: int | None = None, save: Callable[[EnrichedDoc], None] | None = None,
) -> tuple[EnrichedDoc, bool]:
    """Enrich selected figures in a new snapshot, optionally saving progress."""
    document = upgrade_resolved(deepcopy(resolved))
    workers = resolve_workers(workers, "BW_ENRICH_WORKERS")
    selected = set(block_ids) if block_ids is not None else None
    owners = {}
    routed = document["schema_version"] == 4
    if routed:
        document["strategies"]["enrich"] = ENRICH_STRATEGY
        document["strategies"]["decide"] = DECISION_STRATEGY
        owners = {block_id: piece for piece in resolved_pieces(document) for block_id in piece["source_ids"]}
    enriched = skipped = reused = missing = failed = 0
    jobs: list[_Job] = []
    for piece, headings, by_id in pieces_with_context(block_document(document, include_excluded=True)):
        if piece["content"]["kind"] != "figure" or (selected is not None and piece["id"] not in selected):
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
            if previous.get("fingerprint") == identity and previous.get("description"):
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
        decision = None
        try:
            images = [api_image(data, 2048) for data in job.image_data]
            if routed:
                decision = decide_figure(
                    job.context, images, job.piece.get("enrichment_decision")
                )
                if not decision.enrich or stop.is_set():
                    return _Result(job, decision=decision)
            description = _description(job.context, images)
        except (OSError, ValueError) as exc:
            if isinstance(exc, openrouter.OpenRouterError) and exc.blocks_run:
                stop.set()
            return _Result(job, error=exc, decision=decision)
        return _Result(job, description=description, decision=decision)

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
                    decision = result.decision
                    owner = owners.get(job.piece["id"])
                    if decision is not None and owner is not None:
                        source = document["source_blocks"][job.piece["id"]]
                        record = asdict(decision)
                        changed = source.get("enrichment_decision") != record
                        apply_visual_decision(owner, source, record)
                        if changed and save is not None:
                            save(document)
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
                    if description is None:
                        skipped += int(decision is not None and not decision.enrich)
                        continue
                    job.piece["enrichment"] = asdict(FigureEnrichment(
                        description.description,
                        description.visible_text,
                        MODEL,
                        FIGURE_DESCRIPTION_VERSION,
                        job.identity,
                        asset_paths(job.piece),
                        cost=description.cost,
                    ))
                    if owner is not None:
                        set_visual_enrichment(
                            owner["visual_part"] or [], document["source_blocks"],
                            job.piece["id"], job.piece["enrichment"],
                        )
                    enriched += 1
                    # Persist each completed request so a later quota or
                    # interrupt can resume without repeating paid work.
                    if save is not None:
                        save(document)
                except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- drain paid work before reraising
                    if interrupted is None:
                        interrupted = exc
                    stop.set()
                    for pending in remaining:
                        pending.cancel()
            if interrupted is not None:
                raise interrupted

    print(
        f"Figures enriched: {enriched}, skipped by Decider: {skipped}, reused: {reused}, "
        f"no image: {missing}, failed: {failed} "
        f"(workers: {workers})"
    )
    if save is not None:
        save(document)
    return document, failed == 0



def enrich_bundle(bundle: Path, *, workers: int | None = None) -> bool:
    with document_lock(bundle):
        writer = JsonWriter(bundle / "document.json")
        _, complete = enrich_document(read_document(bundle), bundle, workers=workers, save=writer.write)
        return complete

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
