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

from . import figure_decision, openrouter
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
from .concurrency import resolve_batch_size, resolve_workers
from .document import (
    EnrichedDoc,
    FigureEnrichment,
    ResolvedDoc,
    apply_visual_decision,
    set_visual_enrichment,
    upgrade_resolved,
)
from .figure_decision import STRATEGY as DECISION_STRATEGY
from .figure_decision import Decision
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
    images: list[tuple[bytes, str]]
    identity: str


@dataclass(frozen=True)
class _Result:
    job: _Job | None
    description: Description | None = None
    error: Exception | None = None


def enrich_document(
    resolved: ResolvedDoc, bundle: Path, *, block_ids: Collection[str] | None = None,
    workers: int | None = None, decider_batch_size: int | None = None,
    save: Callable[[EnrichedDoc], None] | None = None,
) -> tuple[EnrichedDoc, bool]:
    """Enrich selected figures in a new snapshot, optionally saving progress."""
    document = upgrade_resolved(deepcopy(resolved))
    workers = resolve_workers(workers, "BW_ENRICH_WORKERS")
    decide_batch_size = resolve_batch_size(
        decider_batch_size, "BW_DECIDER_BATCH_SIZE",
        default=figure_decision.DEFAULT_BATCH_SIZE,
        maximum=figure_decision.MAX_BATCH_SIZE,
    )
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
            images = [api_image(data, 2048) for data in raw]
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
            jobs.append(_Job(piece, context, images, identity))
        except (OSError, ValueError) as exc:
            print(f"{piece['id']}: {exc}", file=sys.stderr)
            failed += 1

    stop = threading.Event()
    decided: dict[str, Decision] = {}
    requests: list[figure_decision.FigureRequest] = []
    if routed:
        for job in jobs:
            request = figure_decision.prepare_request(
                job.piece["id"], job.context, job.images
            )
            cached = figure_decision.cached_decision(
                request, job.piece.get("enrichment_decision")
            )
            if cached is None:
                requests.append(request)
            else:
                decided[job.piece["id"]] = cached

    decision_requests = decision_batches = decision_singles = 0
    decision_unanswered = decision_unpriced = 0
    decision_cost: float | None = None
    ledger = document.setdefault("decider_requests", []) if routed else None
    ledger_ids = {
        entry.get("id") for entry in ledger or () if isinstance(entry, dict) and entry.get("id")
    }

    def record_decisions(outcome: figure_decision.Outcome) -> None:
        nonlocal failed, decision_requests, decision_batches, decision_singles
        nonlocal decision_unanswered, decision_unpriced, decision_cost
        decision_requests += outcome.requests
        decision_unanswered += max(0, outcome.requests - len(outcome.ledger))
        decision_batches += sum(entry["kind"] == "batch" for entry in outcome.ledger)
        decision_singles += sum(entry["kind"] == "single" for entry in outcome.ledger)
        decision_unpriced += sum(entry.get("cost") is None for entry in outcome.ledger)
        if outcome.actual_cost is not None:
            decision_cost = (
                outcome.actual_cost
                if decision_cost is None
                else decision_cost + outcome.actual_cost
            )
        changed = False
        if outcome.ledger and ledger is not None:
            for entry in outcome.ledger:
                identity = entry.get("id")
                if identity and identity in ledger_ids:
                    continue
                ledger.append(entry)
                if identity:
                    ledger_ids.add(identity)
                changed = True
        for piece_id, decision in outcome.decisions.items():
            decided[piece_id] = decision
            owner = owners.get(piece_id)
            if owner is None:
                continue
            source = document["source_blocks"][piece_id]
            record = asdict(decision)
            if source.get("enrichment_decision") == record:
                continue
            apply_visual_decision(owner, source, record)
            changed = True
        if changed and save is not None:
            # Persist the whole completed batch, including its one actual
            # charge, before waiting for the next batch.
            save(document)
        for piece_id, exc in outcome.errors.items():
            print(f"{piece_id}: {exc}", file=sys.stderr)
            failed += 1
            if isinstance(exc, openrouter.OpenRouterError) and exc.blocks_run:
                stop.set()

    if requests:
        figure_decision.run_decisions(
            requests, batch_size=decide_batch_size, workers=workers,
            stop=stop, on_outcome=record_decisions,
        )

    if routed:
        skipped = sum(
            1 for job in jobs
            if (decision := decided.get(job.piece["id"])) is not None
            and not decision.enrich
        )
        approved = [
            job for job in jobs
            if (decision := decided.get(job.piece["id"])) is not None and decision.enrich
        ]
    else:
        approved = jobs

    def run(job: _Job) -> _Result:
        if stop.is_set():
            return _Result(None)
        try:
            description = _description(job.context, job.images)
        except (OSError, ValueError) as exc:
            if isinstance(exc, openrouter.OpenRouterError) and exc.blocks_run:
                stop.set()
            return _Result(job, error=exc)
        return _Result(job, description=description)

    if approved and not stop.is_set():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, job): job for job in approved}
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
                    if description is None:
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
                    owner = owners.get(job.piece["id"])
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

    suffix = f"(workers: {workers}"
    if routed:
        if not decision_requests:
            cost = "none"
        elif decision_cost is None:
            cost = "unavailable"
        elif decision_unpriced:
            cost = f"${decision_cost:.6f} + {decision_unpriced} unpriced"
        else:
            cost = f"${decision_cost:.6f}"
        breakdown = f"{decision_batches} batch, {decision_singles} single"
        if decision_unanswered:
            breakdown += f", {decision_unanswered} no response"
        suffix += (
            f", decider batch: {decide_batch_size}"
            f", decider requests: {decision_requests} ({breakdown})"
            f", decider cost: {cost}"
        )
    suffix += ")"
    print(
        f"Figures enriched: {enriched}, skipped by Decider: {skipped}, reused: {reused}, "
        f"no image: {missing}, failed: {failed} {suffix}"
    )
    if save is not None:
        save(document)
    return document, failed == 0


def enrich_bundle(
    bundle: Path, *, workers: int | None = None,
    decider_batch_size: int | None = None,
) -> bool:
    with document_lock(bundle):
        writer = JsonWriter(bundle / "document.json")
        _, complete = enrich_document(
            read_document(bundle), bundle, workers=workers,
            decider_batch_size=decider_batch_size, save=writer.write,
        )
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
    parser.add_argument(
        "--decider-batch-size",
        type=int,
        default=None,
        help=(
            "figures per Perplexity Decider request (default: 1, env "
            "BW_DECIDER_BATCH_SIZE; opt in above 1)"
        ),
    )
    args = parser.parse_args()
    try:
        workers = resolve_workers(args.workers, "BW_ENRICH_WORKERS")
        batch_size = resolve_batch_size(
            args.decider_batch_size, "BW_DECIDER_BATCH_SIZE",
            default=figure_decision.DEFAULT_BATCH_SIZE,
            maximum=figure_decision.MAX_BATCH_SIZE,
        )
        return 0 if enrich_bundle(
            args.bundle, workers=workers, decider_batch_size=batch_size,
        ) else 1
    except (OSError, ValueError, Timeout) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
