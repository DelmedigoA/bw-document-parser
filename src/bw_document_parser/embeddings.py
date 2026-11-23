"""Generate portable embeddings and search a single PDF bundle."""

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filelock import Timeout

from . import openrouter
from .bundle import (
    JsonWriter,
    api_image,
    document_lock,
    fingerprint,
    read_document,
    read_image,
    read_json,
)
from .chunks import MAX_TOKENS, _token_count
from .concurrency import resolve_workers
from .document import EmbeddedDoc, EnrichedDoc, upgrade_resolved
from .resolving import block_document, resolved_pieces
from .search_entries import SearchEntry, search_entries

NEMOTRON = "nvidia/nemotron-3.5-embed-8b"
TITAN = "amazon/titan-embed-image-v1"
# Leave headroom under OpenRouter's 128-token limit for tokenizer differences.
TITAN_CHUNK_TOKENS = 96
TEXT_MODEL = "openai/text-embedding-3-small"
DEFAULT_MODE = "text"
MODES = {"text": (TEXT_MODEL, 1536), "multimodal": (f"{NEMOTRON}:free", 4096)}
# Rewriting the whole index is expensive once vectors accumulate.
SAVE_INTERVAL = 10.0
EMBED_STRATEGY = "piece-targets-v1"

# OpenAI accepts an array of strings. Titan embeds one item per request.
# Nemotron uses the same OpenRouter array shape; an unsupported batch falls
# back to single requests without losing the rest of the run.
DEFAULT_TEXT_BATCH = {TEXT_MODEL: 64, TITAN: 1, NEMOTRON: 8}


def _valid_vector(vector, dimensions: int) -> bool:
    return (
        isinstance(vector, list)
        and len(vector) == dimensions
        and all(type(value) in (int, float) and math.isfinite(value) for value in vector)
        and any(vector)
    )


def _payload(
    model: str, dimensions: int, inputs: list, *, query: bool = False
) -> dict:
    payload = {"model": model, "input": inputs, "encoding_format": "float"}
    if model in {TEXT_MODEL, TITAN}:
        payload["dimensions"] = dimensions
    else:
        payload["input_type"] = "query" if query else "document"
    return payload


def _request_embeddings(
    model: str, dimensions: int, inputs: list, *, query: bool = False
) -> tuple[dict[int, list[float]], float | None]:
    """Request one batch and match vectors by the response's own index."""
    response = openrouter.post("embeddings", _payload(model, dimensions, inputs, query=query))
    try:
        items = response["data"]
    except (KeyError, TypeError):
        raise openrouter.OpenRouterError(
            f"OpenRouter returned an invalid {dimensions}-dimensional embedding"
        ) from None
    if not isinstance(items, list):
        raise openrouter.OpenRouterError(
            f"OpenRouter returned an invalid {dimensions}-dimensional embedding"
        )
    vectors: dict[int, list[float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if type(index) is not int or not 0 <= index < len(inputs) or index in vectors:
            continue
        vector = item.get("embedding")
        if _valid_vector(vector, dimensions):
            vectors[index] = vector
    return vectors, openrouter.cost_of(response)


def embed(
    model: str, dimensions: int, text: str, image: tuple[bytes, str] | None = None, *, query: bool = False,
) -> list[float]:
    if model == TITAN and _token_count(text) > TITAN_CHUNK_TOKENS:
        raise ValueError("Titan text input is too long. Use a shorter query or chunk.")
    value = text
    if image is not None:
        content = [{"type": "text", "text": text}] if text else []
        content.append(openrouter.image_part(*image))
        value = {"content": content}
    vectors, _ = _request_embeddings(model, dimensions, [value], query=query)
    vector = vectors.get(0)
    if vector is None:
        raise openrouter.OpenRouterError(
            f"OpenRouter returned an invalid {dimensions}-dimensional embedding"
        )
    return vector


def _path(bundle: Path, mode: str) -> Path:
    return bundle / f"embeddings.{mode}.json"


@dataclass(frozen=True)
class _Work:
    key: str
    text: str
    image: tuple[bytes, str] | None
    sources: list[SearchEntry]
    tokens: int


def _add_costs(
    left: tuple[float | None, str], right: tuple[float | None, str]
) -> tuple[float | None, str]:
    """Add the paid requests that produced one input's vector.

    A partial batch response is paid for the whole batch and then retried for
    the missing inputs, so one input can carry more than one request's cost.
    """
    left_cost, left_allocation = left
    right_cost, right_allocation = right
    if left_cost is None:
        cost = right_cost
    elif right_cost is None:
        cost = left_cost
    else:
        cost = left_cost + right_cost
    allocation = (
        "estimated"
        if "estimated" in {left_allocation, right_allocation}
        else "exact"
    )
    return cost, allocation


@dataclass
class _Outcome:
    vectors: dict[str, list[float]] = field(default_factory=dict)
    costs: dict[str, tuple[float | None, str]] = field(default_factory=dict)
    errors: list[tuple[_Work, Exception]] = field(default_factory=list)
    blocked: bool = False
    requests: int = 0

    def merge(self, other: "_Outcome") -> None:
        self.vectors.update(other.vectors)
        for key, value in other.costs.items():
            self.costs[key] = (
                _add_costs(self.costs[key], value) if key in self.costs else value
            )
        self.errors.extend(other.errors)
        self.blocked = self.blocked or other.blocked
        self.requests += other.requests


def _cost_shares(
    works: list[_Work], cost: float | None
) -> dict[str, tuple[float | None, str]]:
    allocation = "exact" if len(works) == 1 else "estimated"
    if cost is None:
        return {work.key: (None, allocation) for work in works}
    weights = [max(1, work.tokens) for work in works]
    total = sum(weights)
    return {
        work.key: (cost * weight / total, allocation)
        for work, weight in zip(works, weights)
    }


def _execute_batch(
    works: list[_Work],
    model: str,
    dimensions: int,
    stop: threading.Event | None = None,
) -> _Outcome:
    if stop is not None and stop.is_set():
        return _Outcome()
    inputs = []
    for work in works:
        if work.image is None:
            inputs.append(work.text)
        else:
            content = [{"type": "text", "text": work.text}] if work.text else []
            content.append(openrouter.image_part(*work.image))
            inputs.append({"content": content})
    outcome = _Outcome(requests=1)
    try:
        vectors, cost = _request_embeddings(model, dimensions, inputs)
    except openrouter.OpenRouterError as exc:
        if len(works) > 1 and not exc.blocks_run:
            for work in works:
                single = _execute_batch([work], model, dimensions, stop)
                outcome.merge(single)
                if single.blocked:
                    break
            return outcome
        outcome.errors.extend((work, exc) for work in works)
        outcome.blocked = exc.blocks_run
        if outcome.blocked and stop is not None:
            # Set the shared flag here so a worker that has already queued the
            # next batch does not spend another request on exhausted quota.
            stop.set()
        return outcome

    outcome.costs.update(_cost_shares(works, cost))
    missing = []
    for index, work in enumerate(works):
        vector = vectors.get(index)
        if vector is None:
            missing.append(work)
        else:
            outcome.vectors[work.key] = vector
    if missing and len(works) == 1:
        # A single input that comes back without a vector cannot be retried
        # without looping forever on the same response.
        error = openrouter.OpenRouterError(
            f"OpenRouter returned no valid {dimensions}-dimensional embedding"
        )
        outcome.errors.extend((work, error) for work in missing)
        return outcome
    for work in missing:
        # Retry only the entries the provider left out of a partial response.
        single = _execute_batch([work], model, dimensions, stop)
        outcome.merge(single)
        if single.blocked:
            break
    return outcome


def _pages(entry: dict) -> list[int]:
    return sorted({
        location["page"]
        for location in entry.get("locations", [])
        if type(location.get("page")) is int
    })


def _billing(entries: dict[str, dict]) -> dict:
    costs = []
    by_page: dict[str, float] = {}
    allocation = set()
    unpriced = 0
    for entry in entries.values():
        cost = entry.get("cost")
        if type(cost) not in (int, float) or not math.isfinite(cost):
            unpriced += 1
            continue
        costs.append(cost)
        allocation.add(entry.get("cost_allocation", "estimated"))
        pages = _pages(entry)
        if pages:
            share = cost / len(pages)
            for page in pages:
                by_page[str(page)] = by_page.get(str(page), 0.0) + share
        else:
            by_page["unattributed"] = by_page.get("unattributed", 0.0) + cost
    if not costs:
        label = "none"
    elif allocation == {"exact"}:
        label = "exact"
    elif allocation == {"estimated"}:
        label = "estimated"
    else:
        label = "mixed"
    return {
        "allocation": label,
        "total_cost": math.fsum(costs) if costs else None,
        "by_page": by_page,
        "priced_entries": len(costs),
        "unpriced_entries": unpriced,
    }


def _text_batch_size(model: str, value: int | None = None) -> int:
    if value is not None:
        if value < 1:
            raise ValueError("embedding batch size must be positive")
        return value
    return DEFAULT_TEXT_BATCH.get(model.split(":", 1)[0], 1)


def _batches(works: list[_Work], model: str, batch_size: int | None) -> list[list[_Work]]:
    text = [work for work in works if work.image is None]
    images = [work for work in works if work.image is not None]
    text_size = _text_batch_size(model, batch_size)
    batches = [
        text[index : index + text_size] for index in range(0, len(text), text_size)
    ]
    # Image embeddings are one per request across the supported providers.
    batches.extend([work] for work in images)
    return batches


def _input_document_fingerprint(blocks: dict) -> str:
    """Ignore routing audit metadata when checking whether embedding inputs changed."""
    return fingerprint({
        **blocks,
        "pieces": [
            {key: value for key, value in block.items() if key != "enrichment_decision"}
            for block in blocks["pieces"]
        ],
    })


def _resolved_index(document: EmbeddedDoc, mode: str) -> dict:
    metadata = document.get("embedding_modes", {}).get(mode)
    if metadata is None:
        return {}
    return {
        **metadata,
        "entries": [
            entry for piece in resolved_pieces(document)
            for entry in piece["embeddings"] if entry["mode"] == mode
        ],
    }


def embed_document(
    enriched: EnrichedDoc, bundle: Path, mode: str = DEFAULT_MODE, model_override: str | None = None, *,
    targets: Iterable[SearchEntry] | None = None,
    workers: int | None = None, batch_size: int | None = None,
    save: Callable[[dict], None] | None = None,
) -> tuple[EmbeddedDoc, bool]:
    """Embed the standard inputs or an explicit set, preserving piece identity."""
    document = upgrade_resolved(deepcopy(enriched))
    blocks = block_document(document)
    owners = {
        block_id: piece for piece in resolved_pieces(document) for block_id in piece["source_ids"]
    } if document["schema_version"] == 4 else {}
    workers = resolve_workers(workers, "BW_EMBED_WORKERS")
    default_model, dimensions = MODES[mode]
    model = model_override or default_model
    if model != default_model and (mode != "multimodal" or model not in {NEMOTRON, TITAN}):
        raise ValueError(f"{model} is unsupported in {mode} mode")
    if model == TITAN:
        dimensions = 1024

    old = _resolved_index(document, mode) if document["schema_version"] == 4 else {}
    path = _path(bundle, mode)
    if not old and path.is_file():
        # Upgraded bundles can still reuse exact inputs from a legacy index.
        old = read_json(path)

    cached = {}
    if old.get("model") == model and old.get("dimensions") == dimensions:
        cached = {
            item["input_fingerprint"]: item
            for item in old.get("entries", [])
            if isinstance(item, dict)
            and "input_fingerprint" in item
            and _valid_vector(item.get("vector"), dimensions)
        }
    cached_costs = {}
    for item in old.get("entries", []):
        key = item.get("input_fingerprint")
        cost = item.get("cost")
        if key in cached and type(cost) in (int, float) and math.isfinite(cost):
            cached_costs[key] = cached_costs.get(key, 0.0) + cost
    entries = {}
    pending: dict[str, _Work] = {}
    sources = list(targets) if targets is not None else list(search_entries(
        blocks, multimodal=mode == "multimodal",
        max_tokens=TITAN_CHUNK_TOKENS if model == TITAN else MAX_TOKENS,
    ))
    if document["schema_version"] == 4 and any(
        not source.piece_ids or any(block_id not in owners for block_id in source.piece_ids)
        for source in sources
    ):
        raise ValueError("embedding target must reference a resolved block")
    for source in sources:
        try:
            raw = read_image(bundle, source.image_path) if source.image_path else None
            image_hash = hashlib.sha256(raw).hexdigest() if raw is not None else None
            identity = fingerprint(model, dimensions, "document-v1", source.text, image_hash)
            previous = cached.get(identity)
            if previous is not None:
                entry = {
                    **asdict(source),
                    "input_fingerprint": identity,
                    "vector": previous["vector"],
                }
                entries[source.id] = entry
                continue
            work = pending.get(identity)
            if work is None:
                pending[identity] = _Work(
                    identity,
                    source.text,
                    api_image(raw, 1540) if raw is not None else None,
                    [source],
                    max(1, _token_count(source.text)),
                )
            else:
                work.sources.append(source)
        except (OSError, ValueError) as exc:
            print(f"{source.id}: {exc}", file=sys.stderr)

    cached_groups = {}
    for entry in entries.values():
        cached_groups.setdefault(entry["input_fingerprint"], []).append(entry)
    for key, group in cached_groups.items():
        if key in cached_costs:
            for entry in group:
                entry["cost"] = cached_costs[key] / len(group)
                entry["cost_allocation"] = "estimated" if len(group) > 1 else cached[key].get("cost_allocation", "estimated")

    reused = len(entries)
    writes = 0
    document_fingerprint = _input_document_fingerprint(blocks)

    def snapshot() -> dict:
        return {
            "schema_version": 1,
            "document_id": document["document_id"],
            "document_fingerprint": document_fingerprint,
            "model": model,
            "dimensions": dimensions,
            "mode": mode,
            "complete": len(entries) == len(sources),
            "expected_entries": len(sources),
            "entries": [entries[source.id] for source in sources if source.id in entries],
            "billing": _billing(entries),
        }

    def save_snapshot():
        nonlocal writes
        index = snapshot()
        if document["schema_version"] == 4:
            for piece in resolved_pieces(document):
                piece["embeddings"] = [item for item in piece["embeddings"] if item["mode"] != mode]
            for entry in index["entries"]:
                owners[entry["piece_ids"][0]]["embeddings"].append({
                    **entry, "mode": mode, "model": model, "dimensions": dimensions,
                })
            document.setdefault("embedding_modes", {})[mode] = {
                key: value for key, value in index.items() if key != "entries"
            }
            document["strategies"]["embed"] = EMBED_STRATEGY
        if save is not None:
            save(document if document["schema_version"] == 4 else index)
            writes += 1

    works = list(pending.values())
    work_batches = _batches(works, model, batch_size)
    requests = 0
    failed = 0
    stop = threading.Event()
    if work_batches:
        # A full index can be tens of megabytes, so rewriting it after
        # every batch would dominate a large run. Save periodically for
        # crash resilience and always save before returning or raising.
        last_save = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute_batch, batch, model, dimensions, stop): batch
                for batch in work_batches
            }
            remaining = set(futures)
            interrupted = None
            while remaining:
                try:
                    future = next(as_completed(remaining))
                    remaining.remove(future)
                    try:
                        outcome = future.result()
                    except CancelledError:
                        continue
                    requests += outcome.requests
                    progress = False
                    for work, exc in outcome.errors:
                        print(f"{work.sources[0].id}: {exc}", file=sys.stderr)
                        failed += len(work.sources)
                    for key, vector in outcome.vectors.items():
                        work = pending[key]
                        params = outcome.costs.get(work.key)
                        for index, source in enumerate(work.sources):
                            entry = {
                                **asdict(source),
                                "input_fingerprint": work.key,
                                "vector": vector,
                            }
                            if params is not None:
                                cost, allocation = params
                                if cost is not None:
                                    # Duplicate inputs share one paid request.
                                    entry["cost"] = cost / len(work.sources)
                                    entry["cost_allocation"] = (
                                        "exact"
                                        if allocation == "exact"
                                        and len(work.sources) == 1
                                        else "estimated"
                                    )
                            entries[source.id] = entry
                            if index:
                                reused += 1
                        progress = True
                    if progress and time.monotonic() - last_save >= SAVE_INTERVAL:
                        save_snapshot()
                        last_save = time.monotonic()
                    if outcome.blocked:
                        stop.set()
                        for pending_future in futures:
                            pending_future.cancel()
                except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- drain paid work before reraising
                    if interrupted is None:
                        interrupted = exc
                    stop.set()
                    for pending_future in remaining:
                        pending_future.cancel()
            if interrupted is not None:
                save_snapshot()
                raise interrupted

    complete = len(entries) == len(sources)
    if not complete and not entries and old.get("entries"):
        print("Retained existing embeddings", file=sys.stderr)
        return document, False
    save_snapshot()
    print(
        f"{mode} embeddings ({len(entries)}/{len(sources)} entries, {reused} reused, "
        f"{requests} requests, {failed} failed, {writes} saves, "
        f"workers: {workers})"
    )
    return document, complete



def embed_bundle(
    bundle: Path, mode: str = DEFAULT_MODE, model_override: str | None = None, *,
    workers: int | None = None, batch_size: int | None = None,
) -> bool:
    with document_lock(bundle):
        document = read_document(bundle)
        path = bundle / "document.json" if document["schema_version"] == 4 else _path(bundle, mode)
        writer = JsonWriter(path)
        _, complete = embed_document(
            document, bundle, mode, model_override,
            workers=workers, batch_size=batch_size, save=writer.write,
        )
        return complete

def search_bundle(
    bundle: Path, query: str, mode: str = DEFAULT_MODE, limit: int = 5, *, figures_only: bool = False,
) -> list[dict]:
    query = query.strip()
    if not query or limit < 1:
        raise ValueError("provide a query and a positive result limit")
    document = read_document(bundle)
    blocks = block_document(document)
    resolved = document["schema_version"] == 4
    index = _resolved_index(document, mode) if resolved else read_json(_path(bundle, mode))
    if not index:
        raise FileNotFoundError(f"No {mode} embeddings")
    if index.get("document_fingerprint") != _input_document_fingerprint(blocks):
        raise ValueError("document.json changed. Rebuild embeddings for this mode first")
    if not index["entries"]:
        return []
    if not index["complete"]:
        print("Searching a partial index. Rerun embedding to finish it.", file=sys.stderr)
    by_id = {piece["id"]: piece for piece in blocks["pieces"]}
    parents = {
        block_id: piece for piece in resolved_pieces(document) for block_id in piece["source_ids"]
    } if resolved else {}
    candidates = index["entries"]
    if figures_only:
        candidates = [
            entry for entry in candidates
            if by_id[entry["piece_ids"][0]]["content"]["kind"] == "figure"
        ]
    if not candidates:
        return []
    vector = embed(index["model"], index["dimensions"], query, query=True)
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("query vector has invalid magnitude")
    scored = []
    for item in candidates:
        values = item["vector"]
        if not _valid_vector(values, index["dimensions"]):
            raise ValueError("index contains an invalid vector. Rebuild embeddings for this mode")
        other_norm = math.hypot(*values)
        score = math.fsum((left / norm) * (right / other_norm) for left, right in zip(vector, values))
        hit = {key: value for key, value in item.items() if key != "vector"}
        if parent := parents.get(item["piece_ids"][0]):
            hit["parent_piece"] = {key: value for key, value in parent.items() if key != "embeddings"}
        scored.append({
            **hit, "score": score, "document_id": document["document_id"],
            "pieces": [by_id[piece_id] for piece_id in item["piece_ids"]],
        })
    ranked = sorted(scored, key=lambda hit: hit["score"], reverse=True)
    if figures_only:
        unique = {}
        for hit in ranked:
            unique.setdefault(hit["piece_ids"][0], hit)
        ranked = list(unique.values())
    return ranked[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate embeddings for a parsed PDF bundle.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE, help="embedding mode (default: text)")
    parser.add_argument("--model", choices=(NEMOTRON, f"{NEMOTRON}:free", TITAN, TEXT_MODEL),
                        help="Override the model for the selected mode")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="concurrent embedding batches (default: 4, env BW_EMBED_WORKERS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="text inputs per request (default: 64 for OpenAI, 8 for Nemotron, 1 for Titan)",
    )
    args = parser.parse_args()
    try:
        return 0 if embed_bundle(
            args.bundle, args.mode, args.model,
            workers=args.workers, batch_size=args.batch_size,
        ) else 1
    except (OSError, ValueError, Timeout) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1


def search_main() -> int:
    parser = argparse.ArgumentParser(description="Search a parsed PDF bundle by vector similarity.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE, help="embedding mode (default: text)")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    try:
        hits = search_bundle(args.bundle, args.query, args.mode, args.limit)
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(hits, ensure_ascii=False, indent=2))
    return 0
