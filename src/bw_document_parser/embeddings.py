"""Generate portable embeddings and search a single PDF bundle."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from filelock import Timeout

from . import openrouter
from .bundle import api_image, document_lock, fingerprint, read_document, read_image, read_json, write_json
from .chunks import MAX_TOKENS, _token_count
from .search_entries import search_entries

NEMOTRON = "nvidia/nemotron-3.5-embed-8b"
TITAN = "amazon/titan-embed-image-v1"
# Leave headroom under OpenRouter's 128-token limit for tokenizer differences.
TITAN_CHUNK_TOKENS = 96
TEXT_MODEL = "openai/text-embedding-3-small"
MODES = {"multimodal": (f"{NEMOTRON}:free", 4096), "text": (TEXT_MODEL, 1536)}


def _valid_vector(vector, dimensions: int) -> bool:
    return (
        isinstance(vector, list)
        and len(vector) == dimensions
        and all(type(value) in (int, float) and math.isfinite(value) for value in vector)
        and any(vector)
    )


def embed(
    model: str, dimensions: int, text: str, image: tuple[bytes, str] | None = None, *, query: bool = False,
) -> list[float]:
    if model == TITAN and _token_count(text) > TITAN_CHUNK_TOKENS:
        raise ValueError("Titan text input is too long. Use a shorter query or chunk.")
    value = text
    if image is not None:
        content = [{"type": "text", "text": text}] if text else []
        content.append(openrouter.image_part(*image))
        value = [{"content": content}]
    payload = {"model": model, "input": value, "encoding_format": "float"}
    if model in {TEXT_MODEL, TITAN}:
        payload["dimensions"] = dimensions
    else:
        payload["input_type"] = "query" if query else "document"
    response = openrouter.post("embeddings", payload)
    try:
        item, = response["data"]
        vector = item["embedding"]
        if item["index"] != 0 or not _valid_vector(vector, dimensions):
            raise ValueError("invalid vector")
    except (KeyError, TypeError, ValueError, OverflowError):
        raise openrouter.OpenRouterError(
            f"OpenRouter returned an invalid {dimensions}-dimensional embedding"
        ) from None
    return vector


def _path(bundle: Path, mode: str) -> Path:
    return bundle / f"embeddings.{mode}.json"


def embed_bundle(bundle: Path, mode: str, model_override: str | None = None) -> bool:
    default_model, dimensions = MODES[mode]
    model = model_override or default_model
    if model != default_model and (mode != "multimodal" or model not in {NEMOTRON, TITAN}):
        raise ValueError(f"{model} is unsupported in {mode} mode")
    if model == TITAN:
        dimensions = 1024
    with document_lock(bundle):
        document = read_document(bundle)
        path = _path(bundle, mode)
        old = read_json(path) if path.is_file() else {}
        cached = {}
        if old.get("model") == model and old.get("dimensions") == dimensions:
            cached = {
                item["input_fingerprint"]: item["vector"] for item in old["entries"]
                if _valid_vector(item.get("vector"), dimensions)
            }
        entries = {}
        pending = []
        sources = list(search_entries(
            document, multimodal=mode == "multimodal",
            max_tokens=TITAN_CHUNK_TOKENS if model == TITAN else MAX_TOKENS,
        ))
        for source in sources:
            try:
                raw = read_image(bundle, source.image_path) if source.image_path else None
                image_hash = hashlib.sha256(raw).hexdigest() if raw is not None else None
                identity = fingerprint(model, dimensions, "document-v1", source.text, image_hash)
                if identity in cached:
                    entries[source.id] = {**asdict(source), "input_fingerprint": identity, "vector": cached[identity]}
                else:
                    pending.append((source, identity))
            except (OSError, ValueError) as exc:
                print(f"{source.id}: {exc}", file=sys.stderr)

        reused = len(entries)
        for source, identity in pending:
            try:
                if identity in cached:
                    vector = cached[identity]
                    reused += 1
                else:
                    image = api_image(read_image(bundle, source.image_path), 1540) if source.image_path else None
                    vector = cached[identity] = embed(model, dimensions, source.text, image)
                entries[source.id] = {**asdict(source), "input_fingerprint": identity, "vector": vector}
            except (OSError, ValueError) as exc:
                print(f"{source.id}: {exc}", file=sys.stderr)
                if isinstance(exc, openrouter.OpenRouterError) and exc.blocks_run:
                    break
        complete = len(entries) == len(sources)
        if not complete and not entries and old.get("entries"):
            print(f"Retained the existing {path.name}", file=sys.stderr)
            return False
        write_json(path, {
            "schema_version": 1,
            "document_id": document["document_id"],
            "document_fingerprint": fingerprint(document),
            "model": model,
            "dimensions": dimensions,
            "mode": mode,
            "complete": complete,
            "expected_entries": len(sources),
            "entries": [entries[source.id] for source in sources if source.id in entries],
        })
        print(f"{path} ({len(entries)}/{len(sources)} entries, {reused} reused)")
        return complete


def search_bundle(
    bundle: Path, query: str, mode: str, limit: int = 5, *, figures_only: bool = False,
) -> list[dict]:
    query = query.strip()
    if not query or limit < 1:
        raise ValueError("provide a query and a positive result limit")
    document = read_document(bundle)
    index = read_json(_path(bundle, mode))
    if index.get("document_fingerprint") != fingerprint(document):
        raise ValueError("document.json changed. Rebuild embeddings for this mode first")
    if not index["entries"]:
        return []
    if not index["complete"]:
        print("Searching a partial index. Rerun embedding to finish it.", file=sys.stderr)
    by_id = {piece["id"]: piece for piece in document["pieces"]}
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
    parser.add_argument("--mode", choices=MODES, default="multimodal")
    parser.add_argument("--model", choices=(NEMOTRON, f"{NEMOTRON}:free", TITAN, TEXT_MODEL),
                        help="Override the model for the selected mode")
    args = parser.parse_args()
    try:
        return 0 if embed_bundle(args.bundle, args.mode, args.model) else 1
    except (OSError, ValueError, Timeout) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1


def search_main() -> int:
    parser = argparse.ArgumentParser(description="Search a parsed PDF bundle by vector similarity.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="multimodal")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    try:
        hits = search_bundle(args.bundle, args.query, args.mode, args.limit)
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(hits, ensure_ascii=False, indent=2))
    return 0
