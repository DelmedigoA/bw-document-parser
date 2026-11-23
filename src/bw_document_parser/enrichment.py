"""Describe previously parsed figures without reconverting the PDF."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from filelock import Timeout

from . import openrouter
from .bundle import (
    api_image,
    asset_paths,
    document_lock,
    fingerprint,
    linked_text,
    pieces_with_context,
    read_document,
    read_image,
    write_json,
)
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


def describe(context: dict, images: list[tuple[bytes, str]]) -> tuple[str | None, str | None, bool]:
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
        return None, None, True
    description, visible_text = values.get("description"), values.get("visible_text")
    if not isinstance(description, str) or not description.strip():
        raise openrouter.OpenRouterError("Hy4 Flash returned no usable description text")
    if visible_text is not None and not isinstance(visible_text, str):
        raise openrouter.OpenRouterError("Hy4 Flash returned an invalid visible_text field")
    return description.strip(), (visible_text or "").strip() or None, False


def enrich_bundle(bundle: Path) -> bool:
    with document_lock(bundle):
        document = read_document(bundle)
        enriched = omitted = reused = missing = failed = 0
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
                description, visible_text, decorative = describe(
                    context, [api_image(data, 2048) for data in raw],
                )
            except (OSError, ValueError) as exc:
                print(f"{piece['id']}: {exc}", file=sys.stderr)
                failed += 1
                if isinstance(exc, openrouter.OpenRouterError) and exc.blocks_run:
                    break
            else:
                piece["enrichment"] = asdict(FigureEnrichment(
                    description, visible_text, MODEL, FIGURE_DESCRIPTION_VERSION, identity, paths, decorative,
                ))
                write_json(bundle / "document.json", document, indent=2)
                omitted += int(decorative)
                enriched += int(not decorative)
        print(
            f"Figures enriched: {enriched}, decorative: {omitted}, reused: {reused}, "
            f"no image: {missing}, failed: {failed}"
        )
        return failed == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        return 0 if enrich_bundle(args.bundle) else 1
    except (OSError, ValueError, Timeout) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
