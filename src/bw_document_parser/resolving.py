"""Group nearby prose and linked source elements into small pieces."""

import argparse
import re
import sys
from copy import deepcopy
from pathlib import Path

from filelock import FileLock, Timeout

from .bundle import read_json, write_json
from .chunks import _Block, _pack_section
from .document import (
    ParsedDoc,
    ResolvedDoc,
    apply_visual_decision,
    set_visual_enrichment,
    unpack_piece,
    upgrade_resolved,
)
from .table_links import link_table_context

RESOLVE_STRATEGY = "adjacent-content-v1"
_SOURCE_PAGE = re.compile(r"p(\d+):")


def related_ids(block: dict):
    for field in ("children", "caption_ids", "footnote_ids", "inferred_note_ids"):
        yield from block.get(field, ())
    if title := block.get("inferred_title_id"):
        yield title


def resolved_pieces(document: ResolvedDoc):
    for page in document["pages"]:
        yield from page["pieces"]


def _is_heading(block: dict) -> bool:
    return (
        block["label"] in {"title", "section_header"}
        and block.get("layer") in (None, "body")
        and bool(block["content"].get("text", "").strip())
    )


def _is_paragraph(block: dict) -> bool:
    return (
        block["label"] in {"text", "paragraph"}
        and block.get("layer") in (None, "body")
        and bool(block["content"].get("text", "").strip())
    )


def resolve_document(parsed: ParsedDoc) -> ResolvedDoc:
    """Create small pieces from a parse checkpoint or legacy structured bundle."""
    if parsed.get("schema_version") != 2:
        raise ValueError("Resolve needs parsed.json or a version 2 document.json")
    source = deepcopy(parsed)
    link_table_context(source)
    pages = {
        page["number"]: {key: value for key, value in page.items() if key != "piece_ids"}
        for page in source["pages"]
    }
    if not pages:
        raise ValueError("Resolve needs at least one parsed page")
    for page in pages.values():
        page["pieces"] = []
    blocks = {block["id"]: block for block in source["pieces"]}
    owners = {}
    roots = {
        block_id: page["number"]
        for page in source["pages"] for block_id in page["piece_ids"]
    }
    # Captions and notes sometimes appear in both the page roots and the
    # owner's relations. Keep those roots in the same piece as their owner.
    attachment_owner = {
        ref: block["id"]
        for block in blocks.values()
        for ref in (
            *block.get("caption_ids", ()), *block.get("footnote_ids", ()),
            *block.get("inferred_note_ids", ()), block.get("inferred_title_id"),
        )
        if ref in blocks and ref != block["id"]
    }

    def start(block_id, number):
        block = blocks[block_id]
        kind = block["content"]["kind"]
        title = block["content"]["text"].strip() if _is_heading(block) else ""
        piece = {
            "id": f"{'section' if title else 'piece'}:{block_id}",
            "title": title,
            "page": number,
            "blocks": [], "embeddings": [],
        }
        if not title and kind in {"table", "figure"}:
            piece["title"] = kind.capitalize()
        pages[number]["pieces"].append(piece)
        return piece

    def assign(block_id, piece, number):
        if block_id in owners or block_id not in blocks:
            return
        owners[block_id] = piece
        block = blocks[block_id]
        match = _SOURCE_PAGE.match(block_id)
        source_page = roots.get(block_id)
        if source_page is None:
            source_page = int(match[1]) if match else next(
                (loc["page"] for loc in block["locations"]), number
            )
        block["page"] = source_page
        block["root"] = block_id in roots
        for ref in related_ids(block):
            if ref not in roots or attachment_owner.get(ref) == block_id:
                assign(ref, piece, source_page)

    fragments: dict[str, list[str]] = {}

    def assign_prose(run: list[str], number: int) -> None:
        atoms = [
            _Block(blocks[block_id]["content"]["text"], "heading" if _is_heading(blocks[block_id]) else "text",
                   block_id)
            for block_id in run
        ]
        offsets: dict[str, int] = {}
        for group in _pack_section(atoms, repeat_heading=False):
            ids = []
            for atom in group:
                block_id = atom.source_id
                original = blocks[block_id]
                if atom.text == original["content"]["text"]:
                    ids.append(block_id)
                    continue
                text = original["content"]["text"]
                start_at = text.index(atom.text, offsets.get(block_id, 0))
                end_at = start_at + len(atom.text)
                offsets[block_id] = end_at
                parts = fragments.setdefault(block_id, [])
                fragment_id = f"{block_id}:prose-{len(parts) + 1}"
                parts.append(fragment_id)
                locations = []
                for location in original["locations"]:
                    span = location.get("text_range")
                    left, right = (max(start_at, span[0]), min(end_at, span[1])) if span else (start_at, end_at)
                    if left < right:
                        locations.append({**location, "text_range": [left - start_at, right - start_at]})
                blocks[fragment_id] = {
                    **original, "id": fragment_id,
                    "content": {**original["content"], "text": atom.text},
                    "source_id": block_id, "source_text_range": [start_at, end_at],
                    "locations": locations,
                }
                roots[fragment_id] = number
                ids.append(fragment_id)
            if ids:
                piece = start(ids[0], number)
                for block_id in ids:
                    assign(block_id, piece, number)

    for page in source["pages"]:
        number = page["number"]
        run: list[str] = []
        for block_id in page["piece_ids"]:
            block = blocks[block_id]
            if block_id in owners or block_id in attachment_owner:
                if run:
                    assign_prose(run, number)
                    run.clear()
                continue
            if _is_paragraph(block) and not any(related_ids(block)) and not block["images"]:
                run.append(block_id)
                continue
            if run:
                assign_prose(run, number)
                run.clear()
            if _is_heading(block):
                run.append(block_id)
            else:
                assign(block_id, start(block_id, number), number)
        if run:
            assign_prose(run, number)

    reading_order = [
        block_id
        for block in source["pieces"]
        for block_id in fragments.get(block["id"], [block["id"]])
    ]
    # Preserve any unreferenced source blocks as well, including old bundle records.
    for block_id in reading_order:
        block = blocks[block_id]
        if block_id not in owners:
            match = _SOURCE_PAGE.match(block_id)
            number = roots.get(block_id, int(match[1]) if match else next(
                (loc["page"] for loc in block["locations"]), next(iter(pages))
            ))
            number = number if number in pages else next(iter(pages))
            assign(block_id, start(block_id, number), number)
        owners[block_id]["blocks"].append(block)

    return upgrade_resolved({
        "schema_version": 3,
        "document_id": source["document_id"],
        "parser": source["parser"],
        "page_count": source.get("page_count", max(pages)),
        "strategies": {"parse": "docling-standard-v1", **source.get("strategies", {}),
                       "resolve": RESOLVE_STRATEGY},
        "pages": list(pages.values()),
    })


def block_document(document: dict, *, include_excluded: bool = False) -> dict:
    """Give the existing enrichment and search strategies their source-block view."""
    if document["schema_version"] == 2:
        return document
    document = upgrade_resolved(document)
    blocks = [
        block for piece in resolved_pieces(document)
        for block in unpack_piece(piece, document["source_blocks"], include_excluded=include_excluded)
    ]
    roots: dict[int, list[str]] = {}
    for block in blocks:
        if block["root"]:
            roots.setdefault(block["page"], []).append(block["id"])
    return {
        "schema_version": 2,
        "document_id": document["document_id"],
        "parser": document["parser"],
        "pages": [
            {"number": page["number"], "width": page["width"], "height": page["height"],
             "piece_ids": roots.get(page["number"], [])}
            for page in document["pages"]
        ],
        "pieces": blocks,
    }


def _carry_annotations(previous: ResolvedDoc, resolved: ResolvedDoc) -> None:
    """Keep figure decisions, descriptions, and cached vectors across regrouping."""
    if previous["document_id"] != resolved["document_id"]:
        return

    def source_content(block):
        return {key: value for key, value in block.items()
                if key not in {"enrichment", "enrichment_decision", "excluded_visuals"}}

    prior_blocks = {
        block["id"]: block for block in block_document(previous, include_excluded=True)["pieces"]
    }
    owners = {}
    for piece in resolved_pieces(resolved):
        for block in unpack_piece(piece, resolved["source_blocks"], include_excluded=True):
            block_id = block["id"]
            prior = prior_blocks.get(block_id)
            if prior is None or source_content(prior) != source_content(block):
                continue
            owners[block_id] = piece
            if prior["enrichment"]:
                set_visual_enrichment(
                    piece["visual_part"] or [], resolved["source_blocks"], block_id, prior["enrichment"],
                )
            if decision := prior.get("enrichment_decision"):
                apply_visual_decision(piece, resolved["source_blocks"][block_id], decision)

    # Embed validates the saved input fingerprint before reusing a vector.
    # Search validates the mode's document fingerprint before serving results.
    for piece in resolved_pieces(previous):
        for embedding in piece["embeddings"]:
            if all(block_id in owners for block_id in embedding["piece_ids"]):
                owners[embedding["piece_ids"][0]]["embeddings"].append(embedding)
    resolved["strategies"].update({
        key: value for key, value in previous["strategies"].items()
        if key in {"decide", "enrich", "embed"}
    })
    if "embedding_modes" in previous:
        resolved["embedding_modes"] = previous["embedding_modes"]
    if "decider_requests" in previous:
        # Keep the provider's actual per-request charges as an audit ledger.
        resolved["decider_requests"] = deepcopy(previous["decider_requests"])


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bundle", type=Path, help="bundle containing parsed.json or a legacy document.json")
    args = cli.parse_args()
    bundle = args.bundle
    parsed_path = bundle / "parsed.json"
    canonical = bundle / "document.json"
    try:
        with FileLock(bundle / ".bw-document.lock", timeout=0):
            previous = upgrade_resolved(read_json(canonical)) if canonical.is_file() else None
            parsed = read_json(parsed_path) if parsed_path.is_file() else previous
            if parsed is None:
                raise ValueError("bundle has no parsed.json or legacy document.json")
            resolved = resolve_document(parsed)
            if previous and previous["schema_version"] == 4:
                _carry_annotations(previous, resolved)
            if not parsed_path.is_file():
                write_json(parsed_path, parsed, indent=2)
            write_json(canonical, resolved)
    except (OSError, ValueError, Timeout) as exc:
        print(f"{cli.prog}: {exc}", file=sys.stderr)
        return 1
    print(canonical)
    return 0
