"""Build embed-ready text and source references from structured pieces."""

from collections.abc import Iterator
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .bundle import asset_paths, linked_text, pieces_with_context, text_of
from .chunks import MAX_TOKENS, _split_tokens, _token_count


@dataclass(frozen=True)
class SearchEntry:
    id: str
    piece_ids: list[str]
    locations: list[dict]
    text: str
    image_path: str | None = None
    rows: list[int] | None = None


def _span(cell, field: str) -> int:
    try:
        return max(1, min(1000, int(cell.get(field, 1))))
    except (ValueError, TypeError):
        return 1


def table_rows(html: str) -> tuple[list[str], list[tuple[int, str]]]:
    """Render rows with per-column header paths, including merged headings."""
    table = BeautifulSoup(html, "html.parser").find("table")
    if table is None:
        return [], []
    headers: dict[int, list[str]] = {}
    rows = []
    occupied: dict[int, tuple[int, str, int, bool]] = {}
    leading_headers = True
    for row_number, row in enumerate(
        (row for row in table.find_all("tr") if row.find_parent("table") is table), start=1,
    ):
        cells = row.find_all(["th", "td"], recursive=False)
        is_header = bool(row.find_parent("thead")) or (
            bool(cells) and leading_headers and all(cell.name == "th" for cell in cells)
        )
        leading_headers = leading_headers and is_header
        occupied = {column: cell for column, cell in occupied.items() if row_number < cell[0]}
        column = 0
        for cell in cells:
            colspan = _span(cell, "colspan")
            while any(current in occupied for current in range(column, column + colspan)):
                column += 1
            value = " ".join(cell.stripped_strings)
            rowspan = _span(cell, "rowspan")
            for current in range(column, column + colspan):
                if is_header:
                    path = headers.setdefault(current, [])
                    if value and (not path or path[-1] != value):
                        path.append(value)
                occupied[current] = row_number + rowspan, value, column, is_header
            column += colspan

        if not is_header:
            fields = []
            for current, (_, value, start, header) in sorted(occupied.items()):
                if header:
                    continue
                label = " → ".join(headers.get(current, []))
                if label or current == start:
                    fields.append(f"{label}: {value}" if label else value)
            if fields:
                rows.append((row_number, " | ".join(fields)))
    return list(dict.fromkeys(" → ".join(path) for path in headers.values() if path)), rows


def _table_chunks(html: str, context: str, footnotes: list[str], max_tokens: int) -> Iterator[tuple[str, list[int]]]:
    headers, rows = table_rows(html)
    headings = f"Columns: {' | '.join(headers)}" if headers else ""
    prefix = "\n".join(filter(None, (context, headings, *footnotes)))
    # An unusually long heading must leave some room for the row text.
    prefix = next(_split_tokens(prefix, min(200, max_tokens // 3)), "")
    budget = max_tokens - _token_count(prefix) - 8
    if not rows and headings:
        yield prefix, []
        return
    selected = []
    for row, text in rows:
        for fragment in _split_tokens(text, budget):
            candidate = "\n".join([*(content for _, content in selected), fragment])
            if selected and _token_count(candidate) > budget:
                yield _render_rows(prefix, selected)
                selected = []
            selected.append((row, fragment))
    if selected:
        yield _render_rows(prefix, selected)


def _render_rows(prefix: str, rows: list[tuple[int, str]]) -> tuple[str, list[int]]:
    return (
        "\n".join(filter(None, (prefix, *(text for _, text in rows)))),
        list(dict.fromkeys(row for row, _ in rows)),
    )


def search_entries(document: dict, *, multimodal: bool, max_tokens: int = MAX_TOKENS) -> Iterator[SearchEntry]:
    figure_by_child = {
        child: piece
        for piece in document["pieces"] if piece["content"]["kind"] == "figure"
        for child in piece["children"]
    }
    for piece, headings, by_id in pieces_with_context(document, include_decorative=False):
        piece_id = piece["id"]
        kind = piece["content"]["kind"]
        parent = figure_by_child.get(piece_id)
        piece_ids = [piece_id] if parent is None else [piece_id, parent["id"]]
        locations = piece["locations"] or (parent["locations"] if parent else [])
        images = asset_paths(piece) if multimodal else []
        captions = linked_text(piece, by_id, "caption_ids")
        if parent and not captions:
            captions = linked_text(parent, by_id, "caption_ids")
        context = "\n".join(filter(None, (" / ".join(headings), *captions)))
        if kind == "table":
            footnotes = linked_text(piece, by_id, "footnote_ids")
            table_overview = next(_split_tokens(context, max_tokens), "")
            for index, (text, rows) in enumerate(_table_chunks(piece["content"]["html"], context, footnotes, max_tokens)):
                if index == 0:
                    table_overview = text
                yield SearchEntry(f"{piece_id}:rows-{index}", piece_ids, locations, text, rows=rows)
            for index, image in enumerate(images):
                yield SearchEntry(f"{piece_id}:image-{index}", piece_ids, locations, table_overview, image)
        elif kind == "figure":
            enrichment = piece.get("enrichment") or {}
            recovered = [text_of(by_id[ref]) for ref in piece["children"] if ref in by_id]
            values = [
                context, *recovered, enrichment.get("description"), enrichment.get("visible_text"),
            ]
            text = "\n".join(dict.fromkeys(filter(None, values)))
            fragments = iter(_split_tokens(text, max_tokens))
            first = next(fragments, "")
            for index, image in enumerate(images or [None]):
                if first or image:
                    yield SearchEntry(f"{piece_id}:figure-{index}", piece_ids, locations, first, image)
            for index, fragment in enumerate(fragments, start=1):
                yield SearchEntry(f"{piece_id}:figure-text-{index}", piece_ids, locations, fragment)
        elif kind in {"text", "other"} and piece["layer"] in {None, "body"} and text_of(piece):
            text = text_of(piece)
            if headings and headings[-1] == text:
                headings = headings[:-1]
            prefix = next(_split_tokens(" / ".join(headings), min(200, max_tokens // 3)), "")
            for index, fragment in enumerate(_split_tokens(text, max_tokens - _token_count(prefix) - 8)):
                value = "\n".join(filter(None, (prefix, fragment)))
                yield SearchEntry(f"{piece_id}:text-{index}", piece_ids, locations, value)
