"""Link tables to clear source titles and notes in the page reading order."""

import re
from itertools import pairwise

_TABLE_LABEL = re.compile(r"(?<![^\W\d_])(?:table|טבלה|טבלת|(?:ال)?جدول)(?![^\W\d_])", re.IGNORECASE)
_CONTENTS = re.compile(r"\btable of contents\b", re.IGNORECASE)


_NOTE_LABEL = re.compile(r"^\s*(?:notes?|הערה|הערות|ملاحظة|ملاحظات)\s*[:：]\s*\S", re.IGNORECASE)


def _box(piece: dict, page_number: int):
    return next((loc["bbox"] for loc in piece["locations"] if loc["page"] == page_number), None)


def _adjacent_title(heading: dict, table: dict, page_number: int) -> bool:
    if (
        heading["label"] not in {"title", "section_header"}
        or heading["content"]["kind"] != "text"
        or heading["children"]
        or table["content"]["kind"] != "table"
        or table["caption_ids"]
        or table.get("inferred_title_id")
    ):
        return False
    text = heading["content"]["text"]
    if not _TABLE_LABEL.search(text) or _CONTENTS.search(text):
        return False

    title_box, table_box = _box(heading, page_number), _box(table, page_number)
    if title_box is None or table_box is None:
        return False
    left, top, right, bottom = title_box
    table_left, table_top, table_right, _ = table_box
    gap = table_top - bottom
    overlap = min(right, table_right) - max(left, table_left)
    return right > left and 0 <= gap <= min(0.025, 2 * (bottom - top)) and overlap >= 0.8 * (right - left)


def _adjacent_note(table: dict, note: dict, page_number: int) -> bool:
    if (
        table["content"]["kind"] != "table"
        or table["footnote_ids"]
        or table.get("inferred_note_ids")
        or note["label"] not in {"text", "paragraph", "footnote"}
        or note["content"]["kind"] != "text"
        or note["children"]
        or not _NOTE_LABEL.match(note["content"]["text"])
    ):
        return False
    table_box, note_box = _box(table, page_number), _box(note, page_number)
    if table_box is None or note_box is None:
        return False
    table_left, _, table_right, table_bottom = table_box
    left, top, right, _ = note_box
    overlap = min(right, table_right) - max(left, table_left)
    return (
        right > left and 0 <= top - table_bottom <= 0.025
        and abs(left - table_left) <= 0.04 and overlap >= 0.9 * (right - left)
    )


def link_table_context(document: dict) -> int:
    """Record inferred title and note links while retaining the source pieces."""
    pieces = {piece["id"]: piece for piece in document["pieces"]}
    owned = {
        child
        for piece in pieces.values()
        for child in (
            *piece["children"], *piece["caption_ids"], *piece["footnote_ids"],
            *piece.get("inferred_note_ids", []),
        )
    }
    linked = 0
    for page in document["pages"]:
        for first_id, second_id in pairwise(page["piece_ids"]):
            if first_id in owned or second_id in owned:
                continue
            first, second = pieces[first_id], pieces[second_id]
            if _adjacent_title(first, second, page["number"]):
                second["inferred_title_id"] = first_id
                linked += 1
            elif _adjacent_note(first, second, page["number"]):
                first["inferred_note_ids"] = [second_id]
                owned.add(second_id)
                linked += 1
    return linked
