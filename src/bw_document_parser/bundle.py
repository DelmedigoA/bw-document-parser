"""Shared access to a parsed document and its local assets."""

import hashlib
import json
import re
import threading
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock
from PIL import Image

from .document import upgrade_resolved


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        # ValueError keeps the message usable through the CLI error handler.
        raise ValueError(f"{path.name} must contain a JSON object")  # noqa: TRY004
    return data


def write_json(path: Path, data: dict, *, indent: int | None = None) -> None:
    # The stream must stay open through json.dump before the atomic replace.
    stream = NamedTemporaryFile(  # noqa: SIM115
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(data, stream, ensure_ascii=False, indent=indent, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class JsonWriter:
    """The only writer for one bundle file.

    Hosted stages submit results from several worker threads but serialize
    every snapshot through one lock. That keeps a partial run resumable
    without letting two writers replace the same file at once.
    """

    def __init__(self, path: Path, *, indent: int | None = None) -> None:
        self.path = path
        self.indent = indent
        self._lock = threading.Lock()
        self.writes = 0

    def write(self, data) -> None:
        with self._lock:
            write_json(self.path, data, indent=self.indent)
            self.writes += 1


def fingerprint(*values) -> str:
    data = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def document_lock(bundle: Path) -> FileLock:
    if not (bundle / "document.json").is_file():
        raise ValueError("bundle has no document.json. Parse the PDF first.")
    return FileLock(bundle / ".bw-document.lock", timeout=0)


def read_document(bundle: Path) -> dict:
    document = read_json(bundle / "document.json")
    if document.get("schema_version") not in {2, 3, 4}:
        raise ValueError("unsupported document.json schema")
    return upgrade_resolved(document)


def text_of(piece: dict) -> str:
    return piece["content"].get("text", "").strip()


def pieces_with_context(document: dict, *, include_decorative: bool = True) -> Iterator[tuple[dict, tuple[str, ...], dict]]:
    """Walk page roots and children, carrying section headings across pages."""
    pieces = {piece["id"]: piece for piece in document["pieces"]}
    seen = set()

    def walk(ids, headings):
        for piece_id in ids:
            if piece_id in seen:
                continue
            seen.add(piece_id)
            piece = pieces[piece_id]
            if not include_decorative and (piece.get("enrichment") or {}).get("decorative") is True:
                continue
            label = piece["label"]
            if label in {"title", "section_header"} and text_of(piece):
                level = 0 if label == "title" else piece["content"].get("level") or 1
                headings[:] = [(rank, text) for rank, text in headings if rank < level]
                headings.append((level, text_of(piece)))
            yield piece, tuple(text for _, text in headings), pieces
            nested = headings if piece["content"]["kind"] == "group" else headings.copy()
            yield from walk(piece["children"], nested)

    headings = []
    for page in document["pages"]:
        yield from walk(page["piece_ids"], headings)


def linked_text(piece: dict, pieces: dict, field: str) -> list[str]:
    return [text for ref in piece[field] if (text := text_of(pieces[ref]))]


def asset_paths(piece: dict) -> list[str]:
    return list(dict.fromkeys(image["path"] for image in piece["images"]))


def read_image(bundle: Path, path: str) -> bytes:
    if not re.fullmatch(r"images/[0-9a-f]{24}\.[a-zA-Z0-9]+", path):
        raise ValueError(f"invalid bundle image path: {path!r}")
    file = bundle / path
    if not file.resolve().is_relative_to(bundle.resolve()):
        raise ValueError(f"bundle image is outside the bundle: {path!r}")
    return file.read_bytes()


def api_image(data: bytes, max_edge: int) -> tuple[bytes, str]:
    """Use an API-compatible image format while preserving the bundle asset."""
    with Image.open(BytesIO(data)) as image:
        if image.format in {"PNG", "JPEG"} and max(image.size) <= max_edge:
            return data, f"image/{image.format.lower()}"
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        with BytesIO() as encoded:
            image.save(encoded, format="PNG")
            return encoded.getvalue(), "image/png"
