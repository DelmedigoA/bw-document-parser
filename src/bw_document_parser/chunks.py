"""Turn one page of Docling HTML into ordered text and image pieces."""

import base64
import hashlib
import mimetypes
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import unquote_to_bytes

import tiktoken
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

MIN_TOKENS = 100
TARGET_TOKENS = 400
MAX_TOKENS = 850

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"script", "style", "details", "hr"}
# Internal markers carry page-wide indexes until pieces are packed.
_IMAGE = re.compile(r"\x00(\d+)\x00")


@dataclass(frozen=True)
class Piece:
    text: str
    image_urls: list[str]


@dataclass(frozen=True)
class Page:
    pieces: list[Piece]


@dataclass(frozen=True)
class _Block:
    text: str
    kind: str = "text"


class ImageFiles:
    """Save distinct embedded images and return paths relative to pages.json."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._paths: dict[str, str] = {}

    def save(self, source: str) -> str:
        if not source.lower().startswith("data:"):
            return source

        header, _, payload = source.partition(",")
        mime, *parameters = header[5:].lower().split(";")
        if not payload or not mime.startswith("image/"):
            raise ValueError("invalid embedded image")

        raw = unquote_to_bytes(payload)
        if "base64" in parameters:
            raw = base64.b64decode(raw, validate=True)
        return self.save_bytes(raw, mime)

    def save_bytes(self, raw: bytes, mime: str) -> str:
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in self._paths:
            extension = (
                ".jpg" if mime == "image/jpeg" else mimetypes.guess_extension(mime)
            )
            filename = digest[:24] + (extension or ".bin")
            self.directory.mkdir(exist_ok=True)
            (self.directory / filename).write_bytes(raw)
            self._paths[digest] = f"{self.directory.name}/{filename}"
        return self._paths[digest]


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_encoding().encode_ordinary(_IMAGE.sub("", text)))


def _tidy(text: str) -> str:
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _is_caption(tag: Tag) -> bool:
    return tag.name == "figcaption" or (
        tag.name == "div" and "caption" in tag.get_attribute_list("class")
    )


def _inline(node: object, image_marker: Callable[[str], str]) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node).replace("\x00", "")
    if not isinstance(node, Tag) or node.name in _SKIP_TAGS:
        return ""
    if node.name == "img":
        source = node.get("src")
        return (
            f"\n{image_marker(source)}\n" if isinstance(source, str) and source else ""
        )
    if node.name == "br":
        return "\n"

    text = "".join(_inline(child, image_marker) for child in node.children)
    if _is_caption(node):
        return f"\nCaption: {text}\n" if text.strip() else ""
    return text


def _make_block(tag: Tag, image_marker: Callable[[str], str]) -> _Block:
    if tag.name == "table":
        rows = []
        for row in tag.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if cells:
                rows.append(
                    " | ".join(_tidy(_inline(cell, image_marker)) for cell in cells)
                )
        return _Block("\n".join(rows), "table")

    if tag.name in {"ul", "ol"}:
        items = (
            _tidy(_inline(li, image_marker))
            for li in tag.find_all("li", recursive=False)
        )
        return _Block("\n".join(f"- {item}" for item in items if item), "list")

    text = _tidy(_inline(tag, image_marker))
    if tag.name in _HEADINGS and text:
        return _Block(f"{'#' * int(tag.name[1])} {text}", "heading")
    if _is_caption(tag):
        return _Block(text, "caption")
    return _Block(text, "figure" if tag.name == "figure" else "text")


def _blocks(parent: Tag, image_marker: Callable[[str], str]) -> Iterator[_Block]:
    for node in parent.children:
        if isinstance(node, Tag):
            if node.name in _SKIP_TAGS:
                continue
            if node.name in {"div", "section", "article", "main"} and not _is_caption(
                node
            ):
                yield from _blocks(node, image_marker)
                continue
            block = _make_block(node, image_marker)
        else:
            block = _Block(_tidy(_inline(node, image_marker)))
        if block.text:
            yield block


def _split_tokens(text: str, limit: int) -> Iterator[str]:
    encoding = _encoding()
    tokens = encoding.encode_ordinary(text)
    start = 0
    while start < len(tokens):
        end = min(start + limit, len(tokens))
        # A token boundary can fall inside a UTF-8 character.
        while True:
            try:
                yield encoding.decode(tokens[start:end], errors="strict")
                break
            except UnicodeDecodeError:
                end -= 1
        start = end


def _split_block(block: _Block, limit: int) -> Iterator[_Block]:
    if (
        block.kind in {"figure", "table", "heading"}
        or _IMAGE.search(block.text)
        or _token_count(block.text) <= limit
    ):
        yield block
        return

    current = ""
    segments = re.split(r"(?<=[.!?])(?=\s)|(?<=\n)", block.text)
    for segment in segments:
        if _token_count(current + segment) <= limit:
            current += segment
            continue
        if current.strip():
            yield _Block(current.strip(), block.kind)
        current = segment.lstrip()
        if _token_count(current) > limit:
            for fragment in _split_tokens(current, limit):
                if fragment.strip():
                    yield _Block(fragment.strip(), block.kind)
            current = ""
    if current.strip():
        yield _Block(current.strip(), block.kind)


def _join(blocks: list[_Block]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _pack_section(section: list[_Block]) -> list[list[_Block]]:
    prefix = section[:1] if section[0].kind == "heading" else []
    limit = (
        max(10, MAX_TOKENS - _token_count(_join(prefix)) - 5) if prefix else MAX_TOKENS
    )
    groups: list[list[_Block]] = []
    current = prefix.copy()
    size = _token_count(_join(current))

    for block in section[len(prefix) :]:
        for atom in _split_block(block, limit):
            combined_size = _token_count(_join([*current, atom]))
            if len(current) > len(prefix) and (
                combined_size > MAX_TOKENS
                or (
                    size >= MIN_TOKENS
                    and abs(TARGET_TOKENS - size) < abs(TARGET_TOKENS - combined_size)
                )
            ):
                groups.append(current)
                current = prefix.copy()
                combined_size = _token_count(_join([*current, atom]))
            current.append(atom)
            size = combined_size

    # Merge the last short piece at block level. Its repeated heading and image
    # indexes have yet to be rendered, so neither needs a text rewrite.
    if groups and size < MIN_TOKENS:
        merged = groups[-1] + current[len(prefix) :]
        if _token_count(_join(merged)) <= MAX_TOKENS:
            groups[-1] = merged
            return groups
    return [*groups, current]


def parse_page(html: str, save_image: Callable[[str], str]) -> Page:
    """Parse one HTML export, which must represent a single source page."""
    soup = BeautifulSoup(html, "html.parser")
    image_urls: list[str] = []

    def image_marker(source: str) -> str:
        image_urls.append(save_image(source))
        return f"\x00{len(image_urls) - 1}\x00"

    merged: list[_Block] = []
    for block in _blocks(soup.body or soup, image_marker):
        if block.kind == "caption" and merged and merged[-1].kind == "figure":
            merged[-1] = _Block(f"{merged[-1].text}\n{block.text}", "figure")
        else:
            merged.append(block)

    groups: list[list[_Block]] = []
    section: list[_Block] = []
    for block in merged:
        if section and block.kind == "heading":
            groups.extend(_pack_section(section))
            section = []
        section.append(block)
    if section:
        groups.extend(_pack_section(section))

    pieces = []
    for group in groups:
        piece_urls: list[str] = []

        def local_marker(match: re.Match[str], urls: list[str] = piece_urls) -> str:
            urls.append(image_urls[int(match[1])])
            return f"[image:{len(urls) - 1}]"

        pieces.append(
            Piece(text=_IMAGE.sub(local_marker, _join(group)), image_urls=piece_urls)
        )
    return Page(pieces=pieces)
