"""Provider-independent structure for a parsed PDF bundle."""

from dataclasses import dataclass, field
from typing import Literal

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Location:
    page: int
    # Left, top, right, bottom as fractions of the displayed PDF page.
    bbox: Box | None = None
    # Optional half-open range within this piece's text.
    text_range: tuple[int, int] | None = None


def location(page: int, box: Box | None, width: float, height: float) -> Location:
    if box is None:
        return Location(page)
    left, top, right, bottom = box
    return Location(page, (left / width, top / height, right / width, bottom / height))


@dataclass(frozen=True)
class ImageRef:
    path: str
    location: Location | None = None


@dataclass(frozen=True)
class Text:
    text: str
    level: int | None = None
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class Table:
    html: str
    kind: Literal["table"] = "table"


@dataclass(frozen=True)
class Figure:
    kind: Literal["figure"] = "figure"


@dataclass(frozen=True)
class Group:
    name: str = ""
    kind: Literal["group"] = "group"


@dataclass(frozen=True)
class Form:
    data: dict | None = None
    kind: Literal["form"] = "form"


@dataclass(frozen=True)
class Other:
    text: str
    kind: Literal["other"] = "other"


@dataclass(frozen=True)
class FigureEnrichment:
    description: str | None
    visible_text: str | None
    model: str
    prompt_version: str
    fingerprint: str
    image_paths: list[str]
    decorative: bool = False


@dataclass(frozen=True)
class Piece:
    id: str
    label: str
    content: Text | Table | Figure | Group | Form | Other
    locations: list[Location] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    caption_ids: list[str] = field(default_factory=list)
    footnote_ids: list[str] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    layer: str | None = None
    enrichment: FigureEnrichment | None = None


@dataclass(frozen=True)
class Page:
    number: int
    width: float
    height: float
    piece_ids: list[str]


@dataclass(frozen=True)
class Parser:
    name: str = "docling"
    version: str = "2.152.0"
    pipeline: str = "standard"
    layout: str = "heron"
    table: str = "tableformer-v1-accurate"
    ocr: str = "rapidocr-3.10.0-pt"


@dataclass(frozen=True)
class Document:
    document_id: str
    parser: Parser
    pages: list[Page]
    pieces: list[Piece]
    schema_version: Literal[2] = 2
