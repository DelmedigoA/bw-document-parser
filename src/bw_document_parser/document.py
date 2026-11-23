"""Provider-independent structure for a parsed PDF bundle."""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

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
    # OpenRouter reports one cost for this request; a figure is one request,
    # so the figure's first location attributes the cost exactly.
    cost: float | None = None


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


# These aliases describe successive JSON-compatible snapshots. ResolvedDoc
# owns structure; the last two stages attach data to its blocks and pieces.
ParsedDoc: TypeAlias = dict
ResolvedDoc: TypeAlias = dict
EnrichedDoc: TypeAlias = ResolvedDoc
EmbeddedDoc: TypeAlias = EnrichedDoc


def pack_piece(piece: dict, source_blocks: dict) -> dict:
    """Store readable content and visuals on the piece, with source details indexed by ID."""
    blocks = piece["blocks"]
    owned_text = {
        ref for block in blocks
        for ref in (*block.get("caption_ids", ()), *block.get("footnote_ids", ()),
                    *block.get("inferred_note_ids", ()), block.get("inferred_title_id"))
        if ref
    }
    figure_text = {
        ref for block in blocks if block["content"]["kind"] == "figure"
        for ref in block["children"]
    }
    text = "\n\n".join(
        block["content"]["text"].strip()
        for block in blocks
        if (block["id"] not in figure_text or block["id"] in owned_text)
        and block["content"].get("text", "").strip()
    )
    tables = [
        {"source_id": block["id"], "html": block["content"]["html"]}
        for block in blocks if block["content"]["kind"] == "table"
    ]
    visuals = []
    for block in blocks:
        origin = {key: value for key, value in block.items() if key not in {"images", "enrichment"}}
        if block["content"]["kind"] == "table":
            origin["content"] = {"kind": "table"}
        source_blocks[block["id"]] = origin
        for image in block["images"]:
            visuals.append({"url": image["path"], "source_id": block["id"],
                            "location": image.get("location"), "description": None, "transcription": None})
        if block.get("enrichment"):
            set_visual_enrichment(visuals, source_blocks, block["id"], block["enrichment"])
    return {
        **{key: value for key, value in piece.items() if key != "blocks"},
        "textual_part": {"text": text, "tables": tables} if text or tables else None,
        "visual_part": visuals or None,
        "source_ids": [block["id"] for block in blocks],
    }


def set_visual_enrichment(visuals: list, source_blocks: dict, source_id: str, enrichment: dict) -> None:
    """Keep shared model metadata with the source and the generated text with its images."""
    source_blocks[source_id]["enrichment"] = {
        key: value for key, value in enrichment.items() if key not in {"description", "visible_text"}
    }
    for visual in visuals:
        if visual["source_id"] == source_id:
            visual["description"] = enrichment.get("description")
            visual["transcription"] = enrichment.get("visible_text")


def apply_visual_decision(piece: dict, source: dict, decision: dict) -> None:
    """Move rejected images to provenance so a future decision can reconsider them."""
    source["enrichment_decision"] = decision
    source_id = source["id"]
    visuals = piece["visual_part"] or []
    if decision["enrich"]:
        if excluded := source.pop("excluded_visuals", []):
            order = {block_id: i for i, block_id in enumerate(piece["source_ids"])}
            piece["visual_part"] = sorted(
                [*visuals, *excluded], key=lambda image: order[image["source_id"]]
            )
    else:
        excluded = [image for image in visuals if image["source_id"] == source_id]
        if excluded:
            source["excluded_visuals"] = excluded
            piece["visual_part"] = [
                image for image in visuals if image["source_id"] != source_id
            ] or None


def unpack_piece(
    piece: dict, source_blocks: dict, *, include_excluded: bool = False,
) -> list[dict]:
    """Rebuild the source-block view used by the existing parser consumers."""
    images_by_block: dict[str, list] = {}
    for image in piece["visual_part"] or ():
        images_by_block.setdefault(image["source_id"], []).append(image)
    tables = {
        table["source_id"]: table["html"] for table in (piece["textual_part"] or {}).get("tables", ())
    }
    blocks = []
    for source_id in piece["source_ids"]:
        source = source_blocks[source_id]
        visuals = images_by_block.get(source_id, [])
        if include_excluded and not visuals:
            visuals = source.get("excluded_visuals", [])
        block = {**source, "images": [
            {"path": image["url"], "location": image["location"]} for image in visuals
        ], "enrichment": None}
        if source_id in tables:
            block["content"] = {**source["content"], "html": tables[source_id]}
        if source.get("enrichment"):
            visual = visuals[0] if visuals else {}
            block["enrichment"] = {**source["enrichment"],
                                   "description": visual.get("description"),
                                   "visible_text": visual.get("transcription")}
        blocks.append(block)
    return blocks


def upgrade_resolved(document: dict) -> dict:
    """Read resolved bundles from the original section schema as pieces with parts."""
    if document.get("schema_version") != 3:
        return document
    source_blocks: dict = {}
    return {
        **document,
        "schema_version": 4,
        "pages": [
            {**page, "pieces": [pack_piece(piece, source_blocks) for piece in page["pieces"]]}
            for page in document["pages"]
        ],
        "source_blocks": source_blocks,
    }
