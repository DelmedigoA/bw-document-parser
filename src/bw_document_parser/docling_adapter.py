"""Read typed items and provenance directly from Docling."""

from dataclasses import replace
from io import BytesIO

from docling_core.types.doc import (
    DoclingDocument,
    FieldItem,
    FieldRegionItem,
    FormItem,
    GroupItem,
    KeyValueItem,
    PictureItem,
    RefItem,
    TableItem,
    TextItem,
)

from .chunks import ImageFiles
from .document import (
    Figure,
    Form,
    Group,
    ImageRef,
    Other,
    Page,
    Piece,
    Table,
    Text,
    location,
)


def _read_document(
    document: DoclingDocument, images: ImageFiles
) -> tuple[list[Page], list[Piece], dict[str, int]]:
    """Walk a Docling document once and assign every item to its page.

    The first provenance page owns an item.  Group items follow the first
    page of their content, because Docling keeps one group for a list that
    continues across pages, while a per-page conversion creates a page-local
    group.  Piece IDs are renumbered per page and item collection so the
    bundle keeps the ``p<page>:<ref>`` values produced by per-page parsing.
    """
    page_numbers = sorted(document.pages)
    if not page_numbers:
        return [], [], {}
    pages = {
        number: Page(
            number,
            document.pages[number].size.width,
            document.pages[number].size.height,
            [],
        )
        for number in page_numbers
    }
    records: dict[str, dict] = {}
    order: list[str] = []
    seen: set[str] = set()

    def visit(ref: str, inherited_page: int | None) -> None:
        if ref in seen:
            return
        seen.add(ref)
        item = RefItem(cref=ref).resolve(document)
        provenance = list(getattr(item, "prov", []))
        own_page = provenance[0].page_no if provenance else None

        positions = []
        for prov in provenance:
            size = document.pages[prov.page_no].size
            bbox = prov.bbox.to_top_left_origin(size.height).as_tuple()
            position = location(prov.page_no, bbox, size.width, size.height)
            if isinstance(item, TextItem):
                position = replace(position, text_range=tuple(prov.charspan))
            positions.append(position)

        assets = []
        if isinstance(item, (PictureItem, TableItem)):
            uri = str(item.image.uri) if item.image is not None else ""
            if uri.startswith("data:"):
                path = images.save(uri)
                assets.extend(ImageRef(path, position) for position in positions or [None])
            elif item.image is None:
                for index, position in enumerate(positions):
                    image = item.get_image(document, prov_index=index)
                    if image is not None:
                        with image, BytesIO() as buffer:
                            image.save(buffer, format="PNG")
                            path = images.save_bytes(buffer.getvalue(), "image/png")
                        assets.append(ImageRef(path, position))

        caption_refs = [caption.cref for caption in getattr(item, "captions", [])]
        footnote_refs = [footnote.cref for footnote in getattr(item, "footnotes", [])]
        excluded = {*caption_refs, *footnote_refs}
        raw_children = [child.cref for child in item.children]
        children = [child for child in raw_children if child not in excluded]
        if isinstance(item, TableItem):
            content = Table(item.export_to_html(document, add_caption=False))
        elif isinstance(item, PictureItem):
            content = Figure()
        elif isinstance(item, TextItem):
            content = Text(item.text, getattr(item, "level", None))
        elif isinstance(item, GroupItem):
            content = Group(item.name)
        elif isinstance(item, (FieldItem, FieldRegionItem, FormItem, KeyValueItem)):
            graph = getattr(item, "graph", None)
            content = Form(graph.model_dump(mode="json") if graph is not None else None)
        else:
            content = Other(getattr(item, "text", ""))

        records[ref] = {
            "own_page": own_page,
            "inherited_page": inherited_page,
            "page": own_page or inherited_page or page_numbers[0],
            "label": item.label.value,
            "is_group": isinstance(item, GroupItem),
            "content": content,
            "locations": positions,
            "children": children,
            "caption_ids": caption_refs,
            "footnote_ids": footnote_refs,
            "images": assets,
            "layer": item.content_layer.value,
        }
        order.append(ref)
        # Follow Docling's original child order, including captions, so the
        # flat piece order matches a per-page conversion.
        for related in dict.fromkeys([*raw_children, *caption_refs, *footnote_refs]):
            visit(related, own_page or inherited_page or page_numbers[0])

    for ref in document.body.children:
        visit(ref.cref, None)

    def first_descendant_page(ref: str) -> int | None:
        """The page where a group's content actually starts.

        Docling keeps one list group for a list that continues over several
        pages and gives that group the page where the list began.  A
        per-page conversion instead creates one page-local group, so the
        bundle's group pieces must follow the content, not the group record.
        """
        for child in records[ref]["children"]:
            if child not in records:
                continue
            record = records[child]
            page = record["own_page"] or first_descendant_page(child)
            if page is not None:
                return page
        return None

    for ref, record in records.items():
        if record["is_group"]:
            record["page"] = (
                first_descendant_page(ref)
                or record["own_page"]
                or record["inherited_page"]
                or page_numbers[0]
            )

    # A batch conversion numbers every item in one namespace, while a
    # per-page conversion starts each item collection at zero on each page.
    # Renumber per page and collection so piece IDs stay stable across both
    # conversion strategies.
    collections: dict[tuple[int, str], list[str]] = {}
    for ref, record in records.items():
        collection = ref.split("/")[1]
        collections.setdefault((record["page"], collection), []).append(ref)
    normalized: dict[str, str] = {}
    for (_, collection), refs in collections.items():
        refs.sort(key=lambda ref: int(ref.rsplit("/", 1)[1]))
        for index, ref in enumerate(refs):
            normalized[ref] = f"#/{collection}/{index}"

    piece_ids = {
        ref: f"p{record['page']}:{normalized[ref]}"
        for ref, record in records.items()
    }
    pieces = [
        Piece(
            piece_ids[ref],
            record["label"],
            record["content"],
            record["locations"],
            [piece_ids[child] for child in record["children"] if child in piece_ids],
            [
                piece_ids[caption]
                for caption in record["caption_ids"]
                if caption in piece_ids
            ],
            [
                piece_ids[footnote]
                for footnote in record["footnote_ids"]
                if footnote in piece_ids
            ],
            record["images"],
            record["layer"],
        )
        for ref, record in ((ref, records[ref]) for ref in order)
    ]

    owned = {
        related
        for record in records.values()
        for related in (
            *record["children"],
            *record["caption_ids"],
            *record["footnote_ids"],
        )
    }
    for ref in document.body.children:
        cref = ref.cref
        if cref in owned or cref not in records:
            continue
        pages[records[cref]["page"]].piece_ids.append(piece_ids[cref])

    page_of_piece = {
        piece_ids[ref]: record["page"] for ref, record in records.items()
    }
    return [pages[number] for number in page_numbers], pieces, page_of_piece


def read_pages(
    document: DoclingDocument, images: ImageFiles
) -> tuple[list[Page], list[Piece]]:
    """Read every page of one Docling conversion in reading order."""
    pages, pieces, _ = _read_document(document, images)
    return pages, pieces


def read_page(
    document: DoclingDocument, number: int, images: ImageFiles
) -> tuple[Page, list[Piece]]:
    pages, pieces, page_of_piece = _read_document(document, images)
    page = next(
        (candidate for candidate in pages if candidate.number == number),
        Page(
            number,
            document.pages[number].size.width,
            document.pages[number].size.height,
            [],
        ),
    )
    return page, [piece for piece in pieces if page_of_piece[piece.id] == number]


def keep_page_boundaries(pipeline) -> bool:
    """Stop Docling from merging a paragraph across a source page boundary.

    Docling's reading-order model joins a paragraph on one page with a
    continuing paragraph on the next.  The page-oriented bundle needs those
    paragraphs to stay on their own pages.  The pipeline caches its reading
    order model, so patch that instance once per converter.
    """
    predictor = getattr(
        getattr(getattr(pipeline, "reading_order_model", None), "ro_model", None),
        "predict_merges",
        None,
    )
    if predictor is None:
        return False
    if getattr(predictor, "__bw_page_boundaries__", False):
        return True

    def predict_same_page_merges(sorted_elements):
        merges = predictor(sorted_elements=sorted_elements)
        page_by_cid = {element.cid: element.page_no for element in sorted_elements}
        kept = {}
        for source, targets in merges.items():
            same_page = [
                cid
                for cid in targets
                if page_by_cid.get(cid) == page_by_cid.get(source)
            ]
            if same_page:
                kept[source] = same_page
        return kept

    predict_same_page_merges.__bw_page_boundaries__ = True
    pipeline.reading_order_model.ro_model.predict_merges = predict_same_page_merges
    return True
