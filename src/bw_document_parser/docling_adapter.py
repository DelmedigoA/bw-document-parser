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
from .document import Figure, Form, Group, ImageRef, Other, Page, Piece, Table, Text, location


def read_page(document: DoclingDocument, number: int, images: ImageFiles) -> tuple[Page, list[Piece]]:
    page = document.pages[number]
    # Per-page Docling conversions reuse item references. Qualify them by page.
    items = dict.fromkeys(ref.cref for ref in document.body.children)
    pieces: dict[str, Piece] = {}
    prefix = f"p{number}:"

    def visit(ref):
        piece_id = prefix + ref
        if piece_id in pieces:
            return piece_id
        item = RefItem(cref=ref).resolve(document)
        positions = []
        for prov in getattr(item, "prov", []):
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

        caption_refs = getattr(item, "captions", [])
        footnote_refs = getattr(item, "footnotes", [])
        children = [child.cref for child in item.children]
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

        pieces[piece_id] = Piece(
            piece_id, item.label.value, content, positions,
            [
                prefix + child for child in children
                if child not in {ref.cref for ref in (*caption_refs, *footnote_refs)}
            ],
            [prefix + caption.cref for caption in caption_refs],
            [prefix + footnote.cref for footnote in footnote_refs],
            assets, item.content_layer.value,
        )
        for related in dict.fromkeys([
            *children, *(ref.cref for ref in caption_refs), *(ref.cref for ref in footnote_refs),
        ]):
            visit(related)
        return piece_id

    for ref in items:
        visit(ref)
    owned = {
        child
        for piece in pieces.values()
        for child in (*piece.children, *piece.caption_ids, *piece.footnote_ids)
    }
    root_ids = [prefix + ref for ref in items if prefix + ref not in owned]
    return Page(number, page.size.width, page.size.height, root_ids), list(pieces.values())
