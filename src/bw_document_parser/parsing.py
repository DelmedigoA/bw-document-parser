"""Docling extraction strategy and source assets."""

import hashlib
from dataclasses import asdict
from pathlib import Path

from .chunks import ImageFiles
from .document import Document, ParsedDoc, Parser

DEFAULT_PAGE_BATCH_SIZE = 12


def _page_count(pdf: Path) -> int:
    import pypdfium2

    document = pypdfium2.PdfDocument(str(pdf))
    try:
        return len(document)
    finally:
        document.close()


def parse_pdf(
    pdf: Path, bundle: Path, *, page_range: tuple[int, int] | None = None,
    page_batch_size: int = DEFAULT_PAGE_BATCH_SIZE,
) -> ParsedDoc:
    """Extract an inclusive range of physical pages with the existing Docling strategy."""
    pdf, bundle = Path(pdf), Path(bundle)
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("input must be a local PDF file")
    if page_batch_size < 1:
        raise ValueError("page batch size must be positive")
    total_pages = _page_count(pdf)
    first, last = page_range if page_range is not None else (1, total_pages)
    if not 1 <= first <= last <= total_pages:
        raise ValueError(f"page range must be within 1–{total_pages}")
    images = ImageFiles(bundle / "images")

    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import (
        LayoutObjectDetectionOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ImageRefMode

    from .docling_adapter import keep_page_boundaries, read_pages

    options = PdfPipelineOptions(
        layout_options=LayoutObjectDetectionOptions.from_preset("layout_heron_default"),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
        do_ocr=True,
        # The former automatic selection passed an empty language list to RapidOCR.
        ocr_options=RapidOcrOptions(backend="torch", lang=[]),
        generate_page_images=True,
        generate_picture_images=True,
        use_embedded_picture_images=True,
        images_scale=2.0,
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )
    converter.initialize_pipeline(InputFormat.PDF)
    # The page-oriented bundle must not carry a paragraph across a page break.
    if not keep_page_boundaries(converter._get_pipeline(InputFormat.PDF)):
        raise RuntimeError(
            "Docling's reading-order model changed, so a batched conversion "
            "cannot keep paragraphs on their source pages. Use the pinned "
            "Docling version."
        )

    def convert_range(start: int, end: int) -> ConversionResult | None:
        result = converter.convert(
            pdf, page_range=(start, end), raises_on_error=False
        )
        expected = set(range(start, end + 1))
        if result.status == ConversionStatus.SUCCESS and expected <= set(
            result.document.pages
        ):
            return result
        if start == end:
            detail = "; ".join(
                error.error_message for error in result.errors if error.error_message
            )
            raise ValueError(
                f"PDF conversion failed on page {start} ({result.status.value})"
                + (f": {detail}" if detail else "")
            )
        return None

    def convert_chunk(start: int, end: int) -> list[ConversionResult]:
        result = convert_range(start, end)
        if result is not None:
            return [result]
        # A document-level failure would otherwise lose every page in the chunk.
        singles = []
        for number in range(start, end + 1):
            single = convert_range(number, number)
            assert single is not None  # A failed single page raises above.
            singles.append(single)
        return singles

    html_dir = bundle / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    pages_by_number, pieces = {}, []
    for start in range(first, last + 1, page_batch_size):
        end = min(last, start + page_batch_size - 1)
        for result in convert_chunk(start, end):
            chunk_pages, chunk_pieces = read_pages(result.document, images)
            for page in chunk_pages:
                html = result.document.export_to_html(
                    page_no=page.number, image_mode=ImageRefMode.EMBEDDED
                )
                (html_dir / f"page-{page.number}.html").write_text(
                    html, encoding="utf-8"
                )
                pages_by_number[page.number] = page
            pieces.extend(chunk_pieces)

    missing = set(range(first, last + 1)) - set(pages_by_number)
    if missing:
        raise ValueError(
            "PDF conversion did not return page "
            + ", ".join(str(number) for number in sorted(missing))
        )
    pages = [pages_by_number[number] for number in range(first, last + 1)]
    with pdf.open("rb") as source:
        document_id = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        **asdict(Document(document_id, Parser(), pages, pieces)),
        "page_count": total_pages,
        "strategies": {"parse": "docling-standard-v1"},
    }
