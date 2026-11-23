"""Convert a local PDF and write chunks for each physical page."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .chunks import ImageFiles, parse_page
from .document import Document, Parser

DEFAULT_PAGE_BATCH_SIZE = 12


def _write_bundle(pdf: Path, output_dir: Path, page_batch_size: int) -> None:
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("input must be a local PDF file")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so a failed extraction leaves no output bundle.
    with TemporaryDirectory(dir=output_dir.parent, prefix=".bw-document-parser-") as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        images = ImageFiles(bundle / "images")
        document, pages = _parse_docling(
            pdf, bundle, images, page_batch_size=page_batch_size
        )
        for name, data in (("document.json", asdict(document)), ("pages.json", [asdict(page) for page in pages])):
            (bundle / name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        bundle.rename(output_dir)


def _page_count(pdf: Path) -> int:
    import pypdfium2

    document = pypdfium2.PdfDocument(str(pdf))
    try:
        return len(document)
    finally:
        document.close()


def _parse_docling(
    pdf: Path, bundle: Path, images: ImageFiles, *, page_batch_size: int = DEFAULT_PAGE_BATCH_SIZE
):
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

    def convert_range(start: int, end: int) -> ConversionResult:
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
        return [convert_range(number, number) for number in range(start, end + 1)]

    total_pages = _page_count(pdf)
    if total_pages < 1:
        raise ValueError("PDF has no pages")
    page_batch_size = max(1, page_batch_size)
    html_dir = bundle / "html"
    html_dir.mkdir()
    pages_by_number, pieces = {}, []
    chunks_by_number = {}
    for start in range(1, total_pages + 1, page_batch_size):
        end = min(total_pages, start + page_batch_size - 1)
        for result in convert_chunk(start, end):
            chunk_pages, chunk_pieces = read_pages(result.document, images)
            for page in chunk_pages:
                html = result.document.export_to_html(
                    page_no=page.number, image_mode=ImageRefMode.EMBEDDED
                )
                (html_dir / f"page-{page.number}.html").write_text(
                    html, encoding="utf-8"
                )
                chunks_by_number[page.number] = parse_page(html, images.save)
                pages_by_number[page.number] = page
            pieces.extend(chunk_pieces)

    missing = set(range(1, total_pages + 1)) - set(pages_by_number)
    if missing:
        raise ValueError(
            "PDF conversion did not return page "
            + ", ".join(str(number) for number in sorted(missing))
        )
    pages = [pages_by_number[number] for number in range(1, total_pages + 1)]
    pages_json = [chunks_by_number[number] for number in range(1, total_pages + 1)]
    with pdf.open("rb") as source:
        document_id = hashlib.file_digest(source, "sha256").hexdigest()
    return Document(document_id, Parser(), pages, pieces), pages_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract PDF pages into text pieces and images."
    )
    parser.add_argument("pdf", type=Path, help="local PDF to parse")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for the output bundle",
    )
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=int(
            os.getenv("BW_PARSER_PAGE_BATCH", DEFAULT_PAGE_BATCH_SIZE)
        ),
        help=(
            "pages per Docling conversion (default: %(default)s). Larger values "
            "improve model batching but raise peak memory."
        ),
    )
    args = parser.parse_args()
    if args.page_batch_size < 1:
        parser.error("--page-batch-size must be positive")

    try:
        _write_bundle(args.pdf, args.output_dir, args.page_batch_size)
    except Exception as exc:  # noqa: BLE001 - report any extraction failure cleanly
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1

    print(args.output_dir / "pages.json")
    return 0
