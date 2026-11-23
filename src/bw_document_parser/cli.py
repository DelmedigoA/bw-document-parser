"""Convert a local PDF and write chunks for each physical page."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .chunks import ImageFiles, parse_page
from .document import Document, Parser


def _write_bundle(pdf: Path, output_dir: Path) -> None:
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
        document, pages = _parse_docling(pdf, bundle, images)
        for name, data in (("document.json", asdict(document)), ("pages.json", [asdict(page) for page in pages])):
            (bundle / name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        bundle.rename(output_dir)


def _parse_docling(pdf: Path, bundle: Path, images: ImageFiles):
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

    from .docling_adapter import read_page

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

    def convert_page(number: int) -> ConversionResult:
        # Docling can merge paragraphs across pages during document assembly.
        result = converter.convert(
            pdf, page_range=(number, number), raises_on_error=False
        )
        if (
            result.status != ConversionStatus.SUCCESS
            or number not in result.document.pages
        ):
            detail = "; ".join(
                error.error_message for error in result.errors if error.error_message
            )
            raise ValueError(
                f"PDF conversion failed on page {number} ({result.status.value})"
                + (f": {detail}" if detail else "")
            )
        return result

    first = convert_page(1)
    html_dir = bundle / "html"
    html_dir.mkdir()
    pages, pieces, chunks = [], [], []
    for number in range(1, first.input.page_count + 1):
        result = first if number == 1 else convert_page(number)
        html = result.document.export_to_html(page_no=number, image_mode=ImageRefMode.EMBEDDED)
        (html_dir / f"page-{number}.html").write_text(html, encoding="utf-8")
        chunks.append(parse_page(html, images.save))
        page, page_pieces = read_page(result.document, number, images)
        pages.append(page)
        pieces.extend(page_pieces)
    with pdf.open("rb") as source:
        document_id = hashlib.file_digest(source, "sha256").hexdigest()
    return Document(document_id, Parser(), pages, pieces), chunks


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
    args = parser.parse_args()

    try:
        _write_bundle(args.pdf, args.output_dir)
    except Exception as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1

    print(args.output_dir / "pages.json")
    return 0
