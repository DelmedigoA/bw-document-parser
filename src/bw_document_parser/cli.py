"""Convert a local PDF and write chunks for each physical page."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .bundle import write_json
from .parsing import DEFAULT_PAGE_BATCH_SIZE, parse_pdf
from .resolving import resolve_document


def _write_bundle(
    pdf: Path, output_dir: Path, page_batch_size: int, *,
    page_range: tuple[int, int] | None = None, parse_only: bool = False,
) -> None:
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("input must be a local PDF file")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so a failed extraction leaves no output bundle.
    with TemporaryDirectory(dir=output_dir.parent, prefix=".bw-document-parser-") as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        parsed = parse_pdf(pdf, bundle, page_batch_size=page_batch_size, page_range=page_range)
        write_json(bundle / "parsed.json", parsed, indent=2)
        if not parse_only:
            write_json(bundle / "document.json", resolve_document(parsed), indent=2)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        bundle.rename(output_dir)



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for English with Perplexity Decider, then extract PDF text and images."
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
    parser.add_argument("--parse-only", action="store_true", help="write the parse checkpoint and assets")
    parser.add_argument("--page-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="inclusive, one-based source page range")
    args = parser.parse_args()
    if args.page_batch_size < 1:
        parser.error("--page-batch-size must be positive")

    try:
        _write_bundle(
            args.pdf, args.output_dir, args.page_batch_size,
            page_range=tuple(args.page_range) if args.page_range else None,
            parse_only=args.parse_only,
        )
    except Exception as exc:  # noqa: BLE001 - report any extraction failure cleanly
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1

    print(args.output_dir / ("parsed.json" if args.parse_only else "document.json"))
    return 0
