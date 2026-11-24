"""Serve one source PDF and its parse bundle on the local loopback address."""

import argparse
import json
import mimetypes
import re
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .bundle import read_document
from .document import pack_piece, unpack_piece
from .embeddings import MODES, search_bundle
from .resolving import resolved_pieces

_STATIC = Path(__file__).with_name("static")
_ASSETS = {
    "/": "index.html",
    "/inspector.css": "inspector.css",
    "/inspector.js": "inspector.js",
    "/document-viewer.js": "document-viewer.js",
    "/vendor/pdfjs/pdf.mjs": "vendor/pdfjs/pdf.mjs",
    "/vendor/pdfjs/pdf.worker.mjs": "vendor/pdfjs/pdf.worker.mjs",
}


def _resolved_page(document: dict, number: int) -> dict | None:
    page = next((page for page in document["pages"] if page["number"] == number), None)
    if page is None:
        return None
    sections = []
    source_blocks: dict = {}
    for piece in resolved_pieces(document):
        blocks = [
            block for block in unpack_piece(piece, document["source_blocks"])
            if block["page"] == number or any(loc["page"] == number for loc in block["locations"])
        ]
        if blocks:
            sections.append(pack_piece({
                "id": piece["id"], "title": piece["title"], "page": piece["page"],
                **({"section_heading": piece["section_heading"]} if "section_heading" in piece else {}),
                "blocks": blocks,
                "embeddings": [
                    {key: value for key, value in entry.items() if key != "vector"}
                    for entry in piece["embeddings"]
                    if any(location["page"] == number for location in entry["locations"])
                ],
            }, source_blocks))
    return {**page, "pieces": sections, "source_blocks": source_blocks}


def _handler(pdf: Path, bundle: Path) -> type[BaseHTTPRequestHandler]:
    pages_file = bundle / "pages.json"
    document_file = bundle / "document.json"
    has_document = document_file.is_file()
    document = read_document(bundle) if has_document else None
    resolved = document is not None and document["schema_version"] == 4
    if resolved:
        numbers = [page["number"] for page in document["pages"]]
    else:
        pages = json.loads(pages_file.read_text(encoding="utf-8"))
        if not isinstance(pages, list) or any(
            not isinstance(page, dict) or not isinstance(page.get("pieces"), list)
            for page in pages
        ):
            raise ValueError("pages.json must contain a list of pages with pieces")
        numbers = list(range(1, len(pages) + 1))
    html_files = {
        f"/html/page-{number}.html": bundle / "html" / f"page-{number}.html" for number in numbers
    }
    html_files = {url: file for url, file in html_files.items() if file.is_file()}
    stamp = document_file.stat().st_mtime_ns if resolved else None

    def current_document():
        nonlocal document, stamp
        updated = document_file.stat().st_mtime_ns
        if stamp != updated:
            document = read_document(bundle)
            stamp = updated
        return document

    class InspectorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            _, port = self.server.server_address
            if self.headers.get("Host") != f"127.0.0.1:{port}":
                self.send_error(403)
                return

            url = urlsplit(self.path).path
            if url == "/config.json":
                self._json({
                    "title": pdf.name,
                    "htmlPages": list(html_files),
                    "hasDocument": has_document,
                    "resolved": resolved,
                    "parsedPages": numbers,
                    "searchModes": [
                        mode for mode in MODES
                        if (mode in current_document().get("embedding_modes", {}) if resolved
                            else has_document and (bundle / f"embeddings.{mode}.json").is_file())
                    ],
                })
                return

            if resolved and (match := re.fullmatch(r"/api/pages/([1-9][0-9]{0,8})", url)):
                page = _resolved_page(current_document(), int(match[1]))
                if page is None:
                    self._json({"error": "This source page was not parsed."}, status=404)
                else:
                    self._json(page)
                return

            if url in _ASSETS:
                file = _STATIC / _ASSETS[url]
            elif url == "/source.pdf":
                file = pdf
            elif url == "/pages.json" and not resolved:
                file = pages_file
            elif url == "/document.json" and has_document and not resolved:
                file = document_file
            elif url in html_files:
                file = html_files[url]
            elif re.fullmatch(r"/images/[0-9a-f]{24}\.[a-zA-Z0-9]+", url):
                file = bundle / url.lstrip("/")
            else:
                self.send_error(404)
                return

            try:
                source = file.open("rb")
            except OSError:
                self.send_error(404)
                return
            with source:
                content_type = (
                    "text/javascript"
                    if file.suffix in {".js", ".mjs"}
                    else mimetypes.guess_type(file.name)[0]
                    or "application/octet-stream"
                )
                self._headers(
                    content_type, file.stat().st_size, page_html=url in html_files
                )
                shutil.copyfileobj(source, self.wfile)

        def do_POST(self) -> None:
            _, port = self.server.server_address
            host = f"127.0.0.1:{port}"
            if self.headers.get("Host") != host or self.headers.get("Origin") != f"http://{host}":
                self.send_error(403)
                return
            if self.path != "/api/search":
                self.send_error(404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192 or self.headers.get_content_type() != "application/json":
                    raise ValueError
                data = json.loads(self.rfile.read(length))
                query, mode = data["query"], data["mode"]
                figures_only = data.get("figuresOnly", False)
                if (
                    not isinstance(query, str) or not query.strip() or len(query) > 2000
                    or mode not in MODES or type(figures_only) is not bool
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                self._json({"error": "Enter a search query and choose an available index."}, status=400)
                return

            try:
                hits = search_bundle(bundle, query, mode, limit=10, figures_only=figures_only)
            except FileNotFoundError:
                self._json({"error": "Build embeddings for this bundle first."}, status=404)
                return
            except ValueError as exc:
                self._json({"error": str(exc)}, status=422)
                return
            except OSError:
                self._json({"error": "The search index could not be read."}, status=500)
                return

            results = []
            for hit in hits:
                piece = hit["pieces"][0]
                image = hit["image_path"] or next((item["path"] for item in piece["images"]), None)
                location = next(iter(hit["locations"]), None)
                if location is None:
                    location = next(
                        ({"page": number} for number in numbers
                         if piece["id"].startswith(f"p{number}:")),
                        None,
                    )
                results.append({
                    "pieceId": piece["id"],
                    "kind": piece["content"]["kind"],
                    "page": location["page"] if location else None,
                    "image": image,
                    "text": (piece.get("enrichment") or {}).get("description") or hit["text"],
                })
                if parent := hit.get("parent_piece"):
                    results[-1].update(parentPieceId=parent["id"], targetId=hit["id"])
            self._json({"results": results})

        def _json(self, data: dict, *, status: int = 200) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._headers("application/json", len(payload), status=status)
            self.wfile.write(payload)

        def _headers(
            self, content_type: str, size: int, *, page_html: bool = False, status: int = 200
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if page_html:
                self.send_header(
                    "Content-Security-Policy",
                    "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:",
                )
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return InspectorHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="original PDF")
    parser.add_argument("bundle", type=Path, help="parser output directory")
    parser.add_argument(
        "--port", type=int, default=8765, help="local port (default: 8765)"
    )
    args = parser.parse_args()

    try:
        if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
            raise ValueError("input must be a local PDF file")
        if not 0 <= args.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        handler = _handler(args.pdf.resolve(), args.bundle.resolve())
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(
                f"Parsing inspector: http://127.0.0.1:{server.server_port}", flush=True
            )
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    return 0
