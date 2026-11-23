import contextlib
import io
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from bw_document_parser import embeddings, enrichment, openrouter
from bw_document_parser.bundle import read_json, write_json
from bw_document_parser.chunks import ImageFiles
from bw_document_parser.docling_adapter import read_page
from bw_document_parser.document import Document, Figure, ImageRef, Location, Page, Parser, Piece, Table, Text
from bw_document_parser.search_entries import search_entries, table_rows

TABLE = (
    '<table><thead><tr><th rowspan="2">District</th><th colspan="2">2025</th></tr>'
    '<tr><th>Reported</th><th>Verified</th></tr></thead>'
    '<tbody><tr><th>North</th><td>18</td><td>12</td></tr></tbody></table>'
)


def example_bundle(directory: Path) -> Document:
    with Image.new("RGB", (100, 80), "white") as image, io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        path = ImageFiles(directory / "images").save_bytes(buffer.getvalue(), "image/png")
    position = Location(1, (0.1, 0.1, 0.9, 0.9))
    pieces = [
        Piece("title", "section_header", Text("Incident review", 1), [position]),
        Piece("figure", "picture", Figure(), [position], ["table"], ["caption"], images=[ImageRef(path)]),
        Piece("caption", "caption", Text("Monthly totals"), [position]),
        Piece("table", "table", Table(TABLE), [position], images=[ImageRef(path)]),
        Piece("other", "picture", Figure(), [position], caption_ids=["other-caption"], images=[ImageRef(path)]),
        Piece("other-caption", "caption", Text("Figure without a table"), [position]),
    ]
    doc = Document("example", Parser(), [Page(1, 100, 80, ["title", "figure", "other"])], pieces)
    write_json(directory / "document.json", asdict(doc), indent=2)
    (directory / "pages.json").write_text('[{"pieces": []}]', encoding="utf-8")
    return doc


def description(text: str | None, *, decorative: bool = False) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"decorative": decorative, "description": text, "visible_text": None if decorative else "צפון"}),
    }}]}


class DocumentTests(unittest.TestCase):
    def test_extracts_docling_html_and_keeps_picture_relationships(self):
        document = DoclingDocument(name="fixture")
        document.add_page(1, Size(width=100, height=100))
        location = ProvenanceItem(page_no=1, charspan=(0, 0), bbox=BoundingBox(
            l=10, t=10, r=90, b=90, coord_origin=CoordOrigin.TOPLEFT,
        ))
        caption = document.add_text(DocItemLabel.CAPTION, "Published caption", prov=location)
        picture = document.add_picture(caption=caption, prov=location)
        table = document.add_table(TableData(num_rows=2, num_cols=2, table_cells=[
            TableCell(
                start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0,
                end_col_offset_idx=2, col_span=2, column_header=True, text="2025",
            ),
            TableCell(
                start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0,
                end_col_offset_idx=1, text="North",
            ),
            TableCell(
                start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1,
                end_col_offset_idx=2, text="12",
            ),
        ]), parent=picture, prov=location)
        standalone = document.add_picture(prov=location)
        with TemporaryDirectory() as directory:
            images = ImageFiles(Path(directory) / "images")
            page, pieces = read_page(document, 1, images)
            document.add_page(2, Size(width=100, height=100))
            # Conversion supplies a separate Docling document for each source page.
            blank_doc = DoclingDocument(name="blank", pages={2: document.pages[2]})
            blank, blank_pieces = read_page(blank_doc, 2, images)

        by_id = {piece.id: piece for piece in pieces}
        table_id = "p1:" + table.self_ref
        picture_id = "p1:" + picture.self_ref
        standalone_id = "p1:" + standalone.self_ref
        self.assertEqual([picture_id, standalone_id], page.piece_ids)
        self.assertEqual([table_id], by_id[picture_id].children)
        self.assertEqual(["p1:" + caption.self_ref], by_id[picture_id].caption_ids)
        self.assertFalse(by_id[standalone_id].children)
        self.assertEqual(table.export_to_html(document, add_caption=False), by_id[table_id].content.html)
        self.assertIn('colspan="2"', by_id[table_id].content.html)
        self.assertEqual((0.1, 0.1, 0.9, 0.9), by_id[table_id].locations[0].bbox)
        self.assertFalse(blank.piece_ids)
        self.assertFalse(blank_pieces)

    def test_row_chunks_keep_headers_and_exact_values(self):
        headers, rows = table_rows(TABLE)
        self.assertIn("2025 → Verified", headers)
        self.assertIn("District: North | 2025 → Reported: 18 | 2025 → Verified: 12", rows[0][1])
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            example_bundle(bundle)
            document = read_json(bundle / "document.json")
            table = next(piece for piece in document["pieces"] if piece["id"] == "table")
            table["content"]["html"] = TABLE.replace(
                "</tbody>", "<tr><th>South</th><td>11</td><td>8</td></tr>" * 70 + "</tbody>",
            )
            entries = [entry for entry in search_entries(document, multimodal=True) if entry.rows]
        self.assertGreater(len(entries), 1)
        for entry in entries:
            self.assertEqual(["table", "figure"], entry.piece_ids)
            self.assertIn("2025 → Verified", entry.text)
            self.assertIn("Monthly totals", entry.text)


class HostedTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name)
        example_bundle(self.bundle)
        for stream in ("stdout", "stderr"):
            self.enterContext(getattr(contextlib, f"redirect_{stream}")(io.StringIO()))

    def test_description_keeps_sources_and_resumes_failed_figure(self):
        invalid = description("Not complete")
        invalid["choices"][0]["finish_reason"] = "length"
        with patch.object(openrouter, "post", side_effect=[description("Bar chart"), invalid]) as request:
            self.assertFalse(enrichment.enrich_bundle(self.bundle))
        for _, (endpoint, payload), _ in request.mock_calls:
            self.assertEqual("chat/completions", endpoint)
            self.assertEqual("tencent/hy4-flash", payload["model"])
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertIn("Incident review", payload["messages"][1]["content"][0]["text"])
            self.assertEqual("image_url", payload["messages"][1]["content"][1]["type"])
        document = read_json(self.bundle / "document.json")
        self.assertEqual("Bar chart", document["pieces"][1]["enrichment"]["description"])
        self.assertEqual("צפון", document["pieces"][1]["enrichment"]["visible_text"])
        self.assertEqual("Monthly totals", document["pieces"][2]["content"]["text"])
        self.assertIsNone(document["pieces"][4]["enrichment"])

        with patch.object(openrouter, "post", return_value=description("Second description")) as retry:
            self.assertTrue(enrichment.enrich_bundle(self.bundle))
            retry.assert_called_once()
            self.assertIn("Figure without a table", retry.call_args.args[1]["messages"][1]["content"][0]["text"])

    def test_accepts_fenced_description_without_optional_transcription(self):
        reply = description("Unused")
        reply["choices"][0]["message"]["content"] = '```json\n{"decorative":false,"description":"Chart"}\n```'
        with patch.object(openrouter, "post", return_value=reply) as request:
            self.assertEqual(("Chart", None, False), enrichment.describe({}, []))
            payload = request.call_args.args[1]
            self.assertEqual({"effort": "none"}, payload["reasoning"])
            self.assertEqual(4096, payload["max_tokens"])

    def test_decorative_figures_are_cached_and_omitted_from_search(self):
        with patch.object(openrouter, "post", side_effect=[
            description("Chart"), description(None, decorative=True),
        ]):
            self.assertTrue(enrichment.enrich_bundle(self.bundle))
        doc = read_json(self.bundle / "document.json")
        logo = next(piece for piece in doc["pieces"] if piece["id"] == "other")
        self.assertTrue(logo["enrichment"]["decorative"])
        self.assertIsNone(logo["enrichment"]["description"])
        self.assertTrue(logo["images"])
        entries = list(search_entries(doc, multimodal=True))
        self.assertTrue(any("figure" in entry.piece_ids for entry in entries))
        self.assertFalse(any("other" in entry.piece_ids for entry in entries))
        with patch.object(openrouter, "post") as post:
            self.assertTrue(enrichment.enrich_bundle(self.bundle))
            post.assert_not_called()

    def test_reports_truncation_without_printing_response_content(self):
        reply = description("Private figure text")
        reply["choices"][0]["finish_reason"] = "length"
        with patch.object(openrouter, "post", return_value=reply):
            with self.assertRaisesRegex(openrouter.OpenRouterError, "reached its output token limit"):
                enrichment.describe({}, [])

    def test_embedding_modes_preserve_source_links_and_query_the_same_model(self):
        with patch.object(openrouter, "post", return_value=description("Bar chart by district")):
            self.assertTrue(enrichment.enrich_bundle(self.bundle))

        def response(endpoint, payload):
            self.assertEqual("embeddings", endpoint)
            size = 1536 if payload["model"] == embeddings.TEXT_MODEL else 4096
            return {"data": [{"index": 0, "embedding": [0.0, 1.0] + [0.0] * (size - 2)}]}

        with patch.object(openrouter, "post", side_effect=response) as request:
            self.assertTrue(embeddings.embed_bundle(self.bundle, "multimodal"))
            multimodal_calls = [call.args[1] for call in request.call_args_list]
            self.assertTrue(any(isinstance(call["input"], list) for call in multimodal_calls))
            image_call = next(call for call in multimodal_calls if isinstance(call["input"], list))
            self.assertEqual("image_url", image_call["input"][0]["content"][1]["type"])
            self.assertEqual("document", image_call["input_type"])
            self.assertIn("Bar chart by district", image_call["input"][0]["content"][0]["text"])
            self.assertIn("צפון", image_call["input"][0]["content"][0]["text"])
            request.reset_mock()
            self.assertTrue(embeddings.embed_bundle(self.bundle, "text"))
            self.assertTrue(all(isinstance(call.args[1]["input"], str) for call in request.call_args_list))
            request.reset_mock()
            hits = embeddings.search_bundle(self.bundle, "כמה אומתו?", "multimodal", limit=10)
            self.assertEqual(embeddings.MODES["multimodal"][0], request.call_args.args[1]["model"])
            self.assertEqual("query", request.call_args.args[1]["input_type"])
        self.assertEqual(4096, read_json(self.bundle / "embeddings.multimodal.json")["dimensions"])
        self.assertEqual(1536, read_json(self.bundle / "embeddings.text.json")["dimensions"])
        hit = next(hit for hit in hits if hit["rows"])
        self.assertEqual(TABLE, hit["pieces"][0]["content"]["html"])
        self.assertEqual(1, hit["locations"][0]["page"])
        self.assertEqual("figure", hit["pieces"][1]["id"])

    def test_a_quota_failure_saves_completed_vectors_and_reuses_them(self):
        vector = {"data": [{"index": 0, "embedding": [1.0] + [0.0] * 4095}]}
        quota = openrouter.OpenRouterError("quota", status=429)
        with patch.object(openrouter, "post", side_effect=[vector, quota]) as request:
            self.assertFalse(embeddings.embed_bundle(self.bundle, "multimodal"))
            self.assertEqual(2, request.call_count)
            first_input = request.call_args_list[0].args[1]["input"]
        partial = read_json(self.bundle / "embeddings.multimodal.json")
        self.assertFalse(partial["complete"])
        self.assertEqual(1, len(partial["entries"]))
        with patch.object(openrouter, "post", return_value=vector) as request:
            self.assertTrue(embeddings.embed_bundle(self.bundle, "multimodal"))
            self.assertFalse(any(call.args[1]["input"] == first_input for call in request.call_args_list))
            request.reset_mock()
            self.assertTrue(embeddings.embed_bundle(self.bundle, "multimodal"))
            request.assert_not_called()

    def test_titan_chunks_long_inputs_and_searches_with_titan(self):
        from bw_document_parser.chunks import _token_count

        path = self.bundle / "document.json"
        doc = read_json(path)
        doc["pieces"][1]["enrichment"] = {"description": "Bars by district. " * 120}
        write_json(path, doc)

        def response(endpoint, payload):
            self.assertEqual("embeddings", endpoint)
            self.assertEqual(embeddings.TITAN, payload["model"])
            self.assertEqual(1024, payload["dimensions"])
            self.assertNotIn("input_type", payload)
            return {"data": [{"index": 0, "embedding": [1.0] + [0.0] * 1023}]}

        with patch.object(openrouter, "post", side_effect=response) as request:
            self.assertTrue(embeddings.embed_bundle(self.bundle, "multimodal", embeddings.TITAN))
            self.assertTrue(any(isinstance(call.args[1]["input"], list) for call in request.call_args_list))
            self.assertTrue(embeddings.search_bundle(self.bundle, "North", "multimodal"))
        index = read_json(self.bundle / "embeddings.multimodal.json")
        self.assertEqual(embeddings.TITAN, index["model"])
        self.assertTrue(any("figure-text" in entry["id"] for entry in index["entries"]))
        self.assertTrue(all(_token_count(entry["text"]) <= 96 for entry in index["entries"]))

    def test_an_unavailable_model_override_preserves_the_previous_index(self):
        vector = {"data": [{"index": 0, "embedding": [1.0] + [0.0] * 4095}]}
        with patch.object(openrouter, "post", return_value=vector):
            self.assertTrue(embeddings.embed_bundle(self.bundle, "multimodal"))
        path = self.bundle / "embeddings.multimodal.json"
        previous = path.read_bytes()
        unavailable = openrouter.OpenRouterError("no endpoints", status=404)
        with patch.object(openrouter, "post", side_effect=unavailable):
            self.assertFalse(embeddings.embed_bundle(self.bundle, "multimodal", embeddings.NEMOTRON))
        self.assertEqual(previous, path.read_bytes())

    def test_rejects_invalid_embeddings(self):
        with patch.object(openrouter, "post", return_value={"data": [{"index": 0, "embedding": [0.5]}]}):
            with self.assertRaisesRegex(openrouter.OpenRouterError, "invalid 4096-dimensional"):
                embeddings.embed(embeddings.MODES["multimodal"][0], 4096, "North")

    def test_http_quota_with_long_retry_after_exits_without_reposting(self):
        quota = HTTPError("https://openrouter.ai/api/v1/embeddings", 429, "quota", {"Retry-After": "86400"}, None)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
            with patch.object(openrouter, "urlopen", side_effect=quota) as request:
                with self.assertRaisesRegex(openrouter.OpenRouterError, "Rate limit"):
                    openrouter.post("embeddings", {})
                request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
