import contextlib
import io
import json
import os
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    RefItem,
    Size,
    TableCell,
    TableData,
)
from PIL import Image

from bw_document_parser import embeddings, enrichment, openrouter
from bw_document_parser.bundle import read_json, write_json
from bw_document_parser.chunks import ImageFiles
from bw_document_parser.docling_adapter import (
    keep_page_boundaries,
    read_page,
    read_pages,
)
from bw_document_parser.document import (
    Document,
    Figure,
    ImageRef,
    Location,
    Page,
    Parser,
    Piece,
    Table,
    Text,
)
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


def text_bundle(directory: Path) -> Document:
    position = Location(1, (0.1, 0.1, 0.9, 0.9))
    pieces = [
        Piece("alpha", "text", Text("alpha source"), [position]),
        Piece("bravo", "text", Text("bravo source"), [replace_page(position, 2)]),
        Piece("charlie", "text", Text("charlie source"), [replace_page(position, 3)]),
    ]
    doc = Document(
        "text-example",
        Parser(),
        [
            Page(1, 100, 80, ["alpha"]),
            Page(2, 100, 80, ["bravo"]),
            Page(3, 100, 80, ["charlie"]),
        ],
        pieces,
    )
    write_json(directory / "document.json", asdict(doc), indent=2)
    (directory / "pages.json").write_text('[{"pieces": []}]', encoding="utf-8")
    return doc


def replace_page(location: Location, page: int) -> Location:
    return Location(page, location.bbox, location.text_range)


def description(text: str | None, *, decorative: bool = False) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"decorative": decorative, "description": text, "visible_text": None if decorative else "צפון"}),
    }}]}


def vectors_for(payload: dict, *, cost: float | None = None) -> dict:
    """Return one valid vector per input, keyed by its response index."""
    model = payload["model"]
    dimensions = 1024 if model == embeddings.TITAN else (
        1536 if model == embeddings.TEXT_MODEL else 4096
    )
    data = [
        {"index": index, "embedding": [float(index + 1)] + [0.0] * (dimensions - 1)}
        for index in range(len(payload["input"]))
    ]
    response = {"data": data}
    if cost is not None:
        response["usage"] = {"cost": cost}
    return response


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
            self.assertFalse(enrichment.enrich_bundle(self.bundle, workers=1))
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
            self.assertTrue(enrichment.enrich_bundle(self.bundle, workers=1))
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
            self.assertTrue(enrichment.enrich_bundle(self.bundle, workers=1))
        doc = read_json(self.bundle / "document.json")
        logo = next(piece for piece in doc["pieces"] if piece["id"] == "other")
        self.assertTrue(logo["enrichment"]["decorative"])
        self.assertIsNone(logo["enrichment"]["description"])
        self.assertTrue(logo["images"])
        entries = list(search_entries(doc, multimodal=True))
        self.assertTrue(any("figure" in entry.piece_ids for entry in entries))
        self.assertFalse(any("other" in entry.piece_ids for entry in entries))
        with patch.object(openrouter, "post") as post:
            self.assertTrue(enrichment.enrich_bundle(self.bundle, workers=1))
            post.assert_not_called()

    def test_reports_truncation_without_printing_response_content(self):
        reply = description("Private figure text")
        reply["choices"][0]["finish_reason"] = "length"
        with (
            patch.object(openrouter, "post", return_value=reply),
            self.assertRaisesRegex(openrouter.OpenRouterError, "reached its output token limit"),
        ):
            enrichment.describe({}, [])

    def test_embedding_modes_preserve_source_links_and_query_the_same_model(self):
        with patch.object(openrouter, "post", return_value=description("Bar chart by district")):
            self.assertTrue(enrichment.enrich_bundle(self.bundle, workers=1))

        def response(endpoint, payload):
            self.assertEqual("embeddings", endpoint)
            return vectors_for(payload, cost=0.001)

        with patch.object(openrouter, "post", side_effect=response) as request:
            self.assertTrue(embeddings.embed_bundle(self.bundle, "multimodal"))
            multimodal_calls = [call.args[1] for call in request.call_args_list]
            image_calls = [
                call for call in multimodal_calls
                if any(isinstance(item, dict) and "content" in item for item in call["input"])
            ]
            self.assertTrue(image_calls)
            self.assertTrue(all(call["input_type"] == "document" for call in image_calls))
            # Each image is its own request, and its text part carries the
            # enrichment that was just written to document.json.
            image_call = next(
                call for call in image_calls
                if any(
                    part.get("type") == "text" and "Bar chart by district" in part["text"]
                    for item in call["input"] for part in item["content"]
                )
            )
            parts = image_call["input"][0]["content"]
            self.assertEqual("image_url", parts[-1]["type"])
            self.assertIn("צפון", parts[0]["text"])
            request.reset_mock()
            self.assertTrue(embeddings.embed_bundle(self.bundle, "text"))
            self.assertTrue(all(
                isinstance(call.args[1]["input"], list)
                and all(isinstance(item, str) for item in call.args[1]["input"])
                for call in request.call_args_list
            ))
            request.reset_mock()
            hits = embeddings.search_bundle(self.bundle, "כמה אומתו?", "multimodal", limit=10)
            self.assertEqual(embeddings.MODES["multimodal"][0], request.call_args.args[1]["model"])
            self.assertEqual("query", request.call_args.args[1]["input_type"])
        multimodal = read_json(self.bundle / "embeddings.multimodal.json")
        self.assertEqual(4096, multimodal["dimensions"])
        self.assertGreater(multimodal["billing"]["total_cost"], 0)
        self.assertIn(multimodal["billing"]["allocation"], {"estimated", "mixed"})
        self.assertEqual(1536, read_json(self.bundle / "embeddings.text.json")["dimensions"])
        hit = next(hit for hit in hits if hit["rows"])
        self.assertEqual(TABLE, hit["pieces"][0]["content"]["html"])
        self.assertEqual(1, hit["locations"][0]["page"])
        self.assertEqual("figure", hit["pieces"][1]["id"])

    def test_a_quota_failure_saves_completed_vectors_and_reuses_them(self):
        vector = {"data": [{"index": 0, "embedding": [1.0] + [0.0] * 4095}]}
        quota = openrouter.OpenRouterError("quota", status=429)
        with patch.object(openrouter, "post", side_effect=[vector, quota]) as request:
            self.assertFalse(embeddings.embed_bundle(
                self.bundle, "multimodal", workers=1, batch_size=1,
            ))
            self.assertEqual(2, request.call_count)
            first_input = request.call_args_list[0].args[1]["input"]
        partial = read_json(self.bundle / "embeddings.multimodal.json")
        self.assertFalse(partial["complete"])
        self.assertEqual(1, len(partial["entries"]))
        with patch.object(openrouter, "post", return_value=vector) as request:
            self.assertTrue(embeddings.embed_bundle(
                self.bundle, "multimodal", workers=1, batch_size=1,
            ))
            self.assertFalse(any(call.args[1]["input"] == first_input for call in request.call_args_list))
            request.reset_mock()
            self.assertTrue(embeddings.embed_bundle(
                self.bundle, "multimodal", workers=1, batch_size=1,
            ))
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
            return vectors_for(payload)

        with patch.object(openrouter, "post", side_effect=response) as request:
            self.assertTrue(embeddings.embed_bundle(
                self.bundle, "multimodal", embeddings.TITAN, workers=1,
            ))
            self.assertTrue(any(isinstance(call.args[1]["input"], list) for call in request.call_args_list))
            self.assertTrue(embeddings.search_bundle(self.bundle, "North", "multimodal"))
        index = read_json(self.bundle / "embeddings.multimodal.json")
        self.assertEqual(embeddings.TITAN, index["model"])
        self.assertTrue(any("figure-text" in entry["id"] for entry in index["entries"]))
        self.assertTrue(all(_token_count(entry["text"]) <= 96 for entry in index["entries"]))

    def test_an_unavailable_model_override_preserves_the_previous_index(self):
        vector = {"data": [{"index": 0, "embedding": [1.0] + [0.0] * 4095}]}
        with patch.object(openrouter, "post", return_value=vector):
            self.assertTrue(embeddings.embed_bundle(
                self.bundle, "multimodal", workers=1, batch_size=1,
            ))
        path = self.bundle / "embeddings.multimodal.json"
        previous = path.read_bytes()
        unavailable = openrouter.OpenRouterError("no endpoints", status=404)
        with patch.object(openrouter, "post", side_effect=unavailable):
            self.assertFalse(embeddings.embed_bundle(
                self.bundle, "multimodal", embeddings.NEMOTRON,
                workers=1, batch_size=1,
            ))
        self.assertEqual(previous, path.read_bytes())

    def test_rejects_invalid_embeddings(self):
        with (
            patch.object(openrouter, "post", return_value={"data": [{"index": 0, "embedding": [0.5]}]}),
            self.assertRaisesRegex(openrouter.OpenRouterError, "invalid 4096-dimensional"),
        ):
            embeddings.embed(embeddings.MODES["multimodal"][0], 4096, "North")

    def test_http_quota_with_long_retry_after_exits_without_reposting(self):
        quota = HTTPError("https://openrouter.ai/api/v1/embeddings", 429, "quota", {"Retry-After": "86400"}, None)
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}),
            patch.object(openrouter, "urlopen", side_effect=quota) as request,
            self.assertRaisesRegex(openrouter.OpenRouterError, "Rate limit"),
        ):
            openrouter.post("embeddings", {})
        request.assert_called_once()

    def test_batch_embeddings_match_out_of_order_response_indices(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)

            def response(endpoint, payload):
                items = [
                    {"index": index, "embedding": [float(100 + index)] + [0.0] * 1535}
                    for index in range(len(payload["input"]))
                ]
                items.reverse()
                return {"data": items}

            with patch.object(openrouter, "post", side_effect=response) as request:
                self.assertTrue(embeddings.embed_bundle(
                    bundle, "text", workers=1, batch_size=8,
                ))
                self.assertEqual(1, request.call_count)
            index = read_json(bundle / "embeddings.text.json")
        by_text = {entry["text"]: entry["vector"][0] for entry in index["entries"]}
        self.assertEqual(
            {"alpha source": 100.0, "bravo source": 101.0, "charlie source": 102.0},
            by_text,
        )

    def test_partial_batch_response_requests_only_the_missing_input(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)
            calls = []

            def response(endpoint, payload):
                calls.append(len(payload["input"]))
                if len(calls) == 1:
                    return {
                        "data": [
                            {"index": 0, "embedding": [10.0] + [0.0] * 1535},
                            {"index": 2, "embedding": [30.0] + [0.0] * 1535},
                        ],
                        "usage": {"cost": 0.003},
                    }
                return vectors_for(payload, cost=0.001)

            with patch.object(openrouter, "post", side_effect=response):
                self.assertTrue(embeddings.embed_bundle(
                    bundle, "text", workers=1, batch_size=8,
                ))
            index = read_json(bundle / "embeddings.text.json")
        self.assertEqual([3, 1], calls)
        by_id = {entry["piece_ids"][0]: entry["vector"][0] for entry in index["entries"]}
        self.assertEqual({"alpha": 10.0, "bravo": 1.0, "charlie": 30.0}, by_id)
        # The batch and its retry are both billed to this index.
        self.assertAlmostEqual(0.004, index["billing"]["total_cost"])

    def test_empty_single_response_fails_without_retrying_forever(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)
            with patch.object(
                openrouter, "post", return_value={"data": []}
            ) as request:
                self.assertFalse(embeddings.embed_bundle(
                    bundle, "text", workers=1, batch_size=8,
                ))
            # One batch, then one request per input; a single input with no
            # vector must fail instead of recursively retrying itself.
            self.assertEqual(4, request.call_count)
            index = read_json(bundle / "embeddings.text.json")
            self.assertEqual([], index["entries"])
            self.assertFalse(index["complete"])

    def test_rejected_batch_falls_back_to_single_requests(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)
            calls = []

            def response(endpoint, payload):
                calls.append(payload["input"])
                if len(calls) == 1:
                    raise openrouter.OpenRouterError("unsupported batch", status=400)
                return vectors_for(payload)

            with patch.object(openrouter, "post", side_effect=response):
                self.assertTrue(embeddings.embed_bundle(
                    bundle, "text", workers=1, batch_size=8,
                ))
            index = read_json(bundle / "embeddings.text.json")
        self.assertEqual(4, len(calls))
        self.assertEqual(3, len(index["entries"]))

    def test_concurrent_enrichment_attributes_out_of_order_results(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            example_bundle(bundle)

            def response(endpoint, payload):
                context = json.loads(payload["messages"][1]["content"][0]["text"])
                caption = context["published_caption"]
                time.sleep(0.1 if caption == "Monthly totals" else 0.0)
                reply = description(f"desc:{caption}")
                reply["usage"] = {"cost": 0.002}
                return reply

            with patch.object(openrouter, "post", side_effect=response):
                self.assertTrue(enrichment.enrich_bundle(bundle, workers=2))
            document = read_json(bundle / "document.json")
        figures = {
            piece["id"]: piece["enrichment"]
            for piece in document["pieces"] if piece["content"]["kind"] == "figure"
        }
        self.assertEqual("desc:Monthly totals", figures["figure"]["description"])
        self.assertEqual("desc:Figure without a table", figures["other"]["description"])
        self.assertEqual(0.002, figures["figure"]["cost"])

    def test_rate_limit_stops_enrichment_and_retry_only_requests_the_failure(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            example_bundle(bundle)
            calls = []

            def limited(endpoint, payload):
                calls.append(payload)
                if len(calls) == 1:
                    return description("first")
                raise openrouter.OpenRouterError("quota", status=429)

            with patch.object(openrouter, "post", side_effect=limited):
                self.assertFalse(enrichment.enrich_bundle(bundle, workers=2))
            self.assertLessEqual(len(calls), 2)
            document = read_json(bundle / "document.json")
            completed = [
                piece["id"] for piece in document["pieces"]
                if piece.get("enrichment")
            ]
            self.assertEqual(1, len(completed))

            retries = []

            def retry(endpoint, payload):
                retries.append(payload)
                return description("second")

            with patch.object(openrouter, "post", side_effect=retry):
                self.assertTrue(enrichment.enrich_bundle(bundle, workers=2))
            self.assertEqual(1, len(retries))

    def test_rate_limit_stops_embedding_batches_and_resume_reuses_them(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)
            calls = []
            completed = 0

            def limited(endpoint, payload):
                nonlocal completed
                calls.append(payload["input"])
                if not completed and len(calls) == 1:
                    completed += 1
                    return vectors_for(payload)
                raise openrouter.OpenRouterError("quota", status=429)

            with patch.object(openrouter, "post", side_effect=limited):
                self.assertFalse(embeddings.embed_bundle(
                    bundle, "text", workers=2, batch_size=1,
                ))
            # A concurrent worker may already have started before the quota
            # answer sets the stop flag, so allow one request in flight.
            self.assertLessEqual(len(calls), 3)
            partial = read_json(bundle / "embeddings.text.json")
            self.assertEqual(completed, len(partial["entries"]))

            with patch.object(
                openrouter, "post",
                side_effect=lambda endpoint, payload: vectors_for(payload),
            ) as request:
                self.assertTrue(embeddings.embed_bundle(
                    bundle, "text", workers=2, batch_size=1,
                ))
                first_inputs = [call.args[1]["input"] for call in request.call_args_list]
                self.assertNotIn(calls[0], first_inputs)
                self.assertEqual(2, len(first_inputs))


class DoclingPageTests(unittest.TestCase):
    def test_read_pages_assigns_roots_and_links_by_provenance(self):
        document = DoclingDocument(name="fixture")
        document.add_page(1, Size(width=100, height=100))
        document.add_page(2, Size(width=100, height=100))

        def provenance(page):
            return ProvenanceItem(page_no=page, charspan=(0, 1), bbox=BoundingBox(
                l=10, t=10, r=90, b=90, coord_origin=CoordOrigin.TOPLEFT,
            ))

        document.add_text(DocItemLabel.TEXT, "First", prov=provenance(1))
        caption = document.add_text(DocItemLabel.CAPTION, "Caption", prov=provenance(1))
        document.add_picture(caption=caption, prov=provenance(1))
        second = document.add_text(DocItemLabel.TEXT, "Second", prov=provenance(2))
        with TemporaryDirectory() as directory:
            images = ImageFiles(Path(directory) / "images")
            pages, pieces = read_pages(document, images)
            page_two, page_two_pieces = read_page(document, 2, images)
        by_id = {piece.id: piece for piece in pieces}
        self.assertEqual(
            ["p1:#/texts/0", "p1:#/pictures/0"],
            pages[0].piece_ids,
        )
        # Docling numbers texts across the whole conversion (Second is
        # #/texts/2), but the bundle renumbers each collection per page.
        self.assertEqual("#/texts/2", second.self_ref)
        self.assertEqual(["p2:#/texts/0"], pages[1].piece_ids)
        self.assertEqual(
            ["p1:#/texts/1"],
            by_id["p1:#/pictures/0"].caption_ids,
        )
        self.assertEqual(["p2:#/texts/0"], page_two.piece_ids)
        self.assertEqual(["p2:#/texts/0"], [piece.id for piece in page_two_pieces])

    def test_group_piece_follows_the_page_of_its_content(self):
        document = DoclingDocument(name="group")
        document.add_page(1, Size(width=100, height=100))
        document.add_page(2, Size(width=100, height=100))

        def provenance(page):
            return ProvenanceItem(page_no=page, charspan=(0, 1), bbox=BoundingBox(
                l=10, t=10, r=90, b=90, coord_origin=CoordOrigin.TOPLEFT,
            ))

        group = document.add_group(name="list")
        item = document.add_text(DocItemLabel.LIST_ITEM, "Two", prov=provenance(2))
        group.children = [item.get_ref()]
        document.body.children = [RefItem(cref=group.self_ref)]
        with TemporaryDirectory() as directory:
            pages, pieces = read_pages(
                document, ImageFiles(Path(directory) / "images")
            )
        by_id = {piece.id: piece for piece in pieces}
        # Docling starts a batch-wide group on the batch's first page. The
        # bundle keeps the group with its content so each page owns its list,
        # as a per-page conversion would.
        self.assertEqual([], pages[0].piece_ids)
        self.assertEqual(["p2:#/groups/0"], pages[1].piece_ids)
        self.assertEqual(["p2:#/texts/0"], by_id["p2:#/groups/0"].children)

    def test_page_boundary_patch_drops_only_cross_page_merges(self):
        class Element:
            def __init__(self, cid, page_no):
                self.cid = cid
                self.page_no = page_no

        class Predictor:
            def predict_merges(self, sorted_elements):
                return {0: [1, 2], 2: [3]}

        class Model:
            ro_model = Predictor()

        class Pipeline:
            reading_order_model = Model()

        pipeline = Pipeline()
        self.assertTrue(keep_page_boundaries(pipeline))
        merges = pipeline.reading_order_model.ro_model.predict_merges(
            sorted_elements=[Element(0, 1), Element(1, 1), Element(2, 2), Element(3, 2)],
        )
        self.assertEqual({0: [1], 2: [3]}, merges)
        self.assertTrue(keep_page_boundaries(pipeline))


if __name__ == "__main__":
    unittest.main()

class ReviewRegressionTests(unittest.TestCase):
    def test_cached_duplicate_cost_is_redistributed_when_occurrences_change(self):
        from dataclasses import replace

        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)
            source = next(iter(search_entries(read_json(bundle / 'document.json'), multimodal=False)))
            with patch.object(embeddings, 'search_entries', return_value=[source, replace(source, id='duplicate')]), \
                 patch.object(openrouter, 'post', side_effect=lambda _, payload: vectors_for(payload, cost=0.02)):
                self.assertTrue(embeddings.embed_bundle(bundle, 'text', workers=1))
            with patch.object(embeddings, 'search_entries', return_value=[source]), \
                 patch.object(openrouter, 'post', side_effect=AssertionError('unexpected request')):
                self.assertTrue(embeddings.embed_bundle(bundle, 'text', workers=1))
            index = read_json(bundle / 'embeddings.text.json')
            self.assertAlmostEqual(index['billing']['total_cost'], 0.02)

    def test_interrupt_drains_finished_embedding_requests(self):
        from concurrent.futures import as_completed as real_as_completed

        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            text_bundle(bundle)
            calls = 0
            def interrupt_once(futures):
                nonlocal calls
                calls += 1
                if calls == 1:
                    next(real_as_completed(futures))
                    raise KeyboardInterrupt()
                return real_as_completed(futures)
            with patch.object(embeddings, 'as_completed', side_effect=interrupt_once), \
                 patch.object(openrouter, 'post', side_effect=lambda _, payload: vectors_for(payload)), \
                 self.assertRaises(KeyboardInterrupt):
                embeddings.embed_bundle(bundle, 'text', workers=1, batch_size=3)
            index = read_json(bundle / 'embeddings.text.json')
            self.assertEqual(len(index['entries']), 3)

    def test_interrupt_drains_finished_enrichment_requests(self):
        from concurrent.futures import as_completed as real_as_completed

        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            example_bundle(bundle)
            calls = 0
            def interrupt_once(futures):
                nonlocal calls
                calls += 1
                if calls == 1:
                    next(real_as_completed(futures))
                    raise KeyboardInterrupt()
                return real_as_completed(futures)
            with patch.object(enrichment, 'as_completed', side_effect=interrupt_once), \
                 patch.object(openrouter, 'post', return_value=description('A figure')), \
                 self.assertRaises(KeyboardInterrupt):
                enrichment.enrich_bundle(bundle, workers=1)
            document = read_json(bundle / 'document.json')
            self.assertTrue(any(piece.get('enrichment') for piece in document['pieces']))
