import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_rag_bundle import description, example_bundle, vectors_for

from bw_document_parser import openrouter
from bw_document_parser.bundle import fingerprint, read_document, write_json
from bw_document_parser.chunks import MAX_TOKENS, _token_count
from bw_document_parser.document import Location, Page, Piece, Text, unpack_piece
from bw_document_parser.embeddings import embed_bundle, search_bundle
from bw_document_parser.enrichment import enrich_bundle
from bw_document_parser.inspector import _resolved_page
from bw_document_parser.resolving import resolve_document


class ResolvedPipelineTests(unittest.TestCase):
    def test_small_pieces_and_hosted_stages_preserve_structure(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            source = asdict(example_bundle(bundle))
            source["pieces"][0]["label"] = "section_header"
            source["pages"].append(asdict(Page(2, 100, 80, ["continued"])))
            source["pieces"].append(asdict(Piece(
                "continued", "text", Text("Second page evidence"), [Location(2)],
            )))
            document = resolve_document(source)
            opening, continuation = document["pages"]
            heading, figure, other = opening["pieces"]
            self.assertEqual(["title"], heading["source_ids"])
            self.assertEqual(["continued"], continuation["pieces"][0]["source_ids"])
            self.assertTrue(figure["textual_part"]["tables"][0]["html"])
            self.assertTrue(figure["visual_part"][0]["url"])
            self.assertNotIn("blocks", figure)
            visible = _resolved_page(document, 2)["pieces"]
            self.assertEqual(continuation["pieces"][0]["id"], visible[0]["id"])
            self.assertEqual(["continued"], visible[0]["source_ids"])
            write_json(bundle / "document.json", document)

            approved = {"answers": {"enrich": {"type": "noul", "noul": 0.9}}}
            skipped = {"answers": {"enrich": {"type": "noul", "noul": 0.1}}}
            with (
                patch.object(openrouter, "decide", side_effect=[approved, skipped]) as decider,
                patch.object(openrouter, "post", return_value=description("Chart description")) as describe,
            ):
                self.assertTrue(enrich_bundle(bundle, workers=1))
                self.assertEqual(2, decider.call_count)
                self.assertEqual("perplexity/pplx-decider-v1.2", decider.call_args.args[0]["model"])
                self.assertEqual("image_url", decider.call_args.args[0]["state"][1]["type"])
                describe.assert_called_once()
            with (
                patch.object(openrouter, "decide", side_effect=AssertionError("unexpected Decider call")),
                patch.object(openrouter, "post", side_effect=AssertionError("unexpected enrichment call")),
            ):
                self.assertTrue(enrich_bundle(bundle, workers=1))
            with patch.object(openrouter, "post", side_effect=lambda _, payload: vectors_for(payload)):
                self.assertTrue(embed_bundle(bundle, "text", workers=1))

            saved_doc = read_document(bundle)
            _, saved, saved_other = saved_doc["pages"][0]["pieces"]
            self.assertEqual(figure["id"], saved["id"])
            self.assertEqual(
                fingerprint([(block["id"], block["locations"]) for block in unpack_piece(figure, document["source_blocks"])]),
                fingerprint([(block["id"], block["locations"]) for block in unpack_piece(saved, saved_doc["source_blocks"])]),
            )
            self.assertEqual("Chart description", saved["visual_part"][0]["description"])
            self.assertFalse(saved_doc["source_blocks"]["other"]["enrichment_decision"]["enrich"])
            self.assertEqual(other["id"], saved_other["id"])
            self.assertIsNone(saved_other["visual_part"])
            skipped_image, = saved_doc["source_blocks"]["other"]["excluded_visuals"]
            self.assertTrue(skipped_image["url"])
            self.assertIsNone(skipped_image["description"])
            self.assertTrue(saved["embeddings"][0]["vector"])
            self.assertFalse((bundle / "embeddings.text.json").exists())
            with patch.object(openrouter, "post", side_effect=lambda _, payload: vectors_for(payload)):
                hits = search_bundle(bundle, "chart", "text", figures_only=True)
            self.assertEqual(figure["id"], hits[0]["parent_piece"]["id"])
            self.assertEqual("figure", hits[0]["pieces"][0]["id"])
            shown = _resolved_page(read_document(bundle), 1)["pieces"][1]["embeddings"]
            self.assertTrue(shown)
            self.assertTrue(all("vector" not in target for target in shown))

    def test_prose_is_packed_and_large_paragraphs_keep_source_offsets(self):
        with TemporaryDirectory() as directory:
            source = asdict(example_bundle(Path(directory)))
        words = ("word " * 1800).strip()
        extra = [
            Piece("a", "text", Text("Short opening."), [Location(1)]),
            Piece("b", "text", Text("Next paragraph."), [Location(1)]),
            Piece("long", "text", Text(words), [Location(1)]),
        ]
        source["pieces"][1:1] = list(map(asdict, extra))
        source["pages"][0]["piece_ids"][1:1] = ["a", "b", "long"]
        resolved = resolve_document(source)
        pieces = resolved["pages"][0]["pieces"]
        self.assertEqual(["title", "a", "b"], pieces[0]["source_ids"])
        self.assertEqual("Figure", pieces[-2]["title"])
        self.assertTrue(all(
            _token_count(piece["textual_part"]["text"]) <= MAX_TOKENS for piece in pieces[:-2]
        ))
        fragments = [block for block in resolved["source_blocks"].values() if block.get("source_id") == "long"]
        self.assertGreater(len(fragments), 1)
        for fragment in fragments:
            start, end = fragment["source_text_range"]
            self.assertEqual(words[start:end], fragment["content"]["text"])
