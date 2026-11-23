import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bw_document_parser.bundle import read_document
from bw_document_parser.cli import _write_bundle
from bw_document_parser.document import (
    Document,
    Location,
    Page,
    Parser,
    Piece,
    Table,
    Text,
)
from bw_document_parser.resolving import block_document, resolve_document
from bw_document_parser.search_entries import search_entries
from bw_document_parser.table_links import link_table_context


class TableTitlesTests(unittest.TestCase):
    def setUp(self):
        self.heading = Piece(
            "heading", "section_header", Text("Population table"),
            [Location(3, (0.06, 0.69, 0.57, 0.70))],
        )
        self.table = Piece(
            "table", "table", Table("<table><tr><td>12</td></tr></table>"),
            [Location(3, (0.06, 0.71, 0.94, 0.89))],
        )

    def document(self, heading, table):
        return Document("fixture", Parser(), [Page(3, 100, 100, ["heading", "table"])], [heading, table])

    def test_parser_writes_the_link_to_the_bundle(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "source.pdf"
            pdf.touch()
            bundle = root / "bundle"
            document = self.document(self.heading, self.table)
            with patch("bw_document_parser.cli.parse_pdf", return_value=asdict(document)):
                _write_bundle(pdf, bundle, 12)
            resolved = read_document(bundle)
            linked = block_document(resolved)
        self.assertEqual(["heading", "table"], resolved["pages"][0]["pieces"][0]["source_ids"])
        self.assertEqual("heading", linked["pieces"][1]["inferred_title_id"])
        self.assertEqual([], linked["pieces"][1]["caption_ids"])
        self.assertEqual(list(search_entries(json.loads(json.dumps(asdict(document))), multimodal=False)),
                         list(search_entries(linked, multimodal=False)))
        self.assertEqual(0, link_table_context(linked))

    def test_does_not_link_unrelated_headings_or_replace_a_caption(self):
        cases = [
            (replace(self.heading, content=Text("Population findings")), self.table),
            (replace(self.heading, content=Text("Table of contents")), self.table),
            (self.heading, replace(self.table, caption_ids=["explicit"])),
            (self.heading, replace(self.table, locations=[Location(4, (0.06, 0.71, 0.94, 0.89))])),
            (self.heading, replace(self.table, locations=[Location(3, (0.65, 0.71, 0.94, 0.89))])),
        ]
        for heading, table in cases:
            with self.subTest(heading=heading, table=table):
                doc = asdict(self.document(heading, table))
                self.assertEqual(0, link_table_context(doc))
                self.assertIsNone(doc["pieces"][1].get("inferred_title_id"))

    def test_links_a_note_and_retrieves_it_with_the_table(self):
        doc = asdict(self.document(self.heading, self.table))
        note = Piece("note", "text", Text("Note: totals may differ due to rounding."),
                     [Location(3, (0.06, 0.905, 0.94, 0.92))])
        doc["pieces"].append(asdict(note))
        doc["pages"][0]["piece_ids"].append("note")
        self.assertEqual(2, link_table_context(doc))
        resolved = resolve_document(doc)
        self.assertEqual(["heading", "table", "note"], resolved["pages"][0]["pieces"][0]["source_ids"])
        self.assertEqual(["note"], doc["pieces"][1].get("inferred_note_ids", []))
        self.assertEqual([], doc["pieces"][1]["footnote_ids"])
        entries = {entry.id: entry for entry in search_entries(doc, multimodal=False)}
        self.assertEqual(["table", "note"], entries["table:rows-0"].piece_ids)
        self.assertEqual(["note", "table"], entries["note:text-0"].piece_ids)
        self.assertEqual(0, link_table_context(doc))

    def test_leaves_notes_separate_when_relationship_is_unclear(self):
        note = Piece("note", "text", Text("Note: totals may differ due to rounding."),
                     [Location(3, (0.06, 0.905, 0.94, 0.92))])
        for table, candidate in (
            (self.table, replace(note, content=Text("Ordinary prose about the table"))),
            (self.table, replace(note, locations=[Location(3, (0.65, 0.905, 0.94, 0.92))])),
            (self.table, replace(note, locations=[Location(3, (0.06, 0.95, 0.94, 0.97))])),
            (self.table, replace(note, locations=[Location(4, (0.06, 0.905, 0.94, 0.92))])),
            (replace(self.table, footnote_ids=["explicit"]), note),
        ):
            with self.subTest(table=table, note=candidate):
                doc = asdict(self.document(self.heading, table))
                doc["pieces"].append(asdict(candidate))
                doc["pages"][0]["piece_ids"].append("note")
                self.assertEqual(1, link_table_context(doc))
                self.assertEqual([], doc["pieces"][1].get("inferred_note_ids", []))
