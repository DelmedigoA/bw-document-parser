import json
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pypdfium2
from PIL import Image

from bw_document_parser import cli, language_decision, openrouter, parsing

EXAMPLE_PDF = next((Path(__file__).resolve().parents[1] / "examples").glob("*.pdf"))
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def response(probability: float) -> dict:
    return {"answers": {"is_english": {"type": "noul", "noul": probability}}}


class PdfLanguageTests(unittest.TestCase):
    def test_sample_uses_native_text_and_renders_from_first_three_pages(self):
        with closing(pypdfium2.PdfDocument(str(EXAMPLE_PDF))) as pdf:
            self.assertGreater(len(pdf), 3)
            sample = language_decision._prepare_for_decider(pdf)
            expected_texts = []
            for number in range(3):
                with closing(pdf[number]) as page, closing(page.get_textpage()) as text_page:
                    expected_texts.append(text_page.get_text_bounded().strip())

        self.assertTrue(all(expected_texts))
        self.assertEqual(expected_texts, sample.texts)
        self.assertEqual(3, len(sample.images))
        self.assertTrue(all(image.startswith(PNG_SIGNATURE) for image in sample.images))

    def test_one_page_image_only_pdf_sends_a_render_without_native_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            with Image.new("RGB", (120, 160), "white") as image:
                image.save(path)
            with closing(pypdfium2.PdfDocument(str(path))) as pdf:
                sample = language_decision._prepare_for_decider(pdf)

        with patch.object(openrouter, "decide", return_value=response(0.9)) as decide:
            self.assertTrue(language_decision.is_english(sample))
        payload, = decide.call_args.args
        self.assertEqual(openrouter.DECIDER_MODEL, payload["model"])
        text, image = payload["state"]
        self.assertIsNone(json.loads(text)["page_texts"])
        self.assertTrue(image["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_non_english_skips_parsing_even_for_a_later_page_range(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            with (
                patch.object(openrouter, "decide", return_value=response(0.1)) as decide,
                self.assertRaisesRegex(NotImplementedError, "supports English"),
            ):
                cli._write_bundle(EXAMPLE_PDF, bundle, 12, page_range=(4, 4))
            self.assertFalse(bundle.exists())
            self.assertEqual([], list(Path(directory).iterdir()))
        payload, = decide.call_args.args
        self.assertEqual(3, len(json.loads(payload["state"][0])["page_texts"]))
        self.assertEqual(4, len(payload["state"]))

    def test_english_continues_to_parsing(self):
        with (
            patch.object(openrouter, "decide", return_value=response(0.9)),
            patch.object(parsing, "ImageFiles", side_effect=RuntimeError("parsing started")),
            self.assertRaisesRegex(RuntimeError, "parsing started"),
        ):
            parsing.parse_pdf(EXAMPLE_PDF, Path("unused-bundle"))

    def test_invalid_response_is_an_api_error(self):
        with (
            patch.object(openrouter, "decide", return_value={"answers": {}}),
            self.assertRaisesRegex(openrouter.OpenRouterError, "no usable"),
        ):
            parsing.parse_pdf(EXAMPLE_PDF, Path("unused-bundle"))
