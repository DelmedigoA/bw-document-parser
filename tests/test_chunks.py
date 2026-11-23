from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from bw_document_parser.chunks import MAX_TOKENS, ImageFiles, _token_count, parse_page


class ChunkTests(TestCase):
    def test_reading_order_tables_captions_and_blank_pages(self):
        page = parse_page(
            '<div class="page"><h2>Results</h2><p>Before</p>'
            "<table><tr><th>Item</th><td>Value</td></tr></table>"
            '<figure><img src="figure.png"><figcaption>Inside</figcaption></figure>'
            '<div class="caption">Outside</div><details>Hidden</details>'
            "<h2>Next</h2><ul><li>One</li><li>Two</li></ul></div>",
            str,
        )
        self.assertEqual(
            [
                "## Results\n\nBefore\n\nItem | Value\n\n[image:0]\nCaption: Inside\nCaption: Outside",
                "## Next\n\n- One\n- Two",
            ],
            [piece.text for piece in page.pieces],
        )
        self.assertEqual(["figure.png"], page.pieces[0].image_urls)
        self.assertEqual(
            [], parse_page("<body><div class='page'></div></body>", str).pieces
        )

    def test_tail_merge_keeps_relative_image_indexes_and_one_heading(self):
        text = "word " * 390
        tail = "tail " * 25
        page = parse_page(
            f'<h2>Section</h2><figure><img src="first"></figure><p>{text}</p>'
            f'<figure><img src="second"><figcaption>{tail}</figcaption></figure>',
            str,
        )
        self.assertEqual(1, len(page.pieces))
        piece = page.pieces[0]
        self.assertEqual(1, piece.text.count("## Section"))
        self.assertIn("[image:0]", piece.text)
        self.assertIn("[image:1]", piece.text)
        self.assertEqual(["first", "second"], piece.image_urls)

    def test_long_multilingual_text_preserves_characters_and_heading(self):
        content = "עברית中文🙂" * 250
        page = parse_page(f"<h1>Heading</h1><p>{content}</p>", str)
        self.assertGreater(len(page.pieces), 1)
        self.assertTrue(
            all(piece.text.startswith("# Heading\n\n") for piece in page.pieces)
        )
        self.assertTrue(
            all(_token_count(piece.text) <= MAX_TOKENS for piece in page.pieces)
        )
        self.assertEqual(
            content,
            "".join(piece.text.removeprefix("# Heading\n\n") for piece in page.pieces),
        )

    def test_large_table_stays_intact(self):
        cell = "word " * 900
        page = parse_page(
            f"<h2>Heading</h2><table><tr><td>{cell}</td></tr></table>", str
        )
        self.assertEqual(1, len(page.pieces))
        self.assertGreater(_token_count(page.pieces[0].text), MAX_TOKENS)

    def test_embedded_images_are_deduplicated_across_pages(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            images = ImageFiles(output / "images")
            base64_html = '<img src="data:image/png;base64,ZXhhbXBsZQ==">'
            encoded_html = '<img src="data:image/png,%65xample">'
            first = parse_page(base64_html, images.save).pieces[0]
            second = parse_page(encoded_html, images.save).pieces[0]
            self.assertEqual(first, second)
            self.assertEqual("[image:0]", first.text)
            self.assertEqual(b"example", (output / first.image_urls[0]).read_bytes())
            self.assertEqual(1, len(list((output / "images").iterdir())))
