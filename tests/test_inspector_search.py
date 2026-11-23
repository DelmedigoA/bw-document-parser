import json
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch

from bw_document_parser.inspector import _handler

from test_rag_bundle import example_bundle


class InspectorSearchTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        bundle = Path(temporary.name)
        example_bundle(bundle)
        (bundle / "embeddings.multimodal.json").write_text("{}", encoding="utf-8")
        self.bundle = bundle
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(bundle / "example.pdf", bundle))
        self.addCleanup(self.server.server_close)
        thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(self.server.shutdown)
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        self.addCleanup(self.connection.close)

    def request(self, payload, *, same_origin=True):
        origin = f"http://127.0.0.1:{self.server.server_port}" if same_origin else "https://example.org"
        self.connection.request(
            "POST", "/api/search", json.dumps(payload),
            {"Content-Type": "application/json", "Origin": origin},
        )
        response = self.connection.getresponse()
        return response.status, response.read()

    def test_dispatch_and_result_source(self):
        hit = {
            "pieces": [{
                "id": "p1:figure", "images": [{"path": "images/figure.png"}],
                "content": {"kind": "figure"}, "enrichment": {"description": "Chart of incidents"},
            }],
            "image_path": None, "locations": [{"page": 1}], "text": "index text",
        }
        with patch("bw_document_parser.inspector.search_bundle", return_value=[hit]) as search:
            status, response = self.request({"query": "chart", "mode": "multimodal", "figuresOnly": True})
            search.assert_called_once_with(self.bundle, "chart", "multimodal", limit=10, figures_only=True)
        self.assertEqual(200, status)
        self.assertEqual([{
            "pieceId": "p1:figure", "kind": "figure", "page": 1,
            "image": "images/figure.png", "text": "Chart of incidents",
        }], json.loads(response)["results"])

    def test_invalid_and_cross_origin_requests_do_not_submit_search(self):
        with patch("bw_document_parser.inspector.search_bundle") as search:
            self.assertEqual(403, self.request({"query": "chart", "mode": "multimodal"}, same_origin=False)[0])
            self.assertEqual(400, self.request({"query": "chart", "mode": "missing"})[0])
            search.assert_not_called()
