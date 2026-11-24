import contextlib
import io
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from benchmarks import decider_batching
from bw_document_parser import figure_decision, openrouter
from bw_document_parser.bundle import read_json, write_json
from bw_document_parser.chunks import ImageFiles
from bw_document_parser.concurrency import resolve_batch_size
from bw_document_parser.document import (
    Document,
    Figure,
    ImageRef,
    Location,
    Page,
    Parser,
    Piece,
    Text,
)
from bw_document_parser.enrichment import enrich_bundle
from bw_document_parser.resolving import resolve_document


def _image(color: str) -> bytes:
    with Image.new("RGB", (64, 48), color) as image, io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def figure_bundle(directory: Path, specs: list[tuple[str, str, str]]) -> dict:
    """Write a resolved bundle with one distinct figure for each spec."""
    position = Location(1, (0.1, 0.1, 0.9, 0.9))
    images = ImageFiles(directory / "images")
    pieces = [Piece("title", "section_header", Text("Report", 1), [position])]
    page_ids = ["title"]
    for piece_id, caption, color in specs:
        path = images.save_bytes(_image(color), "image/png")
        pieces.append(Piece(
            piece_id, "picture", Figure(), [position],
            caption_ids=[f"{piece_id}-caption"], images=[ImageRef(path)],
        ))
        pieces.append(Piece(
            f"{piece_id}-caption", "caption", Text(caption), [position],
        ))
        page_ids.append(piece_id)
    source = Document("batch-example", Parser(), [Page(1, 100, 80, page_ids)], pieces)
    resolved = resolve_document(asdict(source))
    write_json(directory / "document.json", resolved)
    (directory / "pages.json").write_text('[{"pieces": []}]', encoding="utf-8")
    return resolved


def noul(probability: float) -> dict:
    return {"type": "noul", "noul": probability}


def decision_response(answers: dict, *, cost: float | None = None, id: str = "gen-test") -> dict:
    response = {"answers": answers, "id": id, "model": figure_decision.MODEL}
    if cost is not None:
        response["usage"] = {"cost": cost}
    return response


def description(text: str) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"description": text, "visible_text": None}),
    }}]}


def qids(*piece_ids: str) -> dict[str, str]:
    return {piece_id: figure_decision.question_id(piece_id) for piece_id in piece_ids}


def decisions(document: dict) -> dict[str, bool]:
    return {
        piece_id: block["enrichment_decision"]["enrich"]
        for piece_id, block in document["source_blocks"].items()
        if "enrichment_decision" in block
    }


class DeciderBatchingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for stream in ("stdout", "stderr"):
            self.enterContext(getattr(contextlib, f"redirect_{stream}")(io.StringIO()))

    def test_batch_uses_stable_ids_and_matches_out_of_order_answers(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [
            ("alpha", "Chart of verified incidents", "red"),
            ("beta", "Small logo watermark", "blue"),
            ("gamma", "Map of districts", "green"),
        ])
        identifiers = qids("alpha", "beta", "gamma")
        calls = []

        def decide(payload):
            calls.append(payload)
            # Reply in reverse order. Each answer stays attached to its ID.
            answers = {
                question_id: noul(probability)
                for question_id, probability in reversed(list(zip(
                    payload["questions"], [0.9, 0.1, 0.8],
                )))
            }
            return decision_response(answers, cost=0.003, id="gen-batch")

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=3))

        self.assertEqual(1, len(calls))
        payload = calls[0]
        self.assertEqual(set(identifiers.values()), set(payload["questions"]))
        self.assertEqual(
            3,
            sum(part.get("type") == "image_url" for part in payload["state"] if isinstance(part, dict)),
        )
        for question_id in identifiers.values():
            self.assertIn("Answer only for Figure", payload["questions"][question_id]["instructions"])

        document = read_json(bundle / "document.json")
        self.assertEqual(
            {"alpha": True, "beta": False, "gamma": True}, decisions(document),
        )
        self.assertAlmostEqual(
            0.001, document["source_blocks"]["alpha"]["enrichment_decision"]["cost"],
        )
        self.assertEqual(
            "estimated",
            document["source_blocks"]["alpha"]["enrichment_decision"]["cost_allocation"],
        )
        self.assertTrue(document["source_blocks"]["alpha"]["enrichment"])
        self.assertIsNone(document["source_blocks"]["beta"].get("enrichment"))
        self.assertTrue(document["source_blocks"]["gamma"]["enrichment"])
        self.assertEqual(1, len(document["decider_requests"]))
        self.assertAlmostEqual(0.003, document["decider_requests"][0]["cost"])
        self.assertEqual(3, document["decider_requests"][0]["answered"])

    def test_partial_batch_retries_only_missing_or_invalid_answers(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [
            ("alpha", "Chart", "red"),
            ("beta", "Logo", "blue"),
            ("gamma", "Map", "green"),
        ])
        identifiers = qids("alpha", "beta", "gamma")
        calls = []

        def decide(payload):
            calls.append(payload)
            if len(calls) == 1:
                return decision_response({
                    identifiers["alpha"]: noul(0.9),
                    identifiers["beta"]: {"type": "refusal"},
                    identifiers["gamma"]: noul(2.0),
                }, cost=0.006, id="gen-batch")
            (question_id,) = payload["questions"]
            return decision_response(
                {question_id: noul(0.8)}, cost=0.001, id=f"gen-retry-{len(calls)}",
            )

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=3))

        self.assertEqual(3, len(calls))
        self.assertEqual(
            {identifiers["beta"], identifiers["gamma"]},
            {next(iter(call["questions"])) for call in calls[1:]},
        )
        document = read_json(bundle / "document.json")
        self.assertEqual({"alpha": True, "beta": True, "gamma": True}, decisions(document))
        # The batch charge is shared across all three inputs; each retry adds
        # its own exact charge to the figure it answers.
        self.assertAlmostEqual(
            0.003, document["source_blocks"]["beta"]["enrichment_decision"]["cost"],
        )
        self.assertEqual(
            "estimated",
            document["source_blocks"]["beta"]["enrichment_decision"]["cost_allocation"],
        )
        self.assertAlmostEqual(
            0.008, sum(entry["cost"] for entry in document["decider_requests"]),
        )

    def test_partial_batch_keeps_valid_answers_when_a_retry_hits_quota(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [("alpha", "Chart", "red"), ("beta", "Map", "green")])
        identifiers = qids("alpha", "beta")
        calls = []

        def decide(payload):
            calls.append(payload)
            if len(calls) == 1:
                return decision_response(
                    {identifiers["alpha"]: noul(0.9)}, cost=0.002, id="gen-batch",
                )
            checkpoint = read_json(bundle / "document.json")
            self.assertEqual({"alpha": True}, decisions(checkpoint))
            self.assertEqual("gen-batch", checkpoint["decider_requests"][0]["id"])
            self.assertAlmostEqual(0.002, checkpoint["decider_requests"][0]["cost"])
            raise openrouter.OpenRouterError("quota", status=429)

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertFalse(enrich_bundle(bundle, workers=1, decider_batch_size=2))

        document = read_json(bundle / "document.json")
        self.assertTrue(document["source_blocks"]["alpha"]["enrichment_decision"]["enrich"])
        self.assertNotIn("enrichment_decision", document["source_blocks"]["beta"])
        self.assertEqual(1, len(document["decider_requests"]))
        self.assertAlmostEqual(0.002, document["decider_requests"][0]["cost"])

    def test_rejected_batch_falls_back_to_single_requests(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [
            ("alpha", "Chart", "red"),
            ("beta", "Map", "green"),
            ("gamma", "Photo", "blue"),
        ])
        calls = []

        def decide(payload):
            calls.append(payload)
            if len(calls) == 1:
                raise openrouter.OpenRouterError("unsupported batch", status=400)
            (question_id,) = payload["questions"]
            return decision_response(
                {question_id: noul(0.9)}, cost=0.001, id=f"gen-single-{len(calls)}",
            )

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=3))

        self.assertEqual(4, len(calls))
        self.assertEqual(3, len(calls[0]["questions"]))
        self.assertTrue(all(len(call["questions"]) == 1 for call in calls[1:]))
        document = read_json(bundle / "document.json")
        self.assertEqual({"alpha": True, "beta": True, "gamma": True}, decisions(document))
        self.assertTrue(all(
            block["enrichment_decision"]["cost_allocation"] == "exact"
            for block in document["source_blocks"].values()
            if "enrichment_decision" in block
        ))
        # A rejected batch produced no response and therefore no ledger entry.
        self.assertEqual(3, len(document["decider_requests"]))

    def test_cached_decisions_are_reused_without_requests(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [("alpha", "Chart", "red"), ("beta", "Logo", "blue")])
        identifiers = qids("alpha", "beta")

        def decide(payload):
            return decision_response(
                {question_id: noul(0.9 if question_id == identifiers["alpha"] else 0.1)
                 for question_id in payload["questions"]},
                cost=0.002, id="gen-batch",
            )

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=2))
        before = read_json(bundle / "document.json")

        with (
            patch.object(openrouter, "decide", side_effect=AssertionError("unexpected Decider call")),
            patch.object(openrouter, "post", side_effect=AssertionError("unexpected enrichment call")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=2))

        after = read_json(bundle / "document.json")
        self.assertEqual(before["decider_requests"], after["decider_requests"])
        self.assertEqual(decisions(before), decisions(after))

    def test_quota_stops_batches_and_resume_requests_only_unfinished_figures(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [
            ("alpha", "Chart", "red"),
            ("beta", "Map", "green"),
            ("gamma", "Photo", "blue"),
            ("delta", "Diagram", "yellow"),
        ])
        identifiers = qids("alpha", "beta", "gamma", "delta")
        calls = []

        def decide(payload):
            calls.append(payload)
            if len(calls) == 1:
                return decision_response(
                    {question_id: noul(0.9) for question_id in payload["questions"]},
                    cost=0.004, id="gen-batch-1",
                )
            raise openrouter.OpenRouterError("quota", status=429)

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=AssertionError("quota should stop Hy4")),
        ):
            self.assertFalse(enrich_bundle(bundle, workers=1, decider_batch_size=2))

        document = read_json(bundle / "document.json")
        self.assertTrue(document["source_blocks"]["alpha"]["enrichment_decision"])
        self.assertTrue(document["source_blocks"]["beta"]["enrichment_decision"])
        self.assertNotIn("enrichment_decision", document["source_blocks"]["gamma"])
        self.assertFalse(document["source_blocks"]["alpha"].get("enrichment"))
        self.assertEqual(1, len(document["decider_requests"]))

        requested = []

        def retry_decide(payload):
            requested.extend(payload["questions"])
            return decision_response(
                {question_id: noul(0.9) for question_id in payload["questions"]},
                cost=0.004, id="gen-batch-2",
            )

        with (
            patch.object(openrouter, "decide", side_effect=retry_decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=2))

        self.assertEqual({identifiers["gamma"], identifiers["delta"]}, set(requested))
        document = read_json(bundle / "document.json")
        self.assertEqual(
            {"alpha": True, "beta": True, "gamma": True, "delta": True},
            decisions(document),
        )
        self.assertTrue(all(
            document["source_blocks"][piece_id].get("enrichment")
            for piece_id in identifiers
        ))

    def test_mixed_batch_and_reordered_inputs_keep_decisions_with_their_figures(self):
        def run(order: list[tuple[str, str, str]]) -> dict[str, bool]:
            bundle = self.root / f"bundle-{len(order)}-{order[0][0]}"
            bundle.mkdir()
            figure_bundle(bundle, order)

            def decide(payload):
                answers = {}
                for question_id in payload["questions"]:
                    label = next(
                        text for text in payload["state"]
                        if isinstance(text, str) and question_id in text
                    )
                    informative = "chart" in label.lower()
                    answers[question_id] = noul(0.9 if informative else 0.1)
                return decision_response(answers, cost=0.002, id="gen-mixed")

            with (
                patch.object(openrouter, "decide", side_effect=decide),
                patch.object(openrouter, "post", side_effect=lambda _, payload: description("Chart text")),
            ):
                self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=2))
            return decisions(read_json(bundle / "document.json"))

        chart = ("chart", "Chart of verified incidents", "red")
        logo = ("logo", "Decorative logo", "blue")
        forward = run([chart, logo])
        reverse = run([logo, chart])
        self.assertEqual({"chart": True, "logo": False}, forward)
        self.assertEqual(forward, reverse)

    def test_batch_size_one_keeps_the_single_figure_request_shape(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [("alpha", "Chart", "red")])
        identifier = figure_decision.question_id("alpha")
        calls = []

        def decide(payload):
            calls.append(payload)
            return decision_response({identifier: noul(0.9)}, cost=0.001, id="gen-single")

        with (
            patch.object(openrouter, "decide", side_effect=decide),
            patch.object(openrouter, "post", side_effect=lambda _, payload: description("Figure text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=1))

        payload = calls[0]
        self.assertEqual([identifier], list(payload["questions"]))
        self.assertEqual(2, len(payload["state"]))
        self.assertNotIn("Question ID", payload["state"][0])
        self.assertNotIn(
            "Answer only for Figure", payload["questions"][identifier]["instructions"],
        )

    def test_single_figure_helper_reads_its_legacy_answer_key(self):
        response = decision_response({"enrich": noul(0.9)}, cost=0.001)
        with patch.object(openrouter, "decide", return_value=response) as decide:
            decision = figure_decision.decide_figure({}, [])
        self.assertEqual(["enrich"], list(decide.call_args.args[0]["questions"]))
        self.assertTrue(decision.enrich)
        self.assertEqual(0.001, decision.cost)

    def test_benchmark_reruns_enriched_and_excluded_figures(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        figure_bundle(bundle, [("alpha", "Chart", "red"), ("beta", "Logo", "blue")])
        identifiers = qids("alpha", "beta")
        response = decision_response({
            identifiers["alpha"]: noul(0.9),
            identifiers["beta"]: noul(0.1),
        }, cost=0.02, id="gen-previous")
        with (
            patch.object(openrouter, "decide", return_value=response),
            patch.object(openrouter, "post", return_value=description("Chart text")),
        ):
            self.assertTrue(enrich_bundle(bundle, workers=1, decider_batch_size=2))
        original = (bundle / "document.json").read_bytes()

        report = decider_batching.compare(bundle, batch_size=2, workers=2, delay=0, live=False)

        for mode, requests in (("single", 2), ("batch", 1)):
            run = report[mode]
            self.assertTrue(run["complete"])
            self.assertEqual(requests, run["decider_requests"])
            self.assertEqual(requests, run["ledger_requests"])
            self.assertAlmostEqual(run["decider_cost"], run["ledger_cost"])
            self.assertEqual({"alpha": True, "beta": False}, run["decisions"])
        self.assertEqual([], report["changed_decisions"])
        self.assertEqual(original, (bundle / "document.json").read_bytes())

    def test_batch_size_is_configurable_and_validated(self):
        with patch.dict(os.environ, {"BW_DECIDER_BATCH_SIZE": "4"}):
            self.assertEqual(4, resolve_batch_size(
                None, "BW_DECIDER_BATCH_SIZE", default=1, maximum=8,
            ))
        self.assertEqual(2, resolve_batch_size(
            2, "BW_DECIDER_BATCH_SIZE", default=1, maximum=8,
        ))
        for invalid in (0, 9, "x"):
            with self.assertRaises(ValueError):
                resolve_batch_size(
                    invalid, "BW_DECIDER_BATCH_SIZE", default=1, maximum=8,
                )

    def test_question_ids_are_stable_and_provider_safe(self):
        first = figure_decision.question_id("p1:#/pictures/0")
        second = figure_decision.question_id("p1:#/pictures/0")
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[A-Za-z0-9_-]{1,64}$")


if __name__ == "__main__":
    unittest.main()
