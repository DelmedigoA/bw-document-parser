"""Compare single and batched Perplexity Decider requests.

The default run is an offline replay. OpenRouter is patched, so elapsed time
measures request count and simulated concurrency, not provider latency, and
the decisions come from each figure's caption rather than from Perplexity.
Pass ``--live`` with ``OPENROUTER_API_KEY`` set to compare real decisions and
charges. Both modes run against a temporary copy of the bundle.

    python benchmarks/decider_batching.py --bundle /path/to/bundle \
        --batch-size 8 --workers 4
    python benchmarks/decider_batching.py --bundle /path/to/bundle \
        --batch-size 8 --live

Every report states whether it is ``replay`` or ``live``. A replay is useful
for request-count and concurrency checks; only a live run can show whether
Perplexity follows the per-figure labels in a shared state.
"""

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bw_document_parser import enrichment, figure_decision, openrouter
from bw_document_parser.bundle import read_document, write_json
from bw_document_parser.document import apply_visual_decision
from bw_document_parser.resolving import resolved_pieces

INFORMATIVE = ("chart", "map", "diagram", "photo", "scan", "graph", "table")
DECORATIVE = ("logo", "icon", "watermark", "decorative", "brand")


def informative(caption: str) -> bool:
    lowered = caption.lower()
    if any(word in lowered for word in DECORATIVE):
        return False
    return any(word in lowered for word in INFORMATIVE) or not lowered.strip()


def replay_decision(payload: dict, delay: float) -> dict:
    if delay:
        time.sleep(delay)
    questions = payload["questions"]
    answers = {}
    for question_id in questions:
        context = {}
        for item in payload["state"]:
            if (
                isinstance(item, str)
                and f"Question ID: {question_id}" in item
                and "Context: " in item
            ):
                context = json.loads(item.split("Context: ", 1)[1])
                break
        else:
            if payload["state"] and isinstance(payload["state"][0], str):
                context = json.loads(payload["state"][0])
        answers[question_id] = {
            "type": "noul",
            "noul": 0.9 if informative(str(context.get("published_caption", ""))) else 0.1,
        }
    count = max(1, len(questions))
    return {
        "answers": answers,
        "usage": {"cost": 0.000005 + 0.00002 * count},
        "id": f"gen-replay-{count}",
    }


class Recorder:
    """Count Decider calls and charges, whether replayed or live."""

    def __init__(self, *, delay: float, live: bool, decide=None) -> None:
        self.delay = delay
        self.live = live
        self._decide = decide
        self._lock = threading.Lock()
        self.requests = 0
        self.inputs = 0
        self.batch_sizes: list[int] = []
        self.cost = 0.0
        self.unpriced = 0

    def decide(self, payload: dict) -> dict:
        with self._lock:
            self.requests += 1
            request_number = self.requests
            size = len(payload["questions"])
            self.inputs += size
            self.batch_sizes.append(size)
        if self.live:
            response = self._decide(payload)
        else:
            response = replay_decision(payload, self.delay)
            # The ledger treats a provider response ID as unique. Replays must
            # not reuse an ID or their charges collapse into one entry.
            response["id"] = f"gen-replay-{request_number}"
        cost = openrouter.cost_of(response)
        with self._lock:
            if cost is None:
                self.unpriced += 1
            else:
                self.cost += cost
        return response


def replay_description(payload: dict, delay: float) -> dict:
    if delay:
        time.sleep(delay)
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": json.dumps({
                "description": "Replayed figure description",
                "visible_text": None,
            })},
        }],
        "usage": {"cost": 0.00001},
    }


def clear_enrichment(bundle: Path) -> None:
    """Start each run with the same figures and an empty request ledger."""
    document = read_document(bundle)
    if document["schema_version"] != 4:
        raise ValueError("The Decider benchmark needs a resolved bundle. Run bw-document-resolve first.")
    document.pop("decider_requests", None)
    for piece in resolved_pieces(document):
        for source_id in piece["source_ids"]:
            source = document["source_blocks"][source_id]
            if source.get("excluded_visuals"):
                apply_visual_decision(piece, source, {"enrich": True})
            source.pop("enrichment_decision", None)
            source.pop("enrichment", None)
        for visual in piece["visual_part"] or ():
            visual["description"] = None
            visual["transcription"] = None
    write_json(bundle / "document.json", document)


def run(
    bundle: Path, *, batch_size: int, workers: int, delay: float, live: bool,
) -> dict:
    with TemporaryDirectory() as directory:
        copied = Path(directory) / "bundle"
        shutil.copytree(bundle, copied)
        clear_enrichment(copied)
        recorder = Recorder(
            delay=0.0 if live else delay,
            live=live,
            decide=openrouter.decide if live else None,
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(openrouter, "decide", side_effect=recorder.decide)
            )
            if not live:
                stack.enter_context(patch.object(
                    openrouter, "post",
                    side_effect=lambda _, payload: replay_description(payload, delay),
                ))
            started = time.perf_counter()
            complete = enrichment.enrich_bundle(
                copied, workers=workers, decider_batch_size=batch_size,
            )
            elapsed = time.perf_counter() - started
        document = json.loads((copied / "document.json").read_text())
        ledger = document.get("decider_requests", [])
        decisions = {
            piece_id: block["enrichment_decision"]["enrich"]
            for piece_id, block in document.get("source_blocks", {}).items()
            if "enrichment_decision" in block
        }
        described = sum(
            1 for block in document.get("source_blocks", {}).values()
            if block.get("enrichment")
        )
        return {
            "batch_size": batch_size,
            "complete": complete,
            "seconds": round(elapsed, 3),
            "decider_requests": recorder.requests,
            "decider_inputs": recorder.inputs,
            "largest_decider_batch": max(recorder.batch_sizes, default=0),
            "decider_cost": round(recorder.cost, 8),
            "unpriced_decider_responses": recorder.unpriced,
            "described_figures": described,
            "ledger_requests": len(ledger),
            "ledger_cost": round(sum(entry.get("cost") or 0 for entry in ledger), 8),
            "decisions": decisions,
        }


def compare(bundle: Path, *, batch_size: int, workers: int, delay: float, live: bool) -> dict:
    if not 1 <= batch_size <= figure_decision.MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be between 1 and {figure_decision.MAX_BATCH_SIZE}")
    single = run(bundle, batch_size=1, workers=workers, delay=delay, live=live)
    batched = run(bundle, batch_size=batch_size, workers=workers, delay=delay, live=live)
    pieces = sorted(set(single["decisions"]) | set(batched["decisions"]))
    changed = [
        {
            "piece": piece_id,
            "single": single["decisions"].get(piece_id),
            "batch": batched["decisions"].get(piece_id),
        }
        for piece_id in pieces
        if single["decisions"].get(piece_id) != batched["decisions"].get(piece_id)
    ]
    return {
        "measurement": "live" if live else "replay",
        "note": (
            "Real OpenRouter calls and charges."
            if live else
            "OpenRouter was patched. Elapsed time measures request count and "
            "simulated concurrency, not provider latency. Decisions come from "
            "captions, not from Perplexity."
        ),
        "bundle": str(bundle),
        "workers": workers,
        "delay": None if live else delay,
        "single": single,
        "batch": batched,
        "request_reduction": single["decider_requests"] - batched["decider_requests"],
        "seconds_saved": round(single["seconds"] - batched["seconds"], 3),
        "cost_difference": round(single["decider_cost"] - batched["decider_cost"], 8),
        "changed_decisions": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument(
        "--live", action="store_true",
        help="use real OpenRouter calls and charges (spends money)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.live and not os.getenv("OPENROUTER_API_KEY"):
        parser.error("--live needs OPENROUTER_API_KEY")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be a non-negative number")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            report = compare(
                args.bundle, batch_size=args.batch_size, workers=args.workers,
                delay=args.delay, live=args.live,
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(payload)
    if report["changed_decisions"]:
        print(
            f"{len(report['changed_decisions'])} decision(s) changed; "
            "review before enabling batching by default.",
            file=sys.stderr,
        )
    if any(
        not run["complete"] or not run["decider_requests"]
        for run in (report["single"], report["batch"])
    ):
        print("Decider comparison incomplete. Check the request counts and errors.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
