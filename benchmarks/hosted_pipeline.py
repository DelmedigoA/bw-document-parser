"""Benchmark hosted enrichment and embeddings with replayed responses.

Run once against the baseline package and once against the working tree:

    PYTHONPATH=/path/to/baseline/src python benchmarks/hosted_pipeline.py \
        --bundle /path/to/bundle --label baseline
    PYTHONPATH=/path/to/worktree/src python benchmarks/hosted_pipeline.py \
        --bundle /path/to/bundle --label new --workers 4 --batch-size 64

Every ``openrouter.post`` call sleeps for ``--delay`` seconds and returns a
deterministic response, so elapsed time measures concurrency and request count
rather than network or model speed. Each stage runs against a fresh copy of
the bundle.
"""

import argparse
import json
import math
import shutil
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bw_document_parser import embeddings, enrichment, openrouter


def _dimensions(model: str) -> int:
    if model == embeddings.TITAN:
        return 1024
    if model == embeddings.TEXT_MODEL:
        return 1536
    return 4096


def _vector(index: int, dimensions: int) -> list[float]:
    return [float(1 + index)] + [0.0] * (dimensions - 1)


class Replay:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.requests = 0
        self.inputs = 0
        self.batch_sizes: list[int] = []

    def post(self, endpoint: str, payload: dict) -> dict:
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.requests += 1
        if endpoint == "chat/completions":
            with self.lock:
                self.inputs += 1
            content = {
                "decorative": False,
                "description": "Replayed figure description",
                "visible_text": None,
            }
            return {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(content)},
                }],
                "usage": {"cost": 0.00001},
            }
        values = payload["input"]
        if not isinstance(values, list):
            values = [values]
        dimensions = _dimensions(payload["model"])
        with self.lock:
            self.inputs += len(values)
            self.batch_sizes.append(len(values))
        return {
            "data": [
                {"index": index, "embedding": _vector(index, dimensions)}
                for index in range(len(values))
            ],
            "usage": {"cost": 0.0001 * len(values)},
        }

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "inputs": self.inputs,
            "largest_batch": max(self.batch_sizes, default=0),
        }


def _copy_bundle(source: Path, root: Path) -> Path:
    target = root / source.name
    shutil.copytree(source, target)
    return target


def _run(label: str, bundle: Path, replay: Replay, args) -> dict:
    stages = {}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        enriched = _copy_bundle(bundle, root / "enrich")
        replay.requests = replay.inputs = 0
        replay.batch_sizes = []
        started = time.perf_counter()
        with patch.object(openrouter, "post", side_effect=replay.post):
            try:
                enrichment.enrich_bundle(enriched, workers=args.workers)
            except TypeError:
                enrichment.enrich_bundle(enriched)
        stages["enrichment"] = {
            "seconds": round(time.perf_counter() - started, 3),
            **replay.summary(),
        }

        for mode in ("text", "multimodal"):
            copied = _copy_bundle(enriched, root / mode)
            replay.requests = replay.inputs = 0
            replay.batch_sizes = []
            started = time.perf_counter()
            with patch.object(openrouter, "post", side_effect=replay.post):
                try:
                    embeddings.embed_bundle(
                        copied, mode,
                        workers=args.workers, batch_size=args.batch_size,
                    )
                except TypeError:
                    embeddings.embed_bundle(copied, mode)
            stages[mode] = {
                "seconds": round(time.perf_counter() - started, 3),
                **replay.summary(),
            }
            index = json.loads((copied / f"embeddings.{mode}.json").read_text())
            stages[mode]["entries"] = len(index["entries"])
            stages[mode]["complete"] = index["complete"]
            stages[mode]["billed"] = index.get("billing", {}).get("total_cost")
    return {"label": label, "delay": args.delay, **stages}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--label", default="run")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be a non-negative number")
    replay = Replay(args.delay)
    print(json.dumps(_run(args.label, args.bundle, replay, args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
