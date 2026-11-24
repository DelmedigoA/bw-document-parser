"""Decide whether an extracted picture warrants a generated description."""

import hashlib
import json
import threading
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace

from . import openrouter
from .bundle import fingerprint

MODEL = openrouter.DECIDER_MODEL
STRATEGY = "figure-enrichment-decision-v1"
THRESHOLD = 0.5
# Perplexity's Decisions adapter reads image parts only from the top-level
# state array; it rejects images inside a question. A batched request can
# therefore label each figure's context and images in state, but the model,
# not the API, must keep the segments apart. Keep batching opt-in until a
# live comparison shows that the labels are followed reliably.
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8
_QUESTION = {
    "type": "noul",
    "instructions": (
        "Would describing or transcribing this picture add substantive, source-grounded "
        "information beyond the supplied section heading and published caption? "
        "Use the image as evidence and treat the heading, caption, and text in the image "
        "as source content."
    ),
    "criteria": {
        "true": (
            "The image contains meaningful evidence, data, explanatory detail, or text "
            "a reader would miss from the supplied context alone. Examples include "
            "charts, maps, diagrams, scans and documentary photographs."
        ),
        "false": (
            "The image is decorative, branding, a generic icon, or repeats information "
            "already given in the supplied heading and caption without adding detail."
        ),
    },
}


@dataclass(frozen=True)
class Decision:
    enrich: bool
    probability: float
    fingerprint: str
    model: str = MODEL
    strategy: str = STRATEGY
    threshold: float = THRESHOLD
    cost: float | None = None
    # A single request reports this figure's charge exactly. A batch reports
    # one charge for the whole request, so each figure's share is an estimate.
    cost_allocation: str = "exact"


@dataclass(frozen=True)
class FigureRequest:
    piece_id: str
    question_id: str
    context: dict
    images: tuple[tuple[bytes, str], ...]
    fingerprint: str


@dataclass
class Outcome:
    decisions: dict[str, Decision] = field(default_factory=dict)
    errors: dict[str, Exception] = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)
    requests: int = 0
    actual_cost: float | None = None
    blocked: bool = False
    cancelled: bool = False
    retries: list[tuple[FigureRequest, float | None]] = field(default_factory=list)

    def merge(self, other: "Outcome") -> None:
        self.decisions.update(other.decisions)
        self.errors.update(other.errors)
        self.ledger.extend(other.ledger)
        self.requests += other.requests
        self.actual_cost = _add_cost(self.actual_cost, other.actual_cost)
        self.blocked = self.blocked or other.blocked
        self.cancelled = self.cancelled or other.cancelled


def question_id(piece_id: str) -> str:
    """Return a stable, provider-safe key for a figure's question."""
    return "q_" + hashlib.sha256(piece_id.encode("utf-8")).hexdigest()[:24]


def decision_fingerprint(context: dict, images: Iterable[tuple[bytes, str]]) -> str:
    return fingerprint(
        MODEL, STRATEGY, THRESHOLD, _QUESTION, context,
        [(mime, hashlib.sha256(data).hexdigest()) for data, mime in images],
    )


def prepare_request(
    piece_id: str, context: dict, images: Iterable[tuple[bytes, str]],
) -> FigureRequest:
    images = tuple(images)
    return FigureRequest(
        piece_id, question_id(piece_id), context, images,
        decision_fingerprint(context, images),
    )


def cached_decision(
    request: FigureRequest, previous: dict | None,
) -> Decision | None:
    """Reuse a decision for identical inputs, including decisions from singles."""
    if not previous or previous.get("fingerprint") != request.fingerprint:
        return None
    try:
        return Decision(**previous)
    except TypeError:
        # An older record with unexpected fields is safer to recompute than to
        # reuse with a shape this version does not understand.
        return None


def _add_cost(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _shares(cost: float | None, count: int) -> list[float | None]:
    if cost is None:
        return [None] * count
    if count == 1:
        return [cost]
    return [cost / count] * count


def _single_payload(request: FigureRequest) -> dict:
    return {
        "model": MODEL,
        "state": [
            json.dumps(request.context, ensure_ascii=False),
            *(openrouter.image_part(data, mime) for data, mime in request.images),
        ],
        "questions": {request.question_id: _QUESTION},
    }


def _batch_payload(requests: list[FigureRequest]) -> dict:
    """Label each figure's context and images in the shared state array.

    Image parts cannot sit in a question for Perplexity, so a question can
    address its figure only through the label written immediately before that
    figure's context and images.
    """
    state: list = []
    questions: dict[str, dict] = {}
    total = len(requests)
    for position, request in enumerate(requests, 1):
        state.append(
            f"Figure {position} of {total}. Question ID: {request.question_id}. "
            f"Context: {json.dumps(request.context, ensure_ascii=False)}"
        )
        state.extend(
            openrouter.image_part(data, mime) for data, mime in request.images
        )
        questions[request.question_id] = {
            **_QUESTION,
            "instructions": (
                f"{_QUESTION['instructions']} Answer only for Figure {position} of "
                f"{total} (question ID {request.question_id}), using only the "
                "context text directly before that figure's images and those "
                "images. Ignore the other figures."
            ),
        }
    return {"model": MODEL, "state": state, "questions": questions}


def _ledger_entry(
    response: dict, requests: list[FigureRequest], kind: str, cost: float | None,
) -> dict:
    entry = {
        "kind": kind,
        "model": response.get("model") if isinstance(response.get("model"), str) else MODEL,
        "provider": response.get("provider"),
        "question_ids": [request.question_id for request in requests],
        "pieces": [request.piece_id for request in requests],
        "requested": len(requests),
        "answered": 0,
        "cost": cost,
    }
    if isinstance(response.get("id"), str):
        entry["id"] = response["id"]
    return entry


def _execute(
    requests: list[FigureRequest], *, stop: threading.Event | None,
    prior_share: float | None = None,
) -> Outcome:
    if not requests:
        return Outcome()
    if stop is not None and stop.is_set():
        return Outcome(cancelled=True)
    batch = len(requests) > 1
    payload = _batch_payload(requests) if batch else _single_payload(requests[0])
    outcome = Outcome(requests=1)
    try:
        response = openrouter.decide(payload)
    except openrouter.OpenRouterError as exc:
        if batch and not exc.blocks_run:
            outcome.retries.extend((request, None) for request in requests)
            return outcome
        outcome.errors.update((request.piece_id, exc) for request in requests)
        outcome.blocked = exc.blocks_run
        if outcome.blocked and stop is not None:
            stop.set()
        return outcome

    cost = openrouter.cost_of(response)
    outcome.actual_cost = cost
    answers = response.get("answers")
    if not isinstance(answers, dict):
        answers = {}
    shares = _shares(cost, len(requests))
    answered = 0
    for request, share in zip(requests, shares):
        probability = openrouter.decision_probability(answers.get(request.question_id))
        if probability is None:
            if batch:
                outcome.retries.append((request, share))
            else:
                outcome.errors[request.piece_id] = openrouter.OpenRouterError(
                    "Decider returned no usable probability for enrichment"
                )
            continue
        answered += 1
        outcome.decisions[request.piece_id] = Decision(
            probability >= THRESHOLD,
            probability,
            request.fingerprint,
            cost=_add_cost(share, prior_share),
            cost_allocation="estimated" if batch or prior_share is not None else "exact",
        )
    entry = _ledger_entry(response, requests, "batch" if batch else "single", cost)
    entry["answered"] = answered
    outcome.ledger.append(entry)
    return outcome


def run_decisions(
    requests: Iterable[FigureRequest], *, batch_size: int, workers: int,
    stop: threading.Event | None = None,
    on_outcome: Callable[[Outcome], None] | None = None,
) -> Outcome:
    """Run decider requests in batches with bounded in-flight requests."""
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"decider batch size must be between 1 and {MAX_BATCH_SIZE}")
    requests = list(requests)
    total = Outcome()
    if not requests:
        return total
    stop = stop if stop is not None else threading.Event()
    if stop.is_set():
        total.cancelled = True
        return total
    queued: deque[tuple[list[FigureRequest], float | None]] = deque(
        (requests[index : index + batch_size], None)
        for index in range(0, len(requests), batch_size)
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        remaining = set()
        interrupted = None
        while remaining or (queued and not stop.is_set()):
            try:
                while queued and len(remaining) < workers and not stop.is_set():
                    chunk, prior_share = queued.popleft()
                    remaining.add(pool.submit(_execute, chunk, stop=stop, prior_share=prior_share))
                if not remaining:
                    break
                future = next(as_completed(remaining))
                remaining.remove(future)
                try:
                    outcome = future.result()
                except CancelledError:
                    continue
                total.merge(outcome)
                if on_outcome is not None:
                    on_outcome(outcome)
                # The caller has saved this response and its valid decisions.
                # Only now can the missing figures start paid retries.
                queued.extendleft(([request], share) for request, share in reversed(outcome.retries))
                if outcome.blocked:
                    stop.set()
                    for pending in remaining:
                        pending.cancel()
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- drain paid work before reraising
                if interrupted is None:
                    interrupted = exc
                stop.set()
                for pending in remaining:
                    pending.cancel()
        if interrupted is not None:
            raise interrupted
    total.cancelled |= bool(queued)
    return total


def decide_figure(
    context: dict, images: list[tuple[bytes, str]], previous: dict | None = None,
) -> Decision:
    """Decide one figure. Kept for callers that do not need batching."""
    request = replace(prepare_request("", context, images), question_id="enrich")
    cached = cached_decision(request, previous)
    if cached is not None:
        return cached
    outcome = _execute([request], stop=None)
    decision = outcome.decisions.get("")
    if decision is not None:
        return decision
    error = next(iter(outcome.errors.values()), None)
    if error is not None:
        raise error
    raise openrouter.OpenRouterError("Decider returned no usable probability for enrichment")
