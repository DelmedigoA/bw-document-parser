"""Decide whether an extracted picture warrants a generated description."""

import hashlib
import json
from dataclasses import dataclass

from . import openrouter
from .bundle import fingerprint

MODEL = "perplexity/pplx-decider-v1.2"
STRATEGY = "figure-enrichment-decision-v1"
THRESHOLD = 0.5
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


def decide_figure(
    context: dict, images: list[tuple[bytes, str]], previous: dict | None = None,
) -> Decision:
    identity = fingerprint(
        MODEL, STRATEGY, THRESHOLD, _QUESTION, context,
        [(mime, hashlib.sha256(data).hexdigest()) for data, mime in images],
    )
    if previous and previous.get("fingerprint") == identity:
        return Decision(**previous)
    response = openrouter.decide({
        "model": MODEL,
        "state": [
            json.dumps(context, ensure_ascii=False),
            *(openrouter.image_part(data, mime) for data, mime in images),
        ],
        "questions": {"enrich": _QUESTION},
    })
    try:
        answer = response["answers"]["enrich"]
        probability = answer["noul"]
        if answer["type"] != "noul" or type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise openrouter.OpenRouterError("Decider returned no usable probability for enrichment") from None
    return Decision(probability >= THRESHOLD, probability, identity, cost=openrouter.cost_of(response))
