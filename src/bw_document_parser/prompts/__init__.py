"""Prompts and versions used by the document parser."""

from importlib.resources import files

# Bump this when editing the prompt so enrichment can detect stale results.
FIGURE_DESCRIPTION_VERSION = "figure-description-v2"
FIGURE_DESCRIPTION = " ".join(
    files(__package__).joinpath("figure_description.txt").read_text(encoding="utf-8").splitlines()
)
