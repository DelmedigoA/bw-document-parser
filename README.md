# Bearing Witness Document Parser

Turn a PDF into structured document elements, ordered text chunks, and extracted figures.

```sh
uv run bw-document-parser report.pdf --output-dir parsed-report
```

Use Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/). Pick a new output
directory. On the first run, Docling and tiktoken may download model files and
tokenizer data. The PDF is processed locally. No API key is required.

Walk through page 1 of the bundled example in
[`examples/pdf_pipeline_piece_parts.ipynb`](examples/pdf_pipeline_piece_parts.ipynb).

```sh
doppler run --project bearingwitness --config dev_archiveai --only-secrets OPENROUTER_API_KEY -- \
  uv run --with jupyterlab --with ipykernel jupyter lab examples/pdf_pipeline_piece_parts.ipynb
```

Parse writes a `parsed.json` checkpoint, Docling HTML, and image assets. Resolve
writes the canonical `document.json`, with ordered pieces under their
source pages. Each piece has a `textual_part` with readable text and any table
HTML, and a `visual_part` with image URLs and optional descriptions. Either
part can be null. Source IDs point to the original blocks and their locations.
Enrich and Embed add to that document. They retain the piece IDs and the source blocks used for citations.

```sh
uv run bw-document-parser report.pdf --output-dir page-one --page-range 1 1 --parse-only
uv run bw-document-resolve page-one
```

`bw-document-resolve` also accepts existing bundles with a version 2
`document.json`. It saves the existing structured parse as a checkpoint and
writes the resolved document in the bundle. Existing indexes can supply cached
vectors to the next embedding run when the model and inputs match. Older bundles
remain readable by the inspector without converting them.

The parser reuses one standard Docling converter and converts the PDF in page
batches (12 pages by default) so Docling's internal layout, table, and OCR
batches can fill. A page-boundary patch stops Docling from joining a paragraph
across two pages, and piece IDs are renumbered per page so they stay identical
to a page-at-a-time conversion. Raise `--page-batch-size` (or
`BW_PARSER_PAGE_BATCH`) for fewer conversion setups at the cost of peak memory.
The resolver groups adjacent prose on each page. It uses 100 tokens as a soft
minimum, aims for 400, and caps prose at 850 tokens. Headings, figures, and
tables start new groups. Tables and figures retain their linked titles,
captions, footnotes, and notes. Embedding can split large inputs into smaller
targets without changing the document's pieces.

Run the offline tests with `uv run python -m unittest discover -s tests`.

## Structured output and figure enrichment

Docling 2.152.0 runs the standard pipeline with Heron, TableFormer V1 in
accurate mode, and RapidOCR 3.10.0 on PyTorch. Every new bundle includes
`document.json` with the PDF's SHA-256 ID and typed pieces in reading order.
Pages contain the resolved pieces that open on that page. Figures reference images, published captions,
and any children Docling links. Tables preserve Docling's HTML, including merged
cells. Locations use one-based page numbers and bounding boxes as
`[left, top, right, bottom]` fractions of the displayed page.

Set `OPENROUTER_API_KEY` to use the hosted commands. The enrichment command
sends saved figure images, their published captions, and section headings to
`tencent/hy4-flash`. It adds a description and optional model-transcribed text
to each figure. Source captions, OCR text, and tables remain independently
addressable. Rerunning the command reuses successful descriptions for unchanged
inputs. Rerun enrichment and embedding on an older bundle to apply a new
prompt version.

For resolved bundles, Enrich first asks Perplexity Decider V1.2 whether the
picture would add information beyond its heading and caption. A probability
of at least 0.5 sends it to Hy4 Flash. A skipped picture moves out of the
piece’s visual part and stays with its source block for provenance. The
decision is saved and reused when the image and context match. An API error leaves the figure
available for retry.

Decider batching is opt-in and off by default. `--decider-batch-size` (or
`BW_DECIDER_BATCH_SIZE`) sends up to 8 figures in one request. Perplexity’s
Decisions adapter reads images only from the top-level `state` array and
rejects an image placed inside a question, so a batch writes a labeled context
block before each figure’s images and asks each question to use only its own
labeled block. Every figure gets a stable question ID derived from its source
block ID, and answers are matched by that ID rather than by position. The
single-figure state shape and question text are unchanged.

A partial or invalid batch answer saves the valid decisions and the batch charge
before retrying only the missing figures as single requests. A batch the provider rejects with a
non-blocking error falls back to single requests. A quota error still stops the
run after saving completed decisions. Each response’s actual OpenRouter charge,
when reported, is recorded once in `document.json` under `decider_requests`; a figure’s share
of a batch is stored on its decision with `cost_allocation: "estimated"`, while
a single-request cost is `"exact"`. Keep the default of 1 until a live
comparison confirms that Perplexity follows the per-figure labels.

Model prompts and their versions live in `src/bw_document_parser/prompts/`.
When changing a prompt, bump its version in that directory so the next
enrichment run can detect previously saved results that need refreshing.

```sh
uv run bw-document-enrich parsed-report
uv run bw-document-embed parsed-report
uv run bw-document-search parsed-report "verified incidents by district"
```

Enrichment sends independent figures concurrently. The default is 4 requests
at once; set `--workers` or `BW_ENRICH_WORKERS` to change it. Each completed
figure is saved immediately, so a quota failure or interrupt keeps the paid
results and a rerun only requests the missing figures.

Compare single-figure and batched Decider runs with
`python benchmarks/decider_batching.py --bundle parsed-report --batch-size 8`.
The default report is a replay: it patches OpenRouter, so its timing measures
request count and concurrency, not model latency. Pass `--live` with
`OPENROUTER_API_KEY` to compare real decisions and charges. The report lists
the request count, elapsed time, cost, and any decision that changed. Both runs
start from fresh enrichment state in temporary copies of the resolved bundle,
including figures that were previously skipped.

The default embedding mode is text, using `openai/text-embedding-3-small`.
Pass `--mode multimodal` to use `nvidia/nemotron-3.5-embed-8b:free`. In that mode, targets store search text, 4,096-dimensional vectors, and source
references on the resolved pieces. Table rows are searchable
as text, and available table crops and figures also receive image-and-text
vectors. NVIDIA's free endpoint excludes confidential information and personal
data. Use public, non-personal or synthetic examples for hosted smoke tests.

If the free route reaches its quota, the command saves the completed vectors
and reports the failure. Rerun it after the quota resets. A different model ID
can be selected explicitly with `--model`; the unsuffixed
`nvidia/nemotron-3.5-embed-8b` currently has no active OpenRouter endpoint.
Switching models rebuilds the vectors for that mode.

Text inputs are sent in batches: 64 per request for OpenAI text embeddings, 8
for Nemotron, and 1 for Titan. Up to four requests run at once. Use
`--batch-size` and `--workers` (or `BW_EMBED_WORKERS`) to tune them. Vectors are
matched by each response item's own index, so an out-of-order reply cannot
attach a vector to the wrong piece. A rejected batch falls back to single
requests, and only the inputs missing from a partial reply are requested again.
The embedding mode metadata records OpenRouter's reported cost under `billing.total_cost`.
Per-page shares of a multi-input batch are labeled `estimated` because the
provider reports one cost per request; a single-input request is `exact`.
Completed batches are saved when a quota error stops the run, and rerunning
reuses them.

For a paid multimodal alternative, run
`uv run bw-document-embed parsed-report --mode multimodal --model amazon/titan-embed-image-v1`.
Titan writes 1,024-dimensional vectors and uses smaller chunks to fit
OpenRouter's text limit. Amazon documents English as the supported text language.
The search command reads the model from the bundle.

```sh
uv run bw-document-embed parsed-report --mode text
uv run bw-document-search parsed-report "verified incidents by district" --mode text
```

Text mode stores 1,536-dimensional vectors from
`openai/text-embedding-3-small` and sends no images to the embedding endpoint.
Search uses the model recorded for the selected mode and returns the matching
target and its parent piece, including source locations. Rerun embedding after updating
a bundle's descriptions. For Hebrew, verify representative queries against
known text and image targets before relying on the default model in production.

## Parsing inspector

```sh
uv run bw-document-inspector report.pdf parsed-report
```

Open the loopback URL printed by the command. The right pane uses ArchiveLens's
PDF.js viewer with page navigation, zoom, and selectable text. The left pane
follows the current page and can show structured pieces with their enrichments,
the Docling HTML, or the parsed JSON. New parses save the HTML in `html/`.
Older bundles still open with their page-oriented output. Keep the original PDF paired
with its own parse directory. Pass `--port 0` to choose a free local port.

When the bundle has an embedding index, open **Search** in the inspector. It
starts with **Figures only** selected. Click a result to open its piece on the
source page. Search uses the chosen index's model and the inspector process's
`OPENROUTER_API_KEY`. Restart the inspector from a shell with that variable set
if it was started without it.
