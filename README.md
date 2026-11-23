# Bearing Witness Document Parser

Turn a PDF into structured document elements, ordered text chunks, and extracted figures.

```sh
uv run bw-document-parser report.pdf --output-dir parsed-report
```

Use Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/). Pick a new output
directory. On the first run, Docling and tiktoken may download model files and
tokenizer data. The PDF is processed locally. No API key is required.

The compatibility output, `pages.json`, has one entry per source page in PDF order.
A blank page has an empty `pieces` list. Each piece has a text string and an
ordered list of image references.

```json
[
  {
    "pieces": [
      {
        "text": "## Findings\n\n[image:0]\nCaption: Survey results",
        "image_urls": ["images/38d6df2d08c4a55ffcbc1947.png"]
      }
    ]
  },
  {"pieces": []}
]
```

Image markers are zero-based within each piece. Resolve local image paths
relative to `pages.json`. Identical embedded images share one file.

The parser reuses one standard Docling converter and exports each page to HTML
in memory. Converting pages separately keeps paragraphs on their source pages.
Headings start sections, and long sections repeat their heading across pieces.
The chunker aims for 400 text tokens and splits at 850 when possible.
Tables, figures, and blocks with images stay intact and may exceed that limit.

Run the offline tests with `uv run python -m unittest discover -s tests`.

## Structured output and figure enrichment

Docling 2.152.0 runs the standard pipeline with Heron, TableFormer V1 in
accurate mode, and RapidOCR 3.10.0 on PyTorch. Every new bundle includes
`document.json` with the PDF's SHA-256 ID and typed pieces in reading order.
Pages reference top-level piece IDs. Figures reference images, published captions,
and any children Docling links. Tables preserve Docling's HTML, including merged
cells. Locations use one-based page numbers and bounding boxes as
`[left, top, right, bottom]` fractions of the displayed page.

Set `OPENROUTER_API_KEY` to use the hosted commands. The enrichment command
sends saved figure images, their published captions, and section headings to
`tencent/hy4-flash`. It adds a description and optional model-transcribed text
to each figure. Source captions, OCR text, and tables remain independently
addressable. Rerunning the command reuses successful descriptions for unchanged
inputs. Figures marked decorative are omitted from the Pieces view and search
index. The source bundle retains their images and their original Docling HTML.
Rerun enrichment and embedding on an older bundle to apply a new prompt version.

Model prompts and their versions live in `src/bw_document_parser/prompts/`.
When changing a prompt, bump its version in that directory so the next
enrichment run can detect previously saved results that need refreshing.

```sh
uv run bw-document-enrich parsed-report
uv run bw-document-embed parsed-report
uv run bw-document-search parsed-report "verified incidents by district"
```

The default embedding model is `nvidia/nemotron-3.5-embed-8b:free`. The
`embeddings.multimodal.json` sidecar stores search text, 4,096-dimensional
vectors, and links to the original pieces and pages. Table rows are searchable
as text, and available table crops and figures also receive image-and-text
vectors. NVIDIA's free endpoint excludes confidential information and personal
data. Use public, non-personal or synthetic examples for hosted smoke tests.

If the free route reaches its quota, the command saves the completed vectors
and reports the failure. Rerun it after the quota resets. A different model ID
can be selected explicitly with `--model`; the unsuffixed
`nvidia/nemotron-3.5-embed-8b` currently has no active OpenRouter endpoint.
Switching models rebuilds the vectors for that mode.

For a paid multimodal alternative, run
`uv run bw-document-embed parsed-report --model amazon/titan-embed-image-v1`.
Titan writes 1,024-dimensional vectors and uses smaller chunks to fit
OpenRouter's text limit. Amazon documents English as the supported text language.
The search command reads the model from the bundle.

```sh
uv run bw-document-embed parsed-report --mode text
uv run bw-document-search parsed-report "verified incidents by district" --mode text
```

Text mode writes `embeddings.text.json` with 1,536-dimensional vectors from
`openai/text-embedding-3-small` and sends no images to the embedding endpoint.
Search uses the model recorded in the chosen sidecar and returns the matching
pieces, including HTML and locations for tables. Rerun embedding after updating
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
