const element = id => document.getElementById(id);
const htmlMode = element("html-mode");
const structureMode = element("structure-mode");
const jsonMode = element("json-mode");
const searchMode = element("search-mode");
const searchPanel = element("search-panel");
const searchForm = element("search-form");
const searchQuery = element("search-query");
const searchSubmit = element("search-submit");
const searchIndex = element("search-index");
const searchStatus = element("search-status");
const searchResults = element("search-results");
const html = element("parsed-html");
const preview = element("parsed-preview");
const json = element("parsed-json");
const parsedStatus = element("parsed-status");
const pdfStatus = element("pdf-status");
const scroll = element("pdf-scroll");
const pageInput = element("page-number");
const prev = element("page-prev");
const next = element("page-next");
const zoomIn = element("zoom-in");
const zoomOut = element("zoom-out");
const zoomReset = element("zoom-reset");

let viewer;
let config;
let pages;
let structure;
let piecesById;
let total;
let page = 1;
let zoom = 1;
let mode = "html";
let pageRevision = 0;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}.`);
  return response.json();
}

// Existing parse bundles have JSON and images but no Docling HTML exports.
function renderChunks(pieces) {
  return pieces.map(piece => {
    const section = document.createElement("section");
    section.className = "chunk";
    const parts = piece.text.split(/(\[image:\d+\])/u);
    for (const part of parts) {
      const marker = /^\[image:(\d+)\]$/u.exec(part);
      const image = marker && piece.image_urls[Number(marker[1])];
      if (image && /^images\/[0-9a-f]{24}\.[a-zA-Z0-9]+$/u.test(image)) {
        const img = document.createElement("img");
        img.src = `/${image}`;
        img.alt = `Extracted image ${Number(marker[1]) + 1}`;
        section.append(img);
      } else if (part) {
        section.append(document.createTextNode(part));
      }
    }
    return section;
  });
}

// Copy only table/formatting elements and span attributes from saved HTML.
const tableTags = new Set([
  "table", "thead", "tbody", "tfoot", "caption", "tr", "td", "th",
  "p", "div", "span", "br", "b", "strong", "i", "em", "sub", "sup", "code", "ul", "ol", "li",
]);

function renderTable(htmlText) {
  const source = new DOMParser().parseFromString(htmlText, "text/html").querySelector("table");
  function copy(node) {
    if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.data);
    if (!tableTags.has(node.localName)) return document.createTextNode(node.textContent ?? "");
    const safe = document.createElement(node.localName);
    for (const name of ["rowspan", "colspan"]) {
      const value = node.getAttribute(name);
      if (value && /^\d{1,4}$/u.test(value)) safe.setAttribute(name, value);
    }
    for (const child of node.childNodes) safe.append(copy(child));
    return safe;
  }
  return source ? copy(source) : document.createDocumentFragment();
}

function renderStructure(current, {resolved = false} = {}) {
  const pieces = current.piece_ids.map(id => piecesById.get(id)).filter(Boolean);
  const owned = new Set(pieces.flatMap(piece => [
    ...piece.children, ...piece.caption_ids, ...piece.footnote_ids, piece.inferred_title_id, ...(piece.inferred_note_ids ?? []),
    ...(piece.content.cells ?? []).flatMap(cell => cell.child_ids),
  ]));

  function render(piece, ancestors = new Set()) {
    if (!piece || ancestors.has(piece.id) || (!resolved && piece.enrichment?.decorative === true)) {
      return document.createDocumentFragment();
    }
    const path = new Set([...ancestors, piece.id]);
    const renderId = id => render(piecesById.get(id), path);
    const content = piece.content;
    const isHeading = ["title", "section_header"].includes(piece.label);
    const tag = content.kind === "figure" ? "figure" :
      isHeading ? `h${Math.min(6, Math.max(1, content.level ?? 1))}` : "section";
    const node = document.createElement(tag);
    node.className = "chunk";
    node.dataset.pieceId = piece.id;
    node.dir = "auto";
    const images = piece.images.flatMap(image => {
      if (!/^images\/[0-9a-f]{24}\.[a-zA-Z0-9]+$/u.test(image.path)) return [];
      const img = document.createElement("img");
      img.src = `/${image.path}`;
      img.alt = content.kind === "table" ? "Extracted table image" : "Extracted figure";
      return [img];
    });
    if (content.kind === "table") {
      const table = typeof content.html === "string" ? renderTable(content.html) : document.createElement("table");
      if (content.cells) {
        const rows = Array.from({length: content.row_count}, () => table.insertRow());
        const cells = [...content.cells].sort((a, b) => a.row - b.row || a.column - b.column);
        for (const cell of cells) {
          const header = cell.roles.includes("column_header") || cell.roles.includes("row_header");
          const td = document.createElement(header ? "th" : "td");
          td.rowSpan = cell.row_span;
          td.colSpan = cell.column_span;
          if (cell.child_ids.length) td.append(...cell.child_ids.map(renderId));
          else td.textContent = cell.text;
          rows[cell.row]?.append(td);
        }
      }
      const inferred = piece.inferred_title_id;
      const captions = inferred ? [inferred] : piece.caption_ids;
      if (table instanceof HTMLTableElement && captions.length) {
        const caption = table.createCaption();
        if (inferred) caption.className = "inferred-table-caption";
        caption.append(...captions.map(renderId));
      }
      node.append(table);
      if (resolved && images.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = images.length === 1 ? "Table image" : "Table images";
        details.append(summary, ...images);
        node.append(details);
      }
    } else {
      node.textContent = content.text ?? content.data?.state ?? "";
      node.append(...images);
      if (tag !== "figure" && piece.caption_ids.length) {
        const caption = document.createElement("div");
        caption.append(...piece.caption_ids.map(renderId));
        node.append(caption);
      }
    }
    const extractedText = tag === "figure"
      ? piece.children.filter(id => piecesById.get(id)?.content.kind === "text")
      : [];
    const collapsedIds = new Set(extractedText);
    node.append(...[...piece.children, ...piece.footnote_ids]
      .filter(id => !collapsedIds.has(id)).map(renderId));
    if (piece.inferred_note_ids?.length) {
      const notes = document.createElement("div");
      notes.className = "table-notes";
      notes.append(...piece.inferred_note_ids.map(renderId));
      node.append(notes);
    }
    if (tag === "figure") {
      const caption = document.createElement("figcaption");
      caption.append(...piece.caption_ids.map(renderId));

      for (const [label, value] of [
        ["Generated description", piece.enrichment?.description],
        ["Model-transcribed text", piece.enrichment?.visible_text],
      ]) {
        if (!value) continue;
        const section = document.createElement("details");
        section.className = "figure-enrichment";
        const heading = document.createElement("summary");
        heading.textContent = label;
        const text = document.createElement("p");
        text.dir = "auto";
        text.textContent = value;
        section.append(heading, text);
        caption.append(section);
      }
      if (!resolved && extractedText.length) {
        const section = document.createElement("details");
        section.className = "figure-enrichment";
        const summary = document.createElement("summary");
        summary.textContent = "Rendered docling figure text (unreliable)";
        section.append(summary, ...extractedText.map(renderId));
        caption.append(section);
      }
      if (caption.hasChildNodes()) node.append(caption);
      if (resolved && !node.hasChildNodes()) return document.createDocumentFragment();
    }
    return node;
  }

  return pieces.filter(piece => !owned.has(piece.id)).map(piece => render(piece));
}

function searchResult(hit) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "search-hit";
  button.disabled = !hit.page;
  if (hit.image && /^images\/[0-9a-f]{24}\.[a-zA-Z0-9]+$/u.test(hit.image)) {
    const image = document.createElement("img");
    image.src = `/${hit.image}`;
    image.alt = "";
    button.append(image);
  }
  const content = document.createElement("span");
  const label = document.createElement("strong");
  const kind = hit.kind === "figure" ? "Figure" : hit.kind === "table" ? "Table" : "Text";
  label.textContent = hit.page ? `${kind} · Page ${hit.page}` : kind;
  const snippet = document.createElement("span");
  snippet.dir = "auto";
  snippet.className = "search-snippet";
  snippet.textContent = hit.text;
  content.append(label, snippet);
  button.append(content);
  button.addEventListener("click", () => {
    mode = "structure";
    showPage(hit.page).then(shown => {
      if (!shown) return;
      highlightResult(hit);
    });
  });
  item.append(button);
  return item;
}

function highlightResult(hit) {
  const piece = preview.querySelector(`[data-piece-id="${CSS.escape(hit.pieceId)}"]`) ??
    (hit.parentPieceId && preview.querySelector(`[data-section-id="${CSS.escape(hit.parentPieceId)}"]`));
  if (piece) {
    for (let section = piece.closest("details"); section; section = section.parentElement.closest("details")) {
      section.open = true;
    }
    piece.classList.add("search-target");
    piece.tabIndex = -1;
    piece.focus({preventScroll: true});
    piece.scrollIntoView({block: "nearest"});
  }
}

async function runSearch(event) {
  event.preventDefault();
  const query = searchQuery.value.trim();
  if (!query || searchSubmit.disabled) return;
  searchSubmit.disabled = true;
  searchStatus.textContent = "Searching…";
  searchResults.replaceChildren();
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        query, mode: searchIndex.value, figuresOnly: element("figures-only").checked,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error ?? "Search failed.");
    searchResults.replaceChildren(...data.results.map(searchResult));
    searchStatus.textContent = data.results.length
      ? `${data.results.length} ${data.results.length === 1 ? "result" : "results"}`
      : "No results. Try another query or check the bundle’s embeddings.";
  } catch (error) {
    searchStatus.textContent = error.message;
  } finally {
    searchSubmit.disabled = false;
  }
}

const collapsedPieces = new Set();

function renderResolved(current) {
  piecesById = new Map();
  for (const piece of current.pieces) {
    const tables = new Map((piece.textual_part?.tables ?? []).map(table => [table.source_id, table.html]));
    const images = new Map();
    for (const image of piece.visual_part ?? []) {
      if (!images.has(image.source_id)) images.set(image.source_id, []);
      images.get(image.source_id).push(image);
    }
    for (const id of piece.source_ids) {
      const source = current.source_blocks[id];
      const visuals = images.get(id) ?? [];
      const content = tables.has(id) ? {...source.content, html: tables.get(id)} : source.content;
      const enrichment = visuals.length ? {
        description: visuals[0].description, visible_text: visuals[0].transcription,
      } : null;
      piecesById.set(id, {
        ...source, content, enrichment,
        images: visuals.map(image => ({path: image.url, location: image.location})),
      });
    }
  }
  return current.pieces.flatMap(piece => {
    const blocks = renderStructure({piece_ids: piece.source_ids}, {resolved: true});
    if (!blocks.some(block => block instanceof HTMLElement)) return [];
    const section = document.createElement("details");
    section.className = "resolved-section";
    section.dataset.sectionId = piece.id;
    section.open = !collapsedPieces.has(piece.id);
    section.addEventListener("toggle", () => {
      if (section.open) collapsedPieces.delete(piece.id);
      else collapsedPieces.add(piece.id);
    });
    const summary = document.createElement("summary");
    const heading = blocks[0];
    if (
      heading instanceof HTMLElement && heading.matches("h1, h2, h3, h4, h5, h6")
      && heading.textContent.trim() === piece.title
    ) {
      summary.append(blocks.shift());
    } else {
      summary.textContent = piece.section_heading
        ? `${piece.title} · ${piece.section_heading}`
        : piece.title || "Untitled piece";
    }
    const body = document.createElement("div");
    body.className = "resolved-section-body";
    body.append(...blocks);
    section.append(summary, body);
    return [section];
  });
}

async function showParsedPage() {
  const revision = ++pageRevision;
  const number = page;
  const selectedMode = mode;
  structureMode.setAttribute("aria-pressed", String(mode === "structure"));
  htmlMode.setAttribute("aria-pressed", String(mode === "html"));
  jsonMode.setAttribute("aria-pressed", String(mode === "json"));
  searchMode.setAttribute("aria-pressed", String(mode === "search"));
  html.hidden = preview.hidden = json.hidden = searchPanel.hidden = true;
  parsedStatus.hidden = mode === "search";
  if (mode === "search") {
    searchPanel.hidden = false;
    return;
  }

  if (config.resolved) {
    if (!config.parsedPages.includes(number)) {
      parsedStatus.textContent = `No parser output for page ${number}.`;
      return false;
    }
    const url = `/html/page-${number}.html`;
    if (selectedMode === "html" && config.htmlPages.includes(url)) {
      if (html.getAttribute("src") !== url) html.src = url;
      html.hidden = false;
      parsedStatus.hidden = true;
      return true;
    }
    parsedStatus.textContent = "Loading parsed page…";
    try {
      const current = await fetchJson(`/api/pages/${number}`);
      if (revision !== pageRevision) return false;
      if (selectedMode === "json") {
        json.textContent = JSON.stringify(current, null, 2);
        json.hidden = false;
        json.scrollTop = 0;
      } else {
        preview.replaceChildren(...renderResolved(current));
        preview.hidden = false;
        preview.scrollTop = 0;
        if (!preview.childElementCount) {
          parsedStatus.textContent = "This page has no pieces to display.";
          return true;
        }
      }
      parsedStatus.hidden = true;
      return true;
    } catch (error) {
      if (revision === pageRevision) parsedStatus.textContent = error.message;
      return false;
    }
  }

  const current = pages[page - 1];
  if (!current) {
    parsedStatus.textContent = `No parser output for page ${page}.`;
    return;
  }

  const structuredPage = structure?.pages[page - 1];
  if (mode === "json") {
    const ordered = new Map();
    function collect(id) {
      const piece = piecesById.get(id);
      if (!piece || ordered.has(id)) return;
      ordered.set(id, piece);
      for (const related of [
        ...piece.children, ...piece.caption_ids, ...piece.footnote_ids, piece.inferred_title_id, ...(piece.inferred_note_ids ?? []),
      ]) {
        collect(related);
      }
    }
    if (structuredPage) structuredPage.piece_ids.forEach(collect);
    const data = structuredPage ? {...structuredPage, pieces: [...ordered.values()]} : current;
    json.textContent = JSON.stringify(data, null, 2);
    json.hidden = false;
    json.scrollTop = 0;
  } else if (mode === "structure" && structuredPage) {
    preview.replaceChildren(...renderStructure(structuredPage));
    preview.hidden = false;
    preview.scrollTop = 0;
    if (!preview.childElementCount) {
      parsedStatus.textContent = "This page has no pieces to display.";
      return;
    }
  } else {
    const url = `/html/page-${page}.html`;
    if (config.htmlPages.includes(url)) {
      if (html.getAttribute("src") !== url) html.src = url;
      html.hidden = false;
    } else if (structuredPage?.piece_ids.length) {
      preview.replaceChildren(...renderStructure(structuredPage));
      preview.hidden = false;
      preview.scrollTop = 0;
    } else if (current.pieces.length) {
      preview.replaceChildren(...renderChunks(current.pieces));
      preview.hidden = false;
      preview.scrollTop = 0;
      parsedStatus.textContent = "Rendered from JSON. This bundle has no Docling HTML for this page.";
      return;
    } else {
      parsedStatus.textContent = "This page has no parsed pieces.";
      return;
    }
  }

  if (pages.length !== total) {
    parsedStatus.textContent = `The PDF has ${total} pages and the bundle has ${pages.length}.`;
  } else {
    parsedStatus.hidden = true;
  }
}

async function showPage(number) {
  page = Math.max(1, Math.min(total, Math.round(number) || 1));
  pageInput.value = page;
  prev.disabled = page === 1;
  next.disabled = page === total;
  scroll.scrollTop = 0;
  viewer.show(page, zoom);
  return (await showParsedPage()) ?? true;
}

function setZoom(value) {
  zoom = Math.max(0.6, Math.min(2.5, value));
  zoomReset.textContent = `${Math.round(zoom * 100)}%`;
  zoomOut.disabled = zoom === 0.6;
  zoomIn.disabled = zoom === 2.5;
  viewer.show(page, zoom);
}

async function start() {
  htmlMode.disabled = jsonMode.disabled = structureMode.disabled = true;
  try {
    const [viewerModule, inspectorConfig] = await Promise.all([
      import("./document-viewer.js"),
      fetchJson("/config.json"),
    ]);
    config = inspectorConfig;
    if (!config.resolved) {
      pages = await fetchJson("/pages.json");
      if (config.hasDocument) {
        structure = await fetchJson("/document.json");
        piecesById = new Map(structure.pieces.map(piece => [piece.id, piece]));
      }
    }
    if (config.hasDocument) {
      structureMode.hidden = false;
      mode = "structure";
    }
    element("filename").textContent = config.title;
    document.title = `${config.title} — Parsing inspector`;

    viewer = new viewerModule.DocumentViewer(scroll, element("pdf-page"), pdfStatus);
    total = await viewer.load("/source.pdf");
    pageInput.max = total;
    element("page-count").textContent = `/ ${total}`;

    htmlMode.disabled = jsonMode.disabled = structureMode.disabled = false;
    element("pdf-controls").disabled = false;
    for (const index of config.searchModes ?? []) {
      searchIndex.add(new Option(index === "multimodal" ? "Multimodal" : "Text only", index));
    }
    searchMode.hidden = !searchIndex.options.length;
    searchForm.addEventListener("submit", runSearch);
    for (const [button, view] of [
      [structureMode, "structure"], [htmlMode, "html"], [jsonMode, "json"], [searchMode, "search"],
    ]) {
      button.addEventListener("click", () => {
        mode = view;
        showParsedPage();
        if (mode === "search") searchQuery.focus();
      });
    }
    prev.addEventListener("click", () => showPage(page - 1));
    next.addEventListener("click", () => showPage(page + 1));
    pageInput.addEventListener("change", () => showPage(pageInput.valueAsNumber));
    zoomOut.addEventListener("click", () => setZoom(zoom - 0.2));
    zoomIn.addEventListener("click", () => setZoom(zoom + 0.2));
    zoomReset.addEventListener("click", () => setZoom(1));

    let width = scroll.clientWidth;
    new ResizeObserver(() => {
      if (width === scroll.clientWidth) return;
      width = scroll.clientWidth;
      viewer.show(page, zoom);
    }).observe(scroll);

    showPage(config.parsedPages?.[0] ?? 1);
  } catch (error) {
    parsedStatus.textContent = "The inspector could not open this document.";
    pdfStatus.textContent = error.message;
  }
}

start();
