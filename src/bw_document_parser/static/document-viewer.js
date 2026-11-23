// Adapted from ArchiveLens's document-viewer.js: PDF.js page canvas, fit-width
// scaling, and a selectable text layer. This viewer has no chat or citation state.
import * as pdfjs from "./vendor/pdfjs/pdf.mjs";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "./vendor/pdfjs/pdf.worker.mjs", import.meta.url,
).href;

export class DocumentViewer {
  constructor(scroll, surface, status) {
    this.scroll = scroll;
    this.surface = surface;
    this.status = status;
    this.revision = 0;
  }

  async load(url) {
    this.pdf = await pdfjs.getDocument({ url, rangeChunkSize: 65536 }).promise;
    return this.pdf.numPages;
  }

  async show(number, zoom) {
    const revision = ++this.revision;
    this.renderTask?.cancel();
    this.textTask?.cancel();

    try {
      const page = await this.pdf.getPage(number);
      if (revision !== this.revision) return;

      const base = page.getViewport({ scale: 1 });
      const width = Math.max(1, this.scroll.clientWidth - 32);
      const scale = Math.min(width / base.width, 2) * zoom;
      const viewport = page.getViewport({ scale });
      const pixelRatio = window.devicePixelRatio || 1;

      const layer = document.createElement("div");
      layer.className = "document-page-layer";
      layer.style.width = `${viewport.width}px`;
      layer.style.height = `${viewport.height}px`;

      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-hidden", "true");
      canvas.width = Math.ceil(viewport.width * pixelRatio);
      canvas.height = Math.ceil(viewport.height * pixelRatio);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;

      const text = document.createElement("div");
      text.className = "textLayer";
      text.style.setProperty("--total-scale-factor", viewport.scale);
      layer.append(canvas, text);
      this.surface.replaceChildren(layer);

      this.textTask = new pdfjs.TextLayer({
        textContentSource: page.streamTextContent(),
        container: text,
        viewport,
      });
      this.renderTask = page.render({
        canvasContext: canvas.getContext("2d"),
        viewport,
        transform: [pixelRatio, 0, 0, pixelRatio, 0, 0],
      });
      await Promise.all([this.renderTask.promise, this.textTask.render()]);
      if (revision === this.revision) this.status.hidden = true;
    } catch (error) {
      if (revision !== this.revision) return;
      this.status.textContent = `Could not render page ${number}. ${error.message}`;
      this.status.hidden = false;
    }
  }
}
