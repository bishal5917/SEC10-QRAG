import time
import base64
from pathlib import Path
from typing import List, Dict, Any

import pymupdf
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image, FigureCaption, NarrativeText, Title, ListItem

from src.logger import get_logger

log = get_logger("pdf_loader")

# Figure page render resolution — 150 DPI is enough for llava to read charts
_FIGURE_DPI    = 150
_FIGURE_MATRIX = pymupdf.Matrix(_FIGURE_DPI / 72, _FIGURE_DPI / 72)

# unstructured element types treated as figures (charts, diagrams, images)
_FIGURE_TYPES = (Image, FigureCaption)

# unstructured element types treated as plain text
_TEXT_TYPES = (NarrativeText, Title, ListItem)

# Geometry fallback thresholds for vector charts (bar/line charts in SEC filings)
_NEUTRAL_THRESHOLD  = 0.15   # color channel spread below this = grey/black/white
_MIN_CHART_RECT_AREA = 400   # px² — chart bars are large; table borders are thin slivers
_MIN_COLORED_FILLS  = 3      # minimum qualifying colored rects to call it a chart page


def _is_colored(fill) -> bool:
    if not fill or not isinstance(fill, (list, tuple)) or len(fill) < 3:
        return False
    r, g, b = fill[0], fill[1], fill[2]
    return max(abs(r - g), abs(r - b), abs(g - b)) > _NEUTRAL_THRESHOLD


def _is_vector_chart(mu_page) -> bool:
    """
    Fallback detector for vector charts (bar/line/pie charts drawn with PDF
    path commands). unstructured fast does not classify these as Image elements
    since they are not embedded raster images — they are drawing instructions.

    A vector chart page has colored filled rectangles that are:
      - large enough to be chart bars (not thin table border lines)
      - not wide flat banners (page header stripes have width >> height)
      - at least 2 distinct colors (table row shading repeats one color)
    """
    qualifying = []
    for d in mu_page.get_drawings():
        fill = d.get("fill")
        if not _is_colored(fill):
            continue
        rect = d.get("rect")
        if rect is None:
            continue
        w = abs(rect.x1 - rect.x0)
        h = abs(rect.y1 - rect.y0)
        if w * h < _MIN_CHART_RECT_AREA:
            continue
        if h == 0 or (w / h) > 5:  # skip wide flat banners
            continue
        qualifying.append(tuple(round(c, 1) for c in fill[:3]))
    return len(qualifying) >= _MIN_COLORED_FILLS and len(set(qualifying)) >= 2


def _render_page_as_b64(mu_doc: pymupdf.Document, page_num: int) -> str:
    """Render a page to PNG at 150 DPI and return as base64 string for llava."""
    pix = mu_doc[page_num].get_pixmap(matrix=_FIGURE_MATRIX, colorspace=pymupdf.csRGB)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _table_to_markdown(element: Table) -> str:
    """
    Convert an unstructured Table element to markdown.
    unstructured returns table content as plain text with whitespace alignment.
    We parse it into rows and render as markdown.
    """
    raw = (element.metadata.text_as_html or element.text or "").strip()

    # If HTML is available (unstructured provides it for tables), parse it
    if element.metadata.text_as_html:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(element.metadata.text_as_html, "html.parser")
            rows = []
            for tr in soup.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                rows.append(cells)
            if not rows:
                return element.text or ""
            n_cols = max(len(r) for r in rows)
            # Pad rows to same width
            rows = [r + [""] * (n_cols - len(r)) for r in rows]
            md = ["| " + " | ".join(rows[0]) + " |",
                  "|" + "---|" * n_cols]
            for row in rows[1:]:
                if any(row):
                    md.append("| " + " | ".join(row) + " |")
            return "\n".join(md)
        except Exception as e:
            log.warning(f"HTML table parse failed ({e}), using plain text")

    return element.text or ""


def load_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract chunks from a PDF using unstructured for element-level detection.

    unstructured natively identifies: Title, NarrativeText, Table, Image,
    FigureCaption, ListItem — no geometry heuristics needed.

    Chunks are grouped per page. Each page produces one chunk typed as:
      'text'   — narrative prose, titles, lists
      'table'  — structured data table (rendered as markdown)
      'figure' — chart/diagram/image (page rendered as PNG for llava)

    If a page has both a figure and a table, figure takes priority.
    """
    log.info(f"Loading: {pdf_path.name}")
    t0 = time.perf_counter()
    source = pdf_path.name

    # partition_pdf with strategy="fast" uses PDF structure only (no ML layout model)
    # strategy="hi_res" uses detectron2 for better accuracy on complex layouts
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        infer_table_structure=True,   # populate text_as_html for Table elements
        include_page_breaks=True,
    )

    # Group elements by page number
    pages: Dict[int, Dict[str, Any]] = {}
    for el in elements:
        page_num = (el.metadata.page_number or 1)
        if page_num not in pages:
            pages[page_num] = {"text_parts": [], "tables": [], "has_figure": False}

        if isinstance(el, _FIGURE_TYPES):
            pages[page_num]["has_figure"] = True
        elif isinstance(el, Table):
            pages[page_num]["tables"].append(el)
        elif isinstance(el, _TEXT_TYPES):
            pages[page_num]["text_parts"].append(el.text.strip())

    # Open pymupdf doc once for figure rendering
    mu_doc = pymupdf.open(str(pdf_path))

    chunks = []
    for page_num in sorted(pages.keys()):
        page = pages[page_num]
        mu_page   = mu_doc[page_num - 1]
        has_figure = page["has_figure"] or _is_vector_chart(mu_page)  # hybrid
        has_table  = bool(page["tables"])

        if has_figure:
            chunk_type = "figure"
            # Spatial text reconstruction as searchable text for retrieval
            words = mu_page.get_text("words", sort=True)
            rows: dict = {}
            for w in words:
                bucket = int(w[1] / 6) * 6
                rows.setdefault(bucket, []).append(w)
            text = "\n".join(
                "  ".join(w[4] for w in sorted(rows[y], key=lambda x: x[0]))
                for y in sorted(rows)
            ).strip()
            image_b64 = _render_page_as_b64(mu_doc, page_num - 1)
            log.debug(f"  p{page_num}: FIGURE — rendered to PNG")

        elif has_table:
            chunk_type = "table"
            parts = []
            # Prepend any text on the page (titles, labels above the table)
            if page["text_parts"]:
                parts.append("\n".join(page["text_parts"]))
            for tbl in page["tables"]:
                parts.append(_table_to_markdown(tbl))
            text = "\n\n".join(parts).strip()
            image_b64 = None
            log.debug(f"  p{page_num}: TABLE — {len(page['tables'])} table(s)")

        else:
            chunk_type = "text"
            text = "\n".join(page["text_parts"]).strip()
            image_b64 = None

        if not text:
            continue

        chunk: Dict[str, Any] = {
            "text":       text,
            "chunk_type": chunk_type,
            "page":       page_num,
            "source":     source,
        }
        if image_b64:
            chunk["image_b64"] = image_b64
        chunks.append(chunk)

    mu_doc.close()

    by_type: dict = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    log.info(
        f"Loaded {pdf_path.name}: {len(chunks)} chunks {by_type} "
        f"in {(time.perf_counter() - t0) * 1000:.1f}ms"
    )
    return chunks


def load_all_pdfs(data_dir: Path) -> List[Dict[str, Any]]:
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")
    log.info(f"Found {len(pdf_files)} PDF(s) in {data_dir}")
    all_chunks = []
    for pdf_path in pdf_files:
        all_chunks.extend(load_pdf(pdf_path))
    log.info(f"All PDFs loaded: {len(all_chunks)} total raw chunks")
    return all_chunks
