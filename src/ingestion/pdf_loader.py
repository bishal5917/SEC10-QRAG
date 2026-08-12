"""
PDF extraction pipeline — extracts TEXT, TABLES, and CHARTS from PDFs.

Chart detection strategy:
  - Only detect ACTUAL data charts (bar, pie, line, area charts)
  - Ignore logos, decorations, headers, background templates
  - Two detection paths:
    1. Raster charts: unique large images (not repeated across pages)
       with chart-like aspect ratios
    2. Vector charts: pages with many colored drawing fills AND low text density

Table extraction:
  - pdfplumber for structural detection
  - Output as clean markdown tables with proper column alignment
  - Floating headers carried across continuation pages
"""

import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import pymupdf
import pdfplumber

from src.logger import get_logger

log = get_logger("pdf_loader")

# ── Constants ─────────────────────────────────────────────────────────────────

_FIGURE_DPI = 150
_FIGURE_MATRIX = pymupdf.Matrix(_FIGURE_DPI / 72, _FIGURE_DPI / 72)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _find_repeated_images(mu_doc) -> Set[int]:
    """
    Find image xrefs that appear on >30% of pages.
    These are always decorative (logos, header bars, watermarks, templates).
    """
    total_pages = len(mu_doc)
    if total_pages < 3:
        return set()

    xref_page_count: Dict[int, int] = {}
    for page in mu_doc:
        seen = set()
        for img in page.get_images(full=True):
            xref = img[0]
            if xref not in seen:
                xref_page_count[xref] = xref_page_count.get(xref, 0) + 1
                seen.add(xref)

    # 30% threshold — anything appearing on a third+ of pages is decorative
    threshold = max(3, total_pages * 0.3)
    repeated = {xref for xref, count in xref_page_count.items() if count >= threshold}

    if repeated:
        log.debug(f"  Skipping {len(repeated)} repeated image(s) (decorative)")
    return repeated


def _page_has_chart_image(mu_page, skip_xrefs: Set[int]) -> bool:
    """
    Detect pages with an embedded raster chart image.

    A chart image is:
    - NOT a repeated/decorative image
    - Large enough (>15% of page area)
    - Has chart-like proportions (wider than tall, or roughly square)
      NOT a narrow vertical banner or a tiny wide strip
    """
    page_area = mu_page.rect.width * mu_page.rect.height
    if page_area == 0:
        return False

    for img in mu_page.get_images(full=True):
        xref = img[0]
        w, h = img[2], img[3]

        # Skip repeated decorative images
        if xref in skip_xrefs:
            continue

        # Must be large relative to page (>15% of page area)
        img_area = w * h
        ratio = img_area / page_area
        if ratio < 0.15:
            continue

        # Chart-like proportions check:
        # - Charts are usually landscape or square (w/h between 0.5 and 4.0)
        # - Reject very tall narrow images (vertical banners) or very thin wide strips
        aspect = w / h if h > 0 else 0
        if aspect < 0.4 or aspect > 5.0:
            continue

        # Must be minimum size (charts are at least 400px wide)
        if w < 400:
            continue

        log.debug(f"  p{mu_page.number+1}: CHART image found ({w}x{h}px, {ratio:.0%} of page, aspect={aspect:.1f})")
        return True

    return False


def _page_has_vector_chart(mu_page) -> bool:
    """
    Detect vector charts (drawn with PDF path commands, not embedded as images).

    Strategy: A vector chart page has BOTH:
    1. Many colored filled shapes (bars, pie slices, area fills)
    2. Low text coverage (chart takes up space, only labels/title as text)

    This catches bar charts, pie charts, area charts drawn with vector commands.
    Line-only charts with no fills won't be caught (acceptable tradeoff).
    """
    page_area = mu_page.rect.width * mu_page.rect.height
    if page_area == 0:
        return False

    # First: quick text coverage check (cheap)
    blocks = mu_page.get_text("dict", sort=True)["blocks"]
    text_area = 0.0
    for block in blocks:
        if block["type"] == 0:
            bbox = block["bbox"]
            text_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    text_coverage = text_area / page_area

    # Charts typically have <20% text coverage (just title + axis labels)
    if text_coverage > 0.20:
        return False

    # Second: count colored fills (expensive, only do if text coverage is low)
    drawings = mu_page.get_drawings()
    if len(drawings) < 15:
        return False

    colored_fills = 0
    distinct_colors = set()

    for d in drawings:
        fill = d.get("fill")
        if not fill or not isinstance(fill, (list, tuple)) or len(fill) < 3:
            continue

        r, g, b = fill[0], fill[1], fill[2]

        # Skip grey/black/white fills (table borders, text backgrounds)
        color_spread = max(abs(r - g), abs(r - b), abs(g - b))
        if color_spread < 0.1:
            continue

        # Skip very small fills (dots, thin lines)
        rect = d.get("rect")
        if rect:
            fw = abs(rect.x1 - rect.x0)
            fh = abs(rect.y1 - rect.y0)
            if fw * fh < 200:  # too small
                continue

        colored_fills += 1
        distinct_colors.add((round(r, 1), round(g, 1), round(b, 1)))

    # A real chart has multiple colored fills in at least 2 distinct colors
    if colored_fills >= 5 and len(distinct_colors) >= 2:
        log.debug(
            f"  p{mu_page.number+1}: VECTOR chart detected "
            f"({colored_fills} colored fills, {len(distinct_colors)} colors, "
            f"text_coverage={text_coverage:.0%})"
        )
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE DETECTION & EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _page_has_table(pl_page) -> bool:
    """
    Detect if page has a meaningful table.
    
    Tries two strategies:
    1. Default (lines/edges) — works for tables with visible borders
    2. Text-based — works for borderless financial tables (aligned columns)
    """
    # Strategy 1: bordered tables
    for table in pl_page.find_tables():
        data = table.extract()
        non_empty = [r for r in data if any(c and str(c).strip() for c in r)]
        if len(non_empty) >= 2:
            return True

    # Strategy 2: borderless tables (common in SEC financial statements)
    # Check if page has text patterns that look like a financial table:
    # - Lines with dollar signs/numbers aligned in columns
    # - Multiple lines with consistent numeric patterns
    text = pl_page.extract_text() or ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Count lines that have $ or multiple numbers (indicating tabular financial data)
    dollar_lines = 0
    for line in lines:
        # Financial table line: has $ or multiple comma-separated numbers
        if "$" in line:
            dollar_lines += 1
        elif line.count(",") >= 2 and any(c.isdigit() for c in line):
            # Lines like "Net revenue    11,715    18,353"
            import re
            numbers = re.findall(r'[\d,]+\.\d+|[\d,]{2,}', line)
            if len(numbers) >= 2:
                dollar_lines += 1

    # If >5 lines look like financial data, treat the page as having a table
    if dollar_lines >= 5:
        return True

    return False


def _extract_tables_as_markdown(mu_page, pl_page, carried_headers: dict = None) -> Tuple[str, dict]:
    """
    Extract all tables on a page as clean markdown.

    Handles two cases:
    1. Bordered tables (pdfplumber detects structure) → proper markdown grid
    2. Borderless financial tables (aligned text columns) → formatted as code block

    Produces output that LLMs can parse reliably.
    """
    try:
        page_words = mu_page.get_text("words", sort=True)
        sections = []
        found_headers = None

        # Try bordered tables first (pdfplumber structural detection)
        bordered_tables = []
        for tbl in pl_page.find_tables():
            data = tbl.extract()
            non_empty = [r for r in data if any(c and str(c).strip() for c in r)]
            if len(non_empty) >= 2:
                bordered_tables.append(tbl)

        if bordered_tables:
            # Extract bordered tables with floating headers
            first_table_top = min(t.bbox[1] for t in bordered_tables)

            # Pre-table title text
            pre_words = [w for w in page_words if w[3] < first_table_top - 40]
            if pre_words:
                pre_rows: dict = {}
                for w in pre_words:
                    bucket = int(w[1] / 6) * 6
                    pre_rows.setdefault(bucket, []).append(w)
                for y in sorted(pre_rows):
                    line = " ".join(w[4] for w in sorted(pre_rows[y], key=lambda x: x[0]))
                    if line.strip():
                        sections.append(line)
                sections.append("")

            for tbl in bordered_tables:
                tbl_top = tbl.bbox[1]

                # Floating headers above table
                header_words = [w for w in page_words if w[3] <= tbl_top and w[1] >= tbl_top - 40]
                if header_words:
                    col_buckets: dict = {}
                    for w in header_words:
                        x_bucket = int(w[0] / 60) * 60
                        col_buckets.setdefault(x_bucket, []).append(w[4])
                    header_line = "  |  ".join(
                        " ".join(words) for _, words in sorted(col_buckets.items())
                    )
                    found_headers = col_buckets
                    sections.append(f"**{header_line}**")
                elif carried_headers:
                    header_line = "  |  ".join(
                        " ".join(words) for _, words in sorted(carried_headers.items())
                    )
                    sections.append(f"**(Continued) {header_line}**")

                # Extract table → markdown
                data = tbl.extract()
                if not data:
                    continue

                cleaned = []
                for row in data:
                    cleaned_row = [
                        str(c).strip().replace("\n", " ").replace("|", "/") if c else ""
                        for c in row
                    ]
                    cleaned.append(cleaned_row)

                n_cols = max(len(r) for r in cleaned)
                header_row = None
                data_rows = []
                for row in cleaned:
                    padded = row + [""] * (n_cols - len(row))
                    if header_row is None and any(padded):
                        header_row = padded
                    elif any(padded):
                        data_rows.append(padded)

                if header_row is None:
                    continue

                md_lines = [
                    "| " + " | ".join(header_row) + " |",
                    "| " + " | ".join(["---"] * n_cols) + " |",
                ]
                for row in data_rows:
                    md_lines.append("| " + " | ".join(row) + " |")

                sections.append("\n".join(md_lines))
                sections.append("")

        else:
            # Borderless table — extract as structured text in code block
            # This preserves whitespace alignment that LLMs can read
            text = mu_page.get_text("text", sort=True).strip()
            if text:
                lines = [l for l in text.split("\n") if l.strip()]
                sections.append("**Financial Data:**")
                sections.append("```")
                sections.extend(lines)
                sections.append("```")

        text = "\n".join(sections).strip()
        if not text:
            text = mu_page.get_text("text", sort=True).strip()

        return text, found_headers

    except Exception as e:
        log.warning(f"Table extraction failed ({e}), using plain text fallback")
        return mu_page.get_text("text", sort=True).strip(), None


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE RENDERING & DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_page_as_b64(mu_doc, page_num: int) -> str:
    """Render page at 150 DPI → PNG → base64."""
    pix = mu_doc[page_num].get_pixmap(matrix=_FIGURE_MATRIX, colorspace=pymupdf.csRGB)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _describe_figure(image_b64: str, source: str, page_num: int) -> str:
    """Send chart image to llava for description. Used as searchable text at ingest."""
    import httpx
    from src.config import OLLAMA_BASE_URL, VISION_MODEL
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": (
                    "This is a chart or graph from a financial document. "
                    "Describe it in detail: What type of chart is it? "
                    "What are the axis labels? What data values are shown? "
                    "What trends or comparisons does it illustrate? "
                    "Include all specific numbers, percentages, and time periods visible."
                ),
                "images": [image_b64],
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        log.warning(f"llava description failed for p{page_num} of {source}: {e}")
        return f"[Chart on page {page_num} of {source} — description unavailable]"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract chunks from a PDF. Each page can produce multiple chunks:
      - TEXT chunk: always (if page has substantial text)
      - TABLE chunk: if pdfplumber detects a table structure
      - FIGURE chunk: ONLY if enabled AND page has an actual data chart

    Figure detection is disabled by default (FIGURE_DETECTION_ENABLED=false).
    Text extraction captures all chart labels and values anyway.
    """
    from src.config import FIGURE_DETECTION_ENABLED

    log.info(f"Loading: {pdf_path.name} (figure_detection={'ON' if FIGURE_DETECTION_ENABLED else 'OFF'})")
    t0 = time.perf_counter()
    source = pdf_path.name

    mu_doc = pymupdf.open(str(pdf_path))
    chunks = []
    last_table_headers = None

    # Pre-scan: identify repeated/decorative images to skip (only if figure detection ON)
    repeated_xrefs = _find_repeated_images(mu_doc) if FIGURE_DETECTION_ENABLED else set()

    with pdfplumber.open(str(pdf_path)) as plumber_doc:
        for page_num in range(1, len(mu_doc) + 1):
            mu_page = mu_doc[page_num - 1]
            pl_page = plumber_doc.pages[page_num - 1]

            has_table = _page_has_table(pl_page)

            # Chart detection — only if enabled
            has_chart = False
            if FIGURE_DETECTION_ENABLED:
                has_chart = (
                    _page_has_chart_image(mu_page, repeated_xrefs)
                    or _page_has_vector_chart(mu_page)
                )

            # ── TABLE extraction ──────────────────────────────────────────────
            if has_table:
                table_md, found_headers = _extract_tables_as_markdown(
                    mu_page, pl_page, last_table_headers
                )
                last_table_headers = found_headers or last_table_headers
                if table_md.strip():
                    chunks.append({
                        "text": table_md.strip(),
                        "chunk_type": "table",
                        "page": page_num,
                        "source": source,
                    })
                    log.debug(f"  p{page_num}: TABLE")
            else:
                last_table_headers = None

            # ── CHART extraction (only real charts) ───────────────────────────
            if has_chart:
                image_b64 = _render_page_as_b64(mu_doc, page_num - 1)
                description = _describe_figure(image_b64, source, page_num)
                if description.strip():
                    chunks.append({
                        "text": description.strip(),
                        "chunk_type": "figure",
                        "page": page_num,
                        "source": source,
                        "image_b64": image_b64,
                    })
                    log.debug(f"  p{page_num}: CHART — described via llava")

            # ── TEXT extraction (always) ──────────────────────────────────────
            page_text = mu_page.get_text("text", sort=True).strip()
            if page_text and len(page_text) > 50:
                # If page has a bordered table, the table markdown already
                # captures that data — only add text if there's significant
                # extra narrative beyond the table content
                if has_table:
                    table_chunk = next(
                        (c for c in chunks if c["page"] == page_num and c["chunk_type"] == "table"),
                        None
                    )
                    table_len = len(table_chunk["text"]) if table_chunk else 0
                    if len(page_text) - table_len > 150:
                        chunks.append({
                            "text": page_text,
                            "chunk_type": "text",
                            "page": page_num,
                            "source": source,
                        })
                else:
                    # No table — always keep the text (captures chart labels too)
                    chunks.append({
                        "text": page_text,
                        "chunk_type": "text",
                        "page": page_num,
                        "source": source,
                    })

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
