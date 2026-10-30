"""Reusable HTML building blocks for the HTML report renderers.

Every helper escapes the values it is given, so renderers can pass raw SET
output (which may contain attacker-controlled text) straight through.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Tones map to the status colours used throughout the report.
TONES = ("passed", "failed", "error", "neutral")


def esc(value: Any) -> str:
    """Escape a value for safe inclusion in HTML text or attribute content.

    Args:
        value: Any value; None renders as an empty string.

    Returns:
        The escaped string representation of the value.
    """
    if value is None:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def humanize(key: str) -> str:
    """Turn a snake_case key into a readable label.

    Args:
        key: A dict key such as "asr_at_injection".

    Returns:
        The label with underscores replaced and the first letter capitalised,
        e.g. "Asr at injection". Known acronyms are upper-cased.
    """
    # "set" in AVISE keys always abbreviates Security Evaluation Test.
    acronyms = {
        "asr",
        "l2",
        "mae",
        "psnr",
        "snr",
        "db",
        "id",
        "api",
        "url",
        "std",
        "set",
    }
    words = str(key).replace("_", " ").split()
    out = [w.upper() if w.lower() in acronyms else w for w in words]
    if out and out[0] == out[0].lower():
        out[0] = out[0].capitalize()
    return " ".join(out)


def fmt_number(value: Any, digits: int = 4) -> str:
    """Format a number compactly, leaving non-numbers unchanged.

    Args:
        value: Value to format.
        digits: Maximum number of decimal places for floats.

    Returns:
        The formatted string; "n/a" for None or NaN.
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    return str(value)


def fmt_percent(value: Any, digits: int = 2) -> str:
    """Format a 0-1 fraction as a percentage.

    Args:
        value: Fraction to format.
        digits: Decimal places of the percentage.

    Returns:
        E.g. "35.83%", or "n/a" when the value is missing or not numeric.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def tip(text: Optional[str]) -> str:
    """Render a "?" icon that reveals an explanation on hover or keyboard focus.

    Args:
        text: Plain-text explanation; falsy values render nothing.

    Returns:
        HTML span, or an empty string when there is no text.
    """
    if not text:
        return ""
    return (
        '<span class="tip" tabindex="0" aria-label="Explanation">?'
        f'<span class="tip-text" role="tooltip">{esc(text)}</span></span>'
    )


def value_html(value: Any, tips: Optional[Dict[str, str]] = None) -> str:
    """Render an arbitrary JSON value as HTML.

    Dicts become key/value tables, lists of dicts become column tables, lists
    of scalars become comma-separated text, and scalars are formatted.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        HTML fragment representing the value.
    """
    if isinstance(value, dict):
        return (
            kv_table(value, tips=tips) if value else '<span class="muted">empty</span>'
        )
    if isinstance(value, (list, tuple)):
        if not value:
            return '<span class="muted">empty</span>'
        if all(isinstance(v, dict) for v in value):
            return records_table(value, tips=tips)
        return esc(", ".join(fmt_number(v) for v in value))
    return esc(fmt_number(value))


def kv_table(
    data: dict,
    skip: Iterable[str] = (),
    tips: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Render a dict as a two-column key/value table, recursing into values.

    Args:
        data: Mapping to render.
        skip: Keys to leave out.
        tips: Optional key -> explanation mapping; matching keys get a "?" icon.
        labels: Optional key -> display label mapping, overriding ``humanize``.

    Returns:
        HTML table, or an empty string when nothing is left to show.
    """
    skip = set(skip)
    tips = tips or {}
    labels = labels or {}
    rows = "".join(
        f"<tr><th>{esc(labels.get(k) or humanize(k))}{tip(tips.get(k))}</th>"
        f"<td>{value_html(v, tips)}</td></tr>"
        for k, v in data.items()
        if k not in skip
    )
    return f'<table class="kv">{rows}</table>' if rows else ""


def records_table(
    records: Sequence[dict], tips: Optional[Dict[str, str]] = None
) -> str:
    """Render a list of dicts as a table with one column per key.

    Args:
        records: Rows to render; columns are the union of keys, in first-seen order.
        tips: Optional key -> explanation mapping; matching headers get a "?" icon.

    Returns:
        HTML table.
    """
    columns: List[str] = []
    for r in records:
        for k in r:
            if k not in columns:
                columns.append(k)
    tips = tips or {}
    head = "".join(f"<th>{esc(humanize(c))}{tip(tips.get(c))}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{value_html(r.get(c), tips)}</td>" for c in columns)
        + "</tr>"
        for r in records
    )
    return f'<table class="grid"><tr>{head}</tr>{body}</table>'


def badge(text: str, tone: str = "neutral") -> str:
    """Render a pill-shaped label.

    Args:
        text: Label text.
        tone: One of TONES; selects the colour.

    Returns:
        HTML span.
    """
    tone = tone if tone in TONES else "neutral"
    return f'<span class="status {tone}">{esc(text)}</span>'


def metric_tiles(tiles: Sequence[Tuple]) -> str:
    """Render a row of small metric cards.

    Args:
        tiles: (label, formatted value, tone) or (label, formatted value, tone,
            explanation) tuples; tone is one of TONES, and an explanation adds
            a "?" icon to the label.

    Returns:
        HTML block, or an empty string when there are no tiles.
    """
    if not tiles:
        return ""
    cells = "".join(
        f'<div class="tile {t[2] if t[2] in TONES else "neutral"}">'
        f'<div class="tile-value">{esc(t[1])}</div>'
        f'<div class="tile-label">{esc(t[0])}{tip(t[3] if len(t) > 3 else None)}</div></div>'
        for t in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def line_chart_svg(
    points: Sequence[Tuple[str, Optional[float]]],
    threshold: Optional[float] = None,
    y_label: str = "",
    threshold_label: str = "threshold",
) -> str:
    """Render a small 0-1 line chart as inline SVG.

    Points whose value is None are skipped. The y axis is shown as 0-100 %.

    Args:
        points: (x label, value in [0, 1] or None) pairs, in plotting order.
        threshold: Optional value drawn as a dashed horizontal line.
        y_label: Axis title for the y axis.
        threshold_label: Legend text for the threshold line.

    Returns:
        SVG markup, or an empty string when no point has a value.
    """
    plotted = [(x, y) for x, y in points if isinstance(y, (int, float))]
    if not plotted:
        return ""

    width, height = 560, 220
    left, right, top, bottom = 52, 20, 16, 40
    pw, ph = width - left - right, height - top - bottom
    n = len(plotted)

    def px(i: int) -> float:
        return left + (pw / 2 if n == 1 else i * pw / (n - 1))

    def py(v: float) -> float:
        return top + ph * (1 - max(0.0, min(1.0, v)))

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(y_label)}">'
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = py(tick)
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid-line"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">{int(tick * 100)}%</text>'
        )
    if y_label:
        parts.append(
            f'<text x="14" y="{top + ph / 2:.1f}" class="axis" text-anchor="middle" '
            f'transform="rotate(-90 14 {top + ph / 2:.1f})">{esc(y_label)}</text>'
        )
    if isinstance(threshold, (int, float)):
        y = py(threshold)
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="threshold"/>'
            f'<text x="{width - right}" y="{y - 6:.1f}" class="threshold-label" '
            f'text-anchor="end">{esc(threshold_label)} {fmt_percent(threshold, 0)}</text>'
        )
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{px(i):.1f},{py(v):.1f}"
        for i, (_, v) in enumerate(plotted)
    )
    parts.append(f'<path d="{path}" class="series"/>')
    for i, (x, v) in enumerate(plotted):
        above = isinstance(threshold, (int, float)) and v >= threshold
        parts.append(
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="5" class="point{" above" if above else ""}">'
            f"<title>{esc(x)}: {fmt_percent(v)}</title></circle>"
            f'<text x="{px(i):.1f}" y="{py(v) - 10:.1f}" class="point-label" '
            f'text-anchor="middle">{fmt_percent(v, 1)}</text>'
            f'<text x="{px(i):.1f}" y="{height - bottom + 18}" class="axis" '
            f'text-anchor="middle">{esc(x)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
