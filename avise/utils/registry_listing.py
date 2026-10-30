"""Render grouped registry contents as a tree for ``--SET-list`` and ``--connector-list``."""

import os
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, TextIO

from .ansi_color_codes import ansi_colors

if TYPE_CHECKING:
    from ..registry.registry import RegistryGroup

UNICODE_GLYPHS = {"branch": "├── ", "last": "└── ", "pipe": "│   ", "space": "    ", "ellipsis": "…"}
ASCII_GLYPHS = {"branch": "|-- ", "last": "`-- ", "pipe": "|   ", "space": "    ", "ellipsis": "..."}

# Descriptions are dropped instead of truncated below this many columns.
MIN_DESCRIPTION_WIDTH = 15


@dataclass
class TerminalStyle:
    """How listing output should be styled for the stream it is printed to.

    Attributes:
        color: Whether to emit ANSI colour codes.
        unicode: Whether to draw the tree with box-drawing characters.
        width: Line width to truncate descriptions to, or None for no limit.
    """

    color: bool = False
    unicode: bool = True
    width: Optional[int] = None


def detect_terminal_style(stream: Optional[TextIO] = None) -> TerminalStyle:
    """Pick colour, glyph set and width for a stream.

    Colour and truncation are only used on an interactive terminal; colour is
    also disabled when the ``NO_COLOR`` environment variable is set.

    Args:
        stream: Stream the listing is printed to. Defaults to ``sys.stdout``.

    Returns:
        The TerminalStyle to render with.
    """
    stream = stream or sys.stdout
    try:
        is_tty = stream.isatty()
    except (AttributeError, ValueError):
        is_tty = False
    try:
        "".join(UNICODE_GLYPHS.values()).encode(getattr(stream, "encoding", None) or "ascii")
        unicode = True
    except (UnicodeEncodeError, LookupError):
        unicode = False
    return TerminalStyle(
        color=is_tty and not os.environ.get("NO_COLOR"),
        unicode=unicode,
        width=shutil.get_terminal_size().columns if is_tty else None,
    )


def _paint(text: str, codes: str, style: TerminalStyle) -> str:
    """Wrap text in ANSI codes when colour is enabled.

    Args:
        text: Text to colour.
        codes: ANSI escape codes to apply.
        style: Output style.

    Returns:
        The text, coloured if ``style.color`` is True.
    """
    return f"{codes}{text}{ansi_colors['reset']}" if style.color else text


def _clean_description(description: object) -> str:
    """Collapse a description to a single line.

    Args:
        description: A class's ``description`` attribute (any type).

    Returns:
        The description with whitespace collapsed, or "" if it is not a string.
    """
    return " ".join(description.split()) if isinstance(description, str) else ""


def render_registry_tree(
    root: "RegistryGroup",
    title: str,
    descriptions: Optional[Dict[str, str]] = None,
    style: Optional[TerminalStyle] = None,
    hint: str = "",
) -> str:
    """Render a grouped registry as a titled tree with aligned descriptions.

    Top-level groups are printed as headings; their subgroups and names are
    drawn as a tree below them. Groups and names are sorted alphabetically.

    Args:
        root: Root group from ``Registry.grouped()``.
        title: Heading printed above the tree.
        descriptions: Maps registered names to descriptions shown beside them.
        style: Output style. Defaults to plain Unicode text with no width limit.
        hint: Optional line printed (dimmed) below the tree.

    Returns:
        The rendered listing, ending in a newline.
    """
    style = style or TerminalStyle()
    glyphs = UNICODE_GLYPHS if style.unicode else ASCII_GLYPHS
    descriptions = {k: _clean_description(v) for k, v in (descriptions or {}).items()}

    # Rows are (kind, prefix, text, count) with kind "title", "heading", "group",
    # "item" or "blank"; widths are measured before colour codes are added.
    rows: List[tuple] = []

    def add_children(group: "RegistryGroup", prefix: str) -> None:
        """Append rows for a group's names and subgroups, recursing into subgroups.

        Args:
            group: Group whose contents are added.
            prefix: Tree glyphs inherited from the parent levels.
        """
        children = [("item", name) for name in sorted(group.items, key=str.casefold)]
        children += [("group", sub) for sub in group.sorted_groups()]
        for i, (kind, child) in enumerate(children):
            last = i == len(children) - 1
            branch = prefix + (glyphs["last"] if last else glyphs["branch"])
            if kind == "item":
                rows.append(("item", branch, child, None))
            else:
                rows.append(("group", branch, child.display_name, child.count()))
                add_children(child, prefix + (glyphs["space"] if last else glyphs["pipe"]))

    rows.append(("title", "", title, root.count()))
    if root.items:
        rows.append(("blank", "", "", None))
        add_children(_items_only(root), "")
    for group in root.sorted_groups():
        rows.append(("blank", "", "", None))
        rows.append(("heading", "", group.display_name, group.count()))
        add_children(group, "")

    item_widths = [len(prefix) + len(text) for kind, prefix, text, _ in rows if kind == "item"]
    description_column = max(item_widths, default=0) + 2

    lines = []
    for kind, prefix, text, count in rows:
        counter = _paint(f" ({count})", ansi_colors["dim"], style) if count is not None else ""
        tree = _paint(prefix, ansi_colors["dim"], style)
        if kind == "blank":
            lines.append("")
        elif kind == "title":
            lines.append(_paint(text, ansi_colors["bold"], style) + counter)
        elif kind == "heading":
            lines.append(_paint(text, ansi_colors["bold_cyan"], style) + counter)
        elif kind == "group":
            lines.append(tree + _paint(text, ansi_colors["bold"], style) + counter)
        else:
            line = tree + _paint(text, ansi_colors["bold"], style)
            description = descriptions.get(text, "")
            available = (
                style.width - description_column if style.width is not None else None
            )
            if description and (available is None or available >= MIN_DESCRIPTION_WIDTH):
                if available is not None and len(description) > available:
                    cut = available - len(glyphs["ellipsis"])
                    description = description[:cut].rstrip() + glyphs["ellipsis"]
                padding = " " * (description_column - len(prefix) - len(text))
                line += padding + description
            lines.append(line)

    if hint:
        lines += ["", _paint(hint, ansi_colors["dim"], style)]
    return "\n".join(lines) + "\n"


def _items_only(group: "RegistryGroup") -> "RegistryGroup":
    """Copy a group without its subgroups.

    Args:
        group: Group to copy.

    Returns:
        A group of the same type holding only ``group.items``.
    """
    return type(group)(key=group.key, display_name=group.display_name, items=list(group.items))


def list_categories(root: "RegistryGroup") -> List[str]:
    """List every category of a grouped registry for help and error messages.

    Args:
        root: Root group from ``Registry.grouped()``.

    Returns:
        Entries like "single_turn (Single-turn)", depth first; the display
        name is omitted when it equals the key.
    """
    return [
        group.key if group.key == group.display_name else f"{group.key} ({group.display_name})"
        for group in root.iter_groups()
    ]
