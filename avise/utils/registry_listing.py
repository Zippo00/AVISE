"""Render grouped registry contents as a tree for ``--SET-list`` and ``--connector-list``."""

import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, TextIO

from .ansi_color_codes import ansi_colors

if TYPE_CHECKING:
    from ..registry.registry import RegistryGroup

UNICODE_GLYPHS = {"branch": "├── ", "last": "└── ", "pipe": "│   ", "space": "    "}
ASCII_GLYPHS = {"branch": "|-- ", "last": "`-- ", "pipe": "|   ", "space": "    "}

# Below this many columns beside the names, descriptions go on the lines under each name.
MIN_SIDE_BY_SIDE_WIDTH = 30
# Stacked descriptions are wrapped to at least this width, even if lines overflow.
MIN_STACKED_WIDTH = 20


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
    With a width limit, descriptions wrap in a column beside the names, or on
    the lines under each name if that column would be too narrow; entries are
    then separated by a tree line.

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

    # Rows are (kind, prefix, text, count, continuation) with kind "title",
    # "heading", "group", "item" or "blank". The continuation is the prefix for
    # further lines of an item. Widths are measured before colour codes are added.
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
            continuation = prefix + (glyphs["space"] if last else glyphs["pipe"])
            if kind == "item":
                rows.append(("item", branch, child, None, continuation))
            else:
                rows.append(("group", branch, child.display_name, child.count(), ""))
                add_children(child, continuation)

    rows.append(("title", "", title, root.count(), ""))
    if root.items:
        rows.append(("blank", "", "", None, ""))
        add_children(_items_only(root), "")
    for group in root.sorted_groups():
        rows.append(("blank", "", "", None, ""))
        rows.append(("heading", "", group.display_name, group.count(), ""))
        add_children(group, "")

    item_widths = [len(row[1]) + len(row[2]) for row in rows if row[0] == "item"]
    description_column = max(item_widths, default=0) + 2
    stacked = (
        style.width is not None
        and style.width - description_column < MIN_SIDE_BY_SIDE_WIDTH
    )

    def description_lines(name: str, prefix: str, continuation: str) -> List[str]:
        """Lay out an item's description as plain-text line suffixes.

        Args:
            name: Registered name of the item.
            prefix: Tree glyphs before the item's name.
            continuation: Tree glyphs for the item's further lines.

        Returns:
            Text to append after the name on the item's first line, followed by
            the text of each further line (without the continuation prefix).
        """
        description = descriptions.get(name, "")
        if not description:
            return [""]
        if style.width is None:
            return [" " * (description_column - len(prefix) - len(name)) + description]
        if stacked:
            width = max(style.width - len(continuation), MIN_STACKED_WIDTH)
            return [""] + textwrap.wrap(description, width, break_on_hyphens=False)
        wrapped = textwrap.wrap(
            description, style.width - description_column, break_on_hyphens=False
        )
        indent = " " * (description_column - len(continuation))
        first = " " * (description_column - len(prefix) - len(name)) + wrapped[0]
        return [first] + [indent + line for line in wrapped[1:]]

    layouts = {
        row[2]: description_lines(row[2], row[1], row[4]) for row in rows if row[0] == "item"
    }
    spaced = any(len(layout) > 1 for layout in layouts.values())

    lines = []
    previous_kind = None
    for kind, prefix, text, count, continuation in rows:
        counter = _paint(f" ({count})", ansi_colors["dim"], style) if count is not None else ""
        tree = _paint(prefix, ansi_colors["dim"], style)
        if spaced and previous_kind == "item" and kind in ("item", "group"):
            # Separate multi-line entries with the tree's vertical line
            spacer = prefix[: -len(glyphs["branch"])] + glyphs["pipe"]
            lines.append(_paint(spacer.rstrip(), ansi_colors["dim"], style))
        previous_kind = kind
        if kind == "blank":
            lines.append("")
        elif kind == "title":
            lines.append(_paint(text, ansi_colors["bold"], style) + counter)
        elif kind == "heading":
            lines.append(_paint(text, ansi_colors["bold_cyan"], style) + counter)
        elif kind == "group":
            lines.append(tree + _paint(text, ansi_colors["bold"], style) + counter)
        else:
            first, *further = layouts[text]
            lines.append(tree + _paint(text, ansi_colors["bold"], style) + first)
            for line in further:
                lines.append(_paint(continuation, ansi_colors["dim"], style) + line)

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
