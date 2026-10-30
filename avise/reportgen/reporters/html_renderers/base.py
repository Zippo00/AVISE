"""Result renderer interface and registry for the HTML reporter.

The HTML reporter owns the page chrome shared by every SET (styles, header,
summary cards, AI summary, grouping). How a single result is drawn depends on
the domain that produced it, so that part is delegated to a ResultRenderer
selected per result. Supporting a new domain means subclassing ResultRenderer
and decorating it with ``@register_result_renderer``; nothing in the reporter
changes.

A SET can force a renderer by setting ``metadata["report_renderer"]`` on its
results to the renderer's ``name``. Otherwise each registered renderer's
``matches()`` is tried in descending ``priority`` order, and the language
model renderer is used when none match.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

RENDERER_HINT_KEY = "report_renderer"

_RENDERERS: List["ResultRenderer"] = []
_DEFAULT: Optional["ResultRenderer"] = None


class ResultRenderer(ABC):
    """Draw one domain's evaluation results inside the shared HTML report.

    Attributes:
        name:     Registry name; also the value SETs may put in
                  ``metadata["report_renderer"]`` to select this renderer.
        priority: Higher values are asked to ``matches()`` first.
    """

    name: str = ""
    priority: int = 0

    @abstractmethod
    def matches(self, result: Any) -> bool:
        """Report whether this renderer understands the given result.

        Args:
            result: An EvaluationResult object or its dict form.

        Returns:
            True if this renderer should draw the result.
        """

    @abstractmethod
    def render_item(self, result: Any, context: Dict[str, Any]) -> str:
        """Render one result as a collapsible ``set-item`` block.

        Args:
            result: An EvaluationResult object or its dict form.
            context: Report-wide information; contains "total_results" (int)
                and "configuration" (dict, the report's configuration).

        Returns:
            HTML fragment.
        """

    def render_sections(self, report_data: Any) -> str:
        """Render report-level sections placed before the results.

        Args:
            report_data: The ReportData being written.

        Returns:
            HTML fragment; empty by default.
        """
        return ""

    def header_meta(self, report_data: Any) -> List[str]:
        """Return extra items for the header's metadata line.

        Args:
            report_data: The ReportData being written.

        Returns:
            Plain-text items (escaped by the caller); empty by default.
        """
        return []

    def extra_css(self) -> str:
        """Return CSS rules this renderer needs beyond the shared stylesheet.

        Returns:
            CSS text; empty by default.
        """
        return ""


def as_dict(result: Any) -> Dict[str, Any]:
    """Return a result in dict form.

    Args:
        result: An object with ``to_dict()``, or a dict.

    Returns:
        The result as a dict (an empty dict for unsupported types).
    """
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {}


def register_result_renderer(cls: Type[ResultRenderer]) -> Type[ResultRenderer]:
    """Register a ResultRenderer subclass (class decorator).

    Args:
        cls: The renderer class; it is instantiated once with no arguments.

    Returns:
        The class, unchanged.

    Raises:
        ValueError: If the class has no ``name`` or the name is already taken.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'.")
    if any(r.name == cls.name for r in _RENDERERS):
        raise ValueError(f"A result renderer named '{cls.name}' is already registered.")
    _RENDERERS.append(cls())
    _RENDERERS.sort(key=lambda r: -r.priority)
    return cls


def set_default_renderer(renderer: ResultRenderer) -> None:
    """Set the renderer used when no registered renderer matches.

    Args:
        renderer: The fallback renderer instance.
    """
    global _DEFAULT
    _DEFAULT = renderer


def get_result_renderer(name: str) -> Optional[ResultRenderer]:
    """Look up a registered renderer by name.

    Args:
        name: Registry name.

    Returns:
        The renderer, or None if no renderer has that name.
    """
    for r in _RENDERERS:
        if r.name == name:
            return r
    if _DEFAULT is not None and _DEFAULT.name == name:
        return _DEFAULT
    return None


def select_result_renderer(result: Any) -> ResultRenderer:
    """Choose the renderer for a result.

    Args:
        result: An EvaluationResult object or its dict form; None selects the
            default renderer.

    Returns:
        The explicitly requested renderer, else the first matching one, else
        the default renderer.

    Raises:
        RuntimeError: If nothing matches and no default renderer is set.
    """
    if result is not None:
        metadata = as_dict(result).get("metadata") or {}
        hinted = get_result_renderer(metadata.get(RENDERER_HINT_KEY, ""))
        if hinted is not None:
            return hinted
        for r in _RENDERERS:
            if r.matches(result):
                return r
    if _DEFAULT is None:
        raise RuntimeError("No HTML result renderer matched and no default is set.")
    return _DEFAULT
