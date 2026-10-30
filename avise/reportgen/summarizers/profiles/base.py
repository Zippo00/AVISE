"""Summary profile interface and registry for the AI summarizer.

A SummaryProfile turns one domain's evaluation results into the facts and
prompts the summary model sees, and supplies the deterministic notes. The
summarizer itself is domain-agnostic: supporting a new kind of SET means
subclassing SummaryProfile and decorating it with ``@register_summary_profile``.

Profiles are selected like HTML result renderers: an explicit
``metadata["report_renderer"]`` hint on the results wins, then the first
profile whose ``matches()`` accepts the first result, then the default
(language model) profile.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

PROFILE_HINT_KEY = "report_renderer"

DEFAULT_SYSTEM_PROMPT = (
    "You are a cybersecurity analyst writing sections of a security assessment report. "
    "Base every statement strictly on the data provided in the request. Never invent numbers, "
    "results or test details. Follow the requested output format exactly."
)

_PROFILES: List["SummaryProfile"] = []
_DEFAULT: Optional["SummaryProfile"] = None


class SummaryProfile(ABC):
    """Domain-specific input and prompts for the AI summary.

    Attributes:
        name:          Registry name; matches the HTML result renderer of the
                       same domain so one metadata hint selects both.
        priority:      Higher values are asked to ``matches()`` first.
        system_prompt: System prompt sent with every request of this profile.
    """

    name: str = ""
    priority: int = 0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @abstractmethod
    def matches(self, result: Dict[str, Any]) -> bool:
        """Report whether this profile understands a result.

        Args:
            result: A result in dict form.

        Returns:
            True if this profile should summarise the results.
        """

    @abstractmethod
    def facts(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        configuration: Dict[str, Any],
        set_name: str = "",
    ) -> str:
        """Format the results as the factual input for the prompts.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            configuration: The report configuration (may be empty).
            set_name: The SET's name as written in its reports.

        Returns:
            Plain text inserted into the prompts.
        """

    @abstractmethod
    def issue_prompt(self, facts: str, set_name: str = "") -> str:
        """Build the request for the issue summary.

        Args:
            facts: Output of ``facts()``.
            set_name: The SET's name as written in its reports.

        Returns:
            The user prompt.
        """

    @abstractmethod
    def remediation_prompt(self, facts: str, set_name: str = "") -> str:
        """Build the request for the remediation recommendations.

        Args:
            facts: Output of ``facts()``.
            set_name: The SET's name as written in its reports.

        Returns:
            The user prompt.
        """

    def postprocess(self, text: str) -> str:
        """Clean up a model answer before it is stored.

        Args:
            text: The model's answer.

        Returns:
            The cleaned text; unchanged by default.
        """
        return text

    def canned_issue(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return a fixed issue summary that replaces the model call, if any.

        Args:
            results: All results in dict form.

        Returns:
            The fixed text, or None to query the model.
        """
        return None

    def canned_remediation(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return fixed remediation text that replaces the model call, if any.

        Args:
            results: All results in dict form.

        Returns:
            The fixed text, or None to query the model.
        """
        return None

    def notes(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        subcategory_runs: Dict[str, int],
        configuration: Dict[str, Any],
    ) -> List[str]:
        """Return deterministic notes appended to the summary.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            subcategory_runs: Number of results per vulnerability subcategory.
            configuration: The report configuration (may be empty).

        Returns:
            Note strings (may contain **bold** markdown).
        """
        return [
            "Automated tests may produce false positives or negatives; human review is advised "
            "for critical evaluations."
        ]


def register_summary_profile(cls):
    """Register a SummaryProfile subclass (class decorator).

    Args:
        cls: The profile class; it is instantiated once with no arguments.

    Returns:
        The class, unchanged.

    Raises:
        ValueError: If the class has no ``name`` or the name is already taken.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'.")
    if any(p.name == cls.name for p in _PROFILES):
        raise ValueError(f"A summary profile named '{cls.name}' is already registered.")
    _PROFILES.append(cls())
    _PROFILES.sort(key=lambda p: -p.priority)
    return cls


def set_default_profile(profile: SummaryProfile) -> None:
    """Set the profile used when no registered profile matches.

    Args:
        profile: The fallback profile instance.
    """
    global _DEFAULT
    _DEFAULT = profile


def select_summary_profile(results: List[Dict[str, Any]]) -> SummaryProfile:
    """Choose the profile for a list of results.

    Args:
        results: All results in dict form.

    Returns:
        The hinted profile, else the first matching one, else the default.

    Raises:
        RuntimeError: If nothing matches and no default profile is set.
    """
    first = results[0] if results else None
    if isinstance(first, dict):
        hint = (first.get("metadata") or {}).get(PROFILE_HINT_KEY)
        for p in _PROFILES + ([_DEFAULT] if _DEFAULT else []):
            if hint and p.name == hint:
                return p
        for p in _PROFILES:
            if p.matches(first):
                return p
    if _DEFAULT is None:
        raise RuntimeError("No AI summary profile matched and no default is set.")
    return _DEFAULT
