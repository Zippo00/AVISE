"""Rebuild ReportData objects from JSON reports written by JSONReporter.

JSON reports carry no explicit domain marker, so each registered schema
inspects the serialised results and the first match supplies the
EvaluationResult and ReportData classes. A new domain registers its schema
with ``register_report_schema``.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union

ResultMatcher = Callable[[Dict[str, Any]], bool]

_SCHEMAS: List[Tuple[str, ResultMatcher, Callable[[], Tuple[Type, Type]]]] = []

# SET display names that have since changed: (schema name, old name) -> current
# name. Older reports keep the old name in their JSON; it is replaced on load
# so re-rendered reports show the current one.
_RENAMED_SETS: Dict[Tuple[str, str], str] = {
    ("continuallearning", "Backdoor"): "Continual Learning Backdoor",
}


def register_report_schema(
    name: str, matcher: ResultMatcher, classes: Callable[[], Tuple[Type, Type]]
) -> None:
    """Register how to recognise and rebuild one domain's JSON reports.

    Args:
        name: Domain name, used in error messages.
        matcher: Returns True for a serialised result dict of this domain.
        classes: Returns the domain's (EvaluationResult, ReportData) classes;
            called lazily so registering does not import the pipeline.
    """
    _SCHEMAS.append((name, matcher, classes))


def _lm_classes() -> Tuple[Type, Type]:
    """Return the language model schema classes.

    Returns:
        The (EvaluationResult, ReportData) classes.
    """
    from ..pipelines.languagemodel.schema import EvaluationResult, ReportData

    return EvaluationResult, ReportData


def _cl_classes() -> Tuple[Type, Type]:
    """Return the continual learning schema classes.

    Returns:
        The (EvaluationResult, ReportData) classes.
    """
    from ..pipelines.continuallearning.schema import EvaluationResult, ReportData

    return EvaluationResult, ReportData


register_report_schema(
    "continuallearning",
    lambda r: "baseline_metrics" in r and "prompt" not in r,
    _cl_classes,
)
register_report_schema("languagemodel", lambda r: "prompt" in r, _lm_classes)


def _build(cls: Type, data: Dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring keys it does not declare.

    Args:
        cls: Dataclass type.
        data: Field values.

    Returns:
        The dataclass instance.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def report_data_from_dict(report: Dict[str, Any]) -> Any:
    """Rebuild a ReportData object from a parsed JSON report.

    Args:
        report: The dict JSONReporter wrote; results are read from
            "set_category" (grouped) or "results" (ungrouped).

    Returns:
        A ReportData instance of the matching domain.

    Raises:
        ValueError: If a required top-level key is missing or no registered
            schema recognises the results.
    """
    missing = [k for k in ("set_name", "summary") if k not in report]
    if missing:
        raise ValueError(f"Not an AVISE JSON report: missing key(s) {missing}.")

    grouped = "set_category" in report
    rows = (
        [r for group in report["set_category"].values() for r in group]
        if grouped
        else list(report.get("results", []))
    )

    schema = None
    for name, matcher, classes in _SCHEMAS:
        if not rows or matcher(rows[0]):
            schema = (name, classes())
            break
    if schema is None:
        known = ", ".join(n for n, _, _ in _SCHEMAS)
        raise ValueError(
            f"No report schema recognises result keys {sorted(rows[0])}; "
            f"registered schemas: {known}."
        )

    result_cls, report_cls = schema[1]
    set_name = _RENAMED_SETS.get((schema[0], report["set_name"]), report["set_name"])
    return report_cls(
        set_name=set_name,
        timestamp=report.get("timestamp", ""),
        execution_time_seconds=report.get("execution_time_seconds"),
        summary=report["summary"],
        results=[_build(result_cls, r) for r in rows],
        configuration=report.get("configuration", {}),
        ai_summary=report.get("ai_summary"),
        group_results=grouped,
    )


def load_report_data(path: Union[str, Path]) -> Any:
    """Load a JSON report file and rebuild its ReportData object.

    Args:
        path: Path to a JSON report.

    Returns:
        A ReportData instance of the matching domain.

    Raises:
        ValueError: If the file is not valid JSON or not an AVISE report.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    return report_data_from_dict(report)
