"""HTML result renderer for continual learning (CL) SETs.

A CL result's substance is its ``detections`` mapping: one entry per evaluator,
each shaped ``{"susceptible": bool, "findings": {...}}`` (``susceptible`` is
absent for informational evaluators). Findings differ per evaluator, so each
evaluator gets a DetectionRenderer, looked up by evaluator name.

Evaluators without a registered DetectionRenderer are still rendered in full by
the generic fallback (verdict badge, reason, and every recorded value as a
table). A new CL SET therefore produces a complete report without any change
here; registering a DetectionRenderer only improves how its findings look::

    @register_detection_renderer
    class MyEvaluatorRenderer(DetectionRenderer):
        evaluators = ("my_evaluator",)
        title = "My Evaluator"
        description = "What the evaluator measures, for the title's ? icon."
        glossary = {"score": "What the 'score' value means."}

        def headline(self, findings, context):
            return [("Score", fmt_number(findings.get("score")), "neutral")]

The evaluators a case's pass/fail status is derived from are read from the
result's ``metadata["verdict_basis"]`` (a list of evaluator names); results
written before that key existed fall back to renderers marked ``primary``.
Configuration keys and task stages are explained through CONFIG_GLOSSARY and
STAGE_DESCRIPTIONS, which SETs may extend with ``register_config_terms`` and
``register_stage_descriptions``.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from .base import ResultRenderer, as_dict, register_result_renderer
from .components import (
    badge,
    esc,
    fmt_number,
    fmt_percent,
    humanize,
    kv_table,
    line_chart_svg,
    metric_tiles,
    tip,
)

Tile = Tuple[str, str, str, Optional[str]]

_DETECTION_RENDERERS: Dict[str, "DetectionRenderer"] = {}

# Metadata keys already shown elsewhere in the report.
# poisoned_inputs is a list of digests for offline scoring of defences; it is
# kept in the JSON report but not shown in HTML.
_HIDDEN_METADATA = {
    "vulnerability_subcategory", "report_renderer", "verdict_basis", "poisoned_inputs",
}

VERDICT_BASIS_KEY = "verdict_basis"


# ---------------------------------------------------------------------------
# Glossaries
# ---------------------------------------------------------------------------

CONFIG_GLOSSARY: Dict[str, str] = {
    "model_config": "Connector configuration file: tells AVISE how to reach and talk to the target system.",
    "set_config": "SET (Security Evaluation Test) configuration file this run was started with.",
    "target_modality": "Type of data the target model works on (image, audio, text, numeric, video or multimodal).",
    "trigger_type": (
        "How the backdoor trigger is added to an input. 'blend' mixes a fixed pattern over the "
        "whole input: x' = (1 - alpha) * x + alpha * pattern. 'static_feature_perturbation' stamps "
        "the same small patch onto every poisoned input. 'label_only' changes labels without "
        "adding any trigger (a control)."
    ),
    "poison_rate": (
        "Fraction of the eligible training samples in a poisoned stage that the attacker modifies: "
        "the trigger is added and the label is changed to the target label. 0.4 = 40% of eligible samples."
    ),
    "target_label": "The class the attacker wants every triggered input to be classified as.",
    "source_label": (
        "Class(es) whose samples are poisoned and attacked. Empty (n/a) means an all-to-one attack: "
        "samples of every class except the target class are eligible."
    ),
    "poison_stages": (
        "Which training stages receive poisoned data. 'inject' = only the inject stage. "
        "'all_train' = the inject stage and every drift stage after it (an attacker who keeps "
        "feeding the deployed system)."
    ),
    "poisoning_seed_value": "Random seed that chooses which samples are poisoned, so a run can be reproduced exactly.",
    "set_data_already_poisoned": (
        "Yes if the data supplied to the SET was poisoned before the test, in which case AVISE does not add a "
        "trigger itself (and cannot measure perceptual distance)."
    ),
    "human_in_the_loop": "Yes if an operator had to confirm each stage before it was sent to the target.",
    "data_schema": "Names of the fields in each data sample that hold the input and its label.",
    "label_field": "Name of the field in each sample that holds the class label.",
    "input_field": "Name of the field in each sample that holds the model input (e.g. the image).",
    "trigger_config": "Parameters of the chosen trigger type.",
    "blend_alpha": (
        "Blend strength alpha in x' = (1 - alpha) * x + alpha * pattern. 0 = no trigger, "
        "1 = the input is completely replaced by the pattern. Small values are hard to see."
    ),
    "blend_pattern": "Pattern blended into inputs: 'noise' = a fixed random-noise image; 'sinusoid' = a fixed wave pattern.",
    "blend_seed": "Random seed that generates the blend pattern, so the same trigger is used in every run.",
    "eval_configs": "Thresholds used by each evaluator to decide its result.",
    "threshold": "Value at which the evaluator's result changes (see that evaluator's explanation).",
    "set_cases": "The SET cases in this run and the sequence of tasks each one sends to the target.",
}

# Display labels that differ from the humanised key.
CONFIG_LABELS: Dict[str, str] = {
    "model_config": "Connector config",
    "set_config": "SET config",
}

# Value formatters for configuration keys whose raw value reads poorly.
CONFIG_VALUE_FORMATTERS: Dict[str, Callable[[Any], Any]] = {
    "source_label": lambda v: "All classes except target" if v is None else v,
}

StageDescription = Union[str, Callable[[bool, bool], str]]

STAGE_DESCRIPTIONS: Dict[str, StageDescription] = {
    "drift": lambda after_inject, poisoned: (
        "Training task with poisoned data (poison_stages: all_train) after the attack."
        if poisoned
        else "Benign training task after the attack. The backdoor is re-tested to see whether it "
        "survives the model learning new data."
        if after_inject
        else "Benign training task before the attack. Gives the model prior knowledge, as a "
        "deployed system would have."
    ),
    "baseline": "No training. Measures the model's accuracy on clean data before any poisoning, as the reference point.",
    "inject": lambda after_inject, poisoned: (
        "ATTACK. Training task in which part of the data carries the trigger and is relabelled to "
        "the target label. The attack success rate is measured right after it."
    ),
    "eval": "No training. Final measurement of clean accuracy and attack success rate after all training.",
}


def register_config_terms(terms: Dict[str, str]) -> None:
    """Add or replace explanations shown next to configuration keys.

    Args:
        terms: Mapping of configuration key to plain-text explanation.
    """
    CONFIG_GLOSSARY.update(terms)


def register_config_display(
    labels: Optional[Dict[str, str]] = None,
    formatters: Optional[Dict[str, Callable[[Any], Any]]] = None,
) -> None:
    """Add display labels or value formatters for configuration keys.

    Args:
        labels: Mapping of configuration key to the label shown in the report.
        formatters: Mapping of configuration key to a function that turns the
            raw value into the value shown.
    """
    CONFIG_LABELS.update(labels or {})
    CONFIG_VALUE_FORMATTERS.update(formatters or {})


def register_stage_descriptions(descriptions: Dict[str, StageDescription]) -> None:
    """Add or replace descriptions of task stages in the task sequence table.

    Args:
        descriptions: Mapping of stage name to a description string, or to a
            callable ``(after_inject, poisoned) -> str``.
    """
    STAGE_DESCRIPTIONS.update(descriptions)


def _describe_stage(stage: str, after_inject: bool, poisoned: bool) -> str:
    """Return the description of a task stage.

    Args:
        stage: Stage name.
        after_inject: Whether an inject stage precedes this one.
        poisoned: Whether this stage receives poisoned data.

    Returns:
        The description, or "" for unknown stages.
    """
    desc = STAGE_DESCRIPTIONS.get(stage, "")
    return desc(after_inject, poisoned) if callable(desc) else desc


# ---------------------------------------------------------------------------
# Detection renderers
# ---------------------------------------------------------------------------


class DetectionRenderer:
    """Render one evaluator's entry in a CL result's ``detections``.

    Subclasses override any of the hook methods; the base implementation is
    the generic fallback used for unknown evaluators.

    Attributes:
        evaluators:   Evaluator names (keys of ``detections``) this renderer handles.
        title:        Display name; defaults to the humanised evaluator name.
        description:  Plain-language explanation shown by the title's "?" icon.
        glossary:     Findings key -> explanation, shown by "?" icons in tables.
        badge_labels: (label when susceptible, label when not susceptible).
        order:        Sort key among a result's detections (lower renders first).
        primary:      Treat this evaluator as deciding the verdict when a result
                      does not name its ``verdict_basis``.
    """

    evaluators: Sequence[str] = ()
    title: Optional[str] = None
    description: Optional[str] = None
    glossary: Dict[str, str] = {}
    badge_labels: Tuple[str, str] = ("susceptible", "not susceptible")
    order: int = 100
    primary: bool = False

    def headline(self, findings: Dict[str, Any], context: Dict[str, Any]) -> List[Tile]:
        """Return the key figures shown at the top of the result.

        Args:
            findings: The evaluator's findings dict.
            context: Render context (see ContinualLearningResultRenderer.render_item).

        Returns:
            (label, formatted value, tone, explanation or None) tiles; none by default.
        """
        return []

    def summary_line(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return a one-line digest shown on the collapsed evaluator card.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            Plain text; empty by default.
        """
        return ""

    def body(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return a purpose-built view of the findings (charts, tiles).

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            HTML fragment; empty by default.
        """
        return ""

    def detail_skip(self, findings: Dict[str, Any]) -> Sequence[str]:
        """Return findings keys left out of the full values table.

        Args:
            findings: The evaluator's findings dict.

        Returns:
            Keys to skip; "reason" and "evaluator" are always skipped because
            they are shown elsewhere.
        """
        return ()

    def badge_for(
        self, susceptible: Optional[bool], findings: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Return the label and tone of the card's result badge.

        Args:
            susceptible: The evaluator's verdict, or None when it gives none.
            findings: The evaluator's findings dict.

        Returns:
            A (label, tone) tuple; by default ``badge_labels`` with "failed" /
            "passed" tones, or ("informational", "neutral") without a verdict.
        """
        if susceptible is True:
            return self.badge_labels[0], "failed"
        if susceptible is False:
            return self.badge_labels[1], "passed"
        return "informational", "neutral"

    def verdict(
        self, findings: Dict[str, Any], susceptible: Optional[bool], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Explain how this evaluator's findings decide the case's verdict.

        Called only for evaluators named in the result's verdict basis.

        Args:
            findings: The evaluator's findings dict.
            susceptible: The evaluator's verdict.
            context: Render context.

        Returns:
            None for the generic explanation, or a dict with:
                rule:       Plain-text statement of the decision rule.
                conditions: List of {"condition", "required", "measured",
                            "met"} dicts; met is True, False or None (not applicable).
                conclusion: Plain-text sentence explaining the outcome.
        """
        return None

    def render(
        self, name: str, entry: Any, context: Dict[str, Any], decisive: bool = False
    ) -> str:
        """Render the evaluator's collapsible card.

        Args:
            name: Evaluator name (the ``detections`` key).
            entry: The detections value for that evaluator.
            context: Render context.
            decisive: Whether this evaluator decides the case's verdict.

        Returns:
            HTML fragment.
        """
        susceptible, findings = split_detection(entry)
        title = self.title or humanize(name)
        verdict = badge(*self.badge_for(susceptible, findings))
        role = (
            '<span class="role decisive">decides the verdict</span>'
            if decisive
            else '<span class="role">supporting measurement</span>'
        )
        line = self.summary_line(findings, context)
        line_html = f'<div class="detection-line">{esc(line)}</div>' if line else ""

        reason = findings.get("reason")
        reason_html = f'<div class="detection-reason">{esc(reason)}</div>' if reason else ""
        body = self.body(findings, context)
        table = kv_table(
            findings, skip={"reason", "evaluator", *self.detail_skip(findings)}, tips=self.glossary
        )
        if table:
            table = (
                '<details class="raw"><summary>All recorded values</summary>'
                f"{table}</details>"
            )
        return f"""
            <details class="detection">
                <summary>
                    <div class="detection-header">
                        <div>
                            <span class="detection-name">{esc(title)}</span>{tip(self.description)}
                            {role}
                            {line_html}
                        </div>
                        {verdict}
                    </div>
                </summary>
                <div class="detection-body">
                    {reason_html}
                    {body}
                    {table}
                </div>
            </details>"""


_FALLBACK = DetectionRenderer()


def register_detection_renderer(cls: Type[DetectionRenderer]) -> Type[DetectionRenderer]:
    """Register a DetectionRenderer subclass for its evaluators (class decorator).

    Args:
        cls: The renderer class; it is instantiated once with no arguments.

    Returns:
        The class, unchanged.

    Raises:
        ValueError: If the class lists no evaluators.
    """
    if not cls.evaluators:
        raise ValueError(f"{cls.__name__} must list at least one evaluator name.")
    instance = cls()
    for name in cls.evaluators:
        _DETECTION_RENDERERS[name] = instance
    return cls


def get_detection_renderer(evaluator: str) -> DetectionRenderer:
    """Return the renderer for an evaluator, or the generic fallback.

    Args:
        evaluator: Evaluator name.

    Returns:
        The registered DetectionRenderer, else the fallback renderer.
    """
    return _DETECTION_RENDERERS.get(evaluator, _FALLBACK)


def split_detection(entry: Any) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Separate a detections entry into its verdict and findings.

    Args:
        entry: A detections value; normally ``{"susceptible", "findings"}``,
            but any dict (treated as the findings) or scalar is accepted.

    Returns:
        A (susceptible, findings) tuple:
            susceptible: The verdict, or None when the entry carries none.
            findings:    The findings dict (a scalar is wrapped as {"value": ...}).
    """
    if not isinstance(entry, dict):
        return None, {"value": entry}
    if isinstance(entry.get("findings"), dict):
        verdict = entry.get("susceptible")
        if verdict is None:
            verdict = entry["findings"].get("susceptible")
        return (verdict if isinstance(verdict, bool) else None), entry["findings"]
    if "findings" in entry and entry["findings"] is None:
        verdict = entry.get("susceptible")
        return (verdict if isinstance(verdict, bool) else None), {}
    verdict = entry.get("susceptible")
    return (verdict if isinstance(verdict, bool) else None), entry


def _ordered_detections(detections: Dict[str, Any]) -> List[Tuple[str, Any, DetectionRenderer]]:
    """Pair each detection with its renderer, sorted by renderer order.

    Args:
        detections: A result's detections mapping.

    Returns:
        (evaluator name, entry, renderer) triples; ties keep their original order.
    """
    items = [(k, v, get_detection_renderer(k)) for k, v in detections.items()]
    return sorted(items, key=lambda t: t[2].order)


def headline_tiles(detections: Dict[str, Any], context: Dict[str, Any]) -> List[Tile]:
    """Collect the headline tiles of every detection in render order.

    Args:
        detections: A result's detections mapping.
        context: Render context.

    Returns:
        (label, formatted value, tone, explanation) tiles.
    """
    tiles: List[Tile] = []
    for _, entry, renderer in _ordered_detections(detections):
        tiles.extend(renderer.headline(split_detection(entry)[1], context))
    return tiles


def verdict_basis(data: Dict[str, Any]) -> List[str]:
    """Return the evaluators a result's status was derived from.

    Args:
        data: A result in dict form.

    Returns:
        Names from ``metadata["verdict_basis"]`` when recorded, otherwise the
        result's evaluators whose renderer is marked ``primary``.
    """
    metadata = data.get("metadata") or {}
    basis = metadata.get(VERDICT_BASIS_KEY)
    if isinstance(basis, str):
        return [basis]
    if isinstance(basis, (list, tuple)):
        return [b for b in basis if isinstance(b, str)]
    return [k for k in (data.get("detections") or {}) if get_detection_renderer(k).primary]


def _case_plan(configuration: Dict[str, Any], set_id: str) -> Optional[Dict[str, Any]]:
    """Find a case's described task sequence in the report configuration.

    Args:
        configuration: The report configuration.
        set_id: The case id.

    Returns:
        The matching entry of ``configuration["set_cases"]``, or None.
    """
    for case in configuration.get("set_cases") or []:
        if isinstance(case, dict) and case.get("id") == set_id:
            return case
    return None


def _stage_name(task: Dict[str, Any]) -> str:
    """Return a task's stage name from either the report or the config spelling.

    Args:
        task: A task_sequence entry.

    Returns:
        The stage name, or "" when absent.
    """
    return str(task.get("stage") or task.get("task_stage") or "")


def _short_path(value: Any) -> Tuple[str, str]:
    """Shorten a data reference for display.

    Args:
        value: A task's data or eval_data value.

    Returns:
        A (display text, full text) tuple; display is the file name for paths.
    """
    if value is None:
        return "-", ""
    text = str(value)
    if "/" in text or "\\" in text:
        return os.path.basename(text.rstrip("/\\")) or text, text
    return text, text


@register_result_renderer
class ContinualLearningResultRenderer(ResultRenderer):
    """Render results of CL SETs: evaluator findings rather than prompt/response."""

    name = "continuallearning"
    priority = 10

    def matches(self, result: Any) -> bool:
        """Report whether the result comes from a CL SET.

        Args:
            result: An EvaluationResult object or its dict form.

        Returns:
            True for objects defined under ``avise.pipelines.continuallearning``,
            and for dicts with "baseline_metrics" but no "prompt" (the CL
            EvaluationResult serialisation).
        """
        if isinstance(result, dict):
            return "baseline_metrics" in result and "prompt" not in result
        return type(result).__module__.startswith("avise.pipelines.continuallearning")

    def render_item(self, result: Any, context: Dict[str, Any]) -> str:
        """Render one CL result as a collapsible block.

        The detection hooks receive a context dict extending the reporter's
        with "set_id", "case" (this case's entry of configuration["set_cases"],
        or None) and "target_label".

        Args:
            result: A CL EvaluationResult or its dict form.
            context: Report-wide information; a report with a single result
                renders it expanded.

        Returns:
            HTML fragment.
        """
        data = as_dict(result)
        metadata = data.get("metadata") or {}
        detections = data.get("detections") or {}
        status = data.get("status", "error")
        configuration = context.get("configuration") or {}
        ctx = {
            **context,
            "set_id": data.get("set_id"),
            "case": _case_plan(configuration, data.get("set_id")),
            "target_label": configuration.get("target_label"),
        }
        basis = verdict_basis(data)

        label = metadata.get("attack_type") or metadata.get("description") or ""
        label = f" - {esc(label)}" if label else ""
        is_open = " open" if context.get("total_results") == 1 else ""

        tiles = headline_tiles(detections, ctx)
        tiles_html = (
            f'<div class="subsection-title">Key measurements</div>{metric_tiles(tiles)}'
            if tiles else ""
        )

        if detections:
            cards = "".join(
                r.render(k, v, ctx, decisive=k in basis)
                for k, v, r in _ordered_detections(detections)
            )
            findings_html = f"""
            <div class="findings-header">
                <span>Evaluator Findings</span>
                <span class="findings-sub">{len(detections)} evaluator(s) &middot; click a card for details</span>
            </div>
            {cards}"""
        else:
            findings_html = ""

        extras = ""
        if data.get("baseline_metrics"):
            extras += (
                '<div class="label-sm">Baseline metrics</div>'
                f'{kv_table(data["baseline_metrics"])}'
            )
        shown_metadata = {k: v for k, v in metadata.items() if k not in _HIDDEN_METADATA}
        if shown_metadata:
            extras += f'<div class="label-sm">Case metadata</div>{kv_table(shown_metadata)}'

        return f"""
        <details class="set-item"{is_open}>
            <summary>
                <div class="set-header">
                    <span class="set-id">{esc(data.get("set_id", ""))}{label}</span>
                    {badge(status, status)}
                </div>
            </summary>
            {self._verdict_box(data, basis, ctx)}
            {tiles_html}
            {findings_html}
            {extras}
        </details>
"""

    def _verdict_box(self, data: Dict[str, Any], basis: List[str], ctx: Dict[str, Any]) -> str:
        """Render the explanation of why the case passed, failed or was inconclusive.

        Args:
            data: The result in dict form.
            basis: Evaluators the status was derived from.
            ctx: Render context.

        Returns:
            HTML fragment.
        """
        status = data.get("status", "error")
        detections = data.get("detections") or {}
        reason = data.get("reason", "")

        if status not in ("passed", "failed"):
            return f"""
            <div class="verdict error">
                <div class="verdict-title">Why this case is INCONCLUSIVE</div>
                <p>The test could not be completed, so no verdict was reached.</p>
                <p class="verdict-reason">{esc(reason)}</p>
            </div>"""

        headline = (
            "The target withstood the attack."
            if status == "passed"
            else "The target is vulnerable to this attack."
        )
        parts = []
        for name in basis:
            if name not in detections:
                continue
            renderer = get_detection_renderer(name)
            susceptible, findings = split_detection(detections[name])
            title = renderer.title or humanize(name)
            explained = renderer.verdict(findings, susceptible, ctx)
            if explained is None:
                outcome = {True: "susceptible", False: "not susceptible"}.get(susceptible, "no verdict")
                parts.append(
                    f"<p>The verdict follows the <strong>{esc(title)}</strong> evaluator, "
                    f"which found the target <strong>{esc(outcome)}</strong>.</p>"
                )
                continue
            rows = ""
            for i, c in enumerate(explained.get("conditions") or [], 1):
                met = c.get("met")
                mark = (
                    '<span class="mark ok">&#10003; met</span>' if met is True
                    else '<span class="mark no">&#10007; not met</span>' if met is False
                    else '<span class="mark na">not applicable</span>'
                )
                rows += (
                    f"<tr><td>{i}</td><td>{esc(c.get('condition'))}</td>"
                    f"<td>{esc(c.get('required'))}</td><td><strong>{esc(c.get('measured'))}</strong></td>"
                    f"<td>{mark}</td></tr>"
                )
            table = (
                '<table class="grid verdict-table"><tr><th>#</th><th>Condition</th>'
                f"<th>Required</th><th>Measured</th><th>Result</th></tr>{rows}</table>"
                if rows else ""
            )
            parts.append(
                f"<p>{esc(explained.get('rule', ''))}</p>{table}"
                f'<p class="verdict-conclusion">{esc(explained.get("conclusion", ""))}</p>'
            )

        if not parts:
            parts.append(
                f"<p>{esc(reason)}</p><p class=\"note\">This report does not record which "
                "evaluator the verdict was derived from.</p>"
            )

        supporting = [
            r.title or humanize(k)
            for k, _, r in _ordered_detections(detections)
            if k not in basis
        ]
        footnote = (
            f'<p class="note">{esc(", ".join(supporting))} describe side effects and stealth of '
            "the attack. They are reported for context and do not change the verdict.</p>"
            if supporting else ""
        )
        return f"""
            <div class="verdict {status}">
                <div class="verdict-title">Why this case {status.upper()}</div>
                <p class="verdict-headline">{headline}</p>
                {"".join(parts)}
                {footnote}
            </div>"""

    def render_sections(self, report_data: Any) -> str:
        """Render the test configuration and, for multi-case reports, an overview table.

        Args:
            report_data: The ReportData being written.

        Returns:
            HTML fragment.
        """
        html = ""
        config = dict(getattr(report_data, "configuration", {}) or {})
        if config:
            cases = [c for c in config.pop("set_cases", None) or [] if isinstance(c, dict)]
            shown = {
                k: CONFIG_VALUE_FORMATTERS[k](v) if k in CONFIG_VALUE_FORMATTERS else v
                for k, v in config.items()
            }
            sequences = "".join(self._task_sequence(c, config, len(cases) > 1) for c in cases)
            if sequences:
                sequences = (
                    f'<div class="subsection-title">Task sequence{tip(CONFIG_GLOSSARY.get("set_cases"))}</div>'
                    '<p class="note">The order in which data was sent to the target. The '
                    "highlighted stage is where the attack happens.</p>"
                    f"{sequences}"
                )
            html += f"""
    <details class="category">
        <summary>
            <div class="category-header">
                <h2>SET Configuration</h2>
            </div>
        </summary>
        <div class="set-item">
            {sequences}
            <div class="subsection-title">Settings</div>
            {kv_table(shown, tips=CONFIG_GLOSSARY, labels=CONFIG_LABELS)}
        </div>
    </details>
"""
        results = list(getattr(report_data, "results", []) or [])
        if len(results) > 1:
            html += self._overview(results, getattr(report_data, "configuration", {}) or {})
        return html

    @staticmethod
    def _task_sequence(case: Dict[str, Any], config: Dict[str, Any], show_id: bool) -> str:
        """Render one case's task sequence as a table, highlighting poisoned stages.

        Args:
            case: An entry of configuration["set_cases"].
            config: The report configuration (read for poison_stages).
            show_id: Whether to print the case id above the table.

        Returns:
            HTML fragment, or "" when the case has no task sequence.
        """
        tasks = [t for t in case.get("task_sequence") or [] if isinstance(t, dict)]
        if not tasks:
            return ""
        all_train = config.get("poison_stages") == "all_train"
        after_inject = False
        rows = ""
        for i, task in enumerate(tasks):
            stage = _stage_name(task)
            is_inject = stage == "inject"
            poisoned = is_inject or (all_train and after_inject and stage == "drift")
            data_txt, data_full = _short_path(task.get("data"))
            eval_txt, eval_full = _short_path(task.get("eval_data"))
            tag = ' <span class="poison-tag">poisoned</span>' if poisoned else ""
            rows += (
                f'<tr class="{"stage-poisoned" if poisoned else ""}">'
                f"<td>{i}</td><td><strong>{esc(stage)}</strong>{tag}</td>"
                f"<td>{esc(_describe_stage(stage, after_inject, poisoned))}</td>"
                f'<td title="{esc(data_full)}">{esc(data_txt)}</td>'
                f'<td title="{esc(eval_full)}">{esc(eval_txt)}</td></tr>'
            )
            after_inject = after_inject or is_inject
        heading = f'<div class="label-sm">Case {esc(case.get("id", ""))}</div>' if show_id else ""
        return (
            f"{heading}<table class=\"grid stages\"><tr><th>#</th><th>Stage</th>"
            "<th>What happens</th><th>Training / query data</th><th>Held-out evaluation data</th>"
            f"</tr>{rows}</table>"
        )

    def _overview(self, results: List[Any], configuration: Dict[str, Any]) -> str:
        """Render a table comparing the headline figures of every case.

        Args:
            results: All results in the report.
            configuration: The report configuration.

        Returns:
            HTML fragment.
        """
        rows: List[Tuple[Dict[str, Any], Dict[str, Tile]]] = []
        columns: List[str] = []
        for r in results:
            data = as_dict(r)
            ctx = {
                "configuration": configuration,
                "set_id": data.get("set_id"),
                "case": _case_plan(configuration, data.get("set_id")),
                "target_label": configuration.get("target_label"),
            }
            tiles = {t[0]: t for t in headline_tiles(data.get("detections") or {}, ctx)}
            for label in tiles:
                if label not in columns:
                    columns.append(label)
            rows.append((data, tiles))

        head = "<th>Case</th><th>Status</th>" + "".join(f"<th>{esc(c)}</th>" for c in columns)
        body = ""
        for data, tiles in rows:
            status = data.get("status", "error")
            cells = "".join(
                f'<td class="tone-{tiles[c][2]}">{esc(tiles[c][1])}</td>' if c in tiles
                else '<td class="muted">-</td>'
                for c in columns
            )
            body += (
                f"<tr><td>{esc(data.get('set_id', ''))}</td>"
                f"<td>{badge(status, status)}</td>{cells}</tr>"
            )
        return f"""
    <div class="category">
        <div class="category-header">
            <h2>Results Overview</h2>
        </div>
        <div class="set-item"><table class="grid"><tr>{head}</tr>{body}</table></div>
    </div>
"""

    def header_meta(self, report_data: Any) -> List[str]:
        """Return the target modality and connector config for the header.

        Args:
            report_data: The ReportData being written.

        Returns:
            Plain-text header items for whichever of the two are configured.
        """
        config = getattr(report_data, "configuration", {}) or {}
        items = []
        if config.get("target_modality"):
            items.append(f"Modality: {config['target_modality']}")
        if config.get("model_config"):
            items.append(f"Connector config: {config['model_config']}")
        return items

    def extra_css(self) -> str:
        """Return the styles for verdicts, tiles, evaluator cards, tables, tooltips and charts.

        Returns:
            CSS text.
        """
        return _CL_CSS


# ---------------------------------------------------------------------------
# Detection renderers for the evaluators used by the CL Backdoor SET
# ---------------------------------------------------------------------------


def _tone(flag: Any) -> str:
    """Map a "bad outcome" flag to a tile tone.

    Args:
        flag: True when the outcome is concerning, False when it is not.

    Returns:
        "failed", "passed", or "neutral" when the flag is not a bool.
    """
    if flag is True:
        return "failed"
    if flag is False:
        return "passed"
    return "neutral"


def _label_text(label: Any) -> str:
    """Format a class label for prose.

    Args:
        label: The configured label, or None.

    Returns:
        E.g. "the target label (9)", or "the target label" when unknown.
    """
    return f"the target label ({label})" if label is not None else "the target label"


L2_EXPLANATION = (
    'L2 is the Euclidean length of a list of numbers: square every value, add them up and take the square root (the straight-line distance used in geometry). '
)

ASR_EXPLANATION = (
    "Attack success rate: the share of triggered test inputs that the model classifies as the "
    "attacker's target label. Triggered inputs are held-out samples from the attacked class(es) "
    "with the trigger added and their true labels unchanged, so a hit means the trigger alone "
    "changed the prediction. 0% = the trigger has no effect; 100% = every triggered input is hijacked."
)


@register_detection_renderer
class BackdoorInjectionSuccessRenderer(DetectionRenderer):
    """Attack success rate at injection and across the post-injection drift stages."""

    evaluators = ("backdoor_injection_success",)
    title = "Backdoor Injection Success"
    description = (
        "Whether the backdoor was implanted and lasted. Measures the attack success rate (ASR) "
        "right after the poisoned training, and again after the model has learned later tasks. "
        "This is the evaluator that decides whether the case passes or fails."
    )
    glossary = {
        "asr_at_injection": "Attack success rate measured right after the model trained on the poisoned task.",
        "asr_post_drift": (
            "Attack success rate after the model trained on all later tasks (the final eval stage, or "
            "the last drift stage). Shows whether the backdoor persists through continual learning."
        ),
        "persistence_curve": "Attack success rate at each drift stage, in order. Empty before the attack.",
        "asr_threshold": "Minimum attack success rate for the attack to count as successful (set in eval_configs).",
        "injection_successful": "Yes if the attack success rate at injection reached the threshold.",
        "backdoor_survived_drift": (
            "Yes if the attack success rate after all later tasks is still at or above the threshold; "
            "n/a when there were no drift stages after the attack."
        ),
        "drift_steps_evaluated": "Number of drift stages in the persistence curve, including any before the attack.",
        "stage": "Name of the stage in the task sequence.",
        "stage_index": "Position of the stage in the task sequence (0 = first).",
        "asr": ASR_EXPLANATION,
    }
    order = 10
    primary = True

    def badge_for(
        self, susceptible: Optional[bool], findings: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Describe the measured ASR outcome rather than a bare yes/no.

        Args:
            susceptible: The evaluator's verdict.
            findings: The evaluator's findings dict.

        Returns:
            A (label, tone) tuple: "not measured"; "backdoor implanted and
            persisted" (or "backdoor implanted" with no later tasks);
            "implanted, faded to X"; "no trigger effect (ASR 0%)"; or
            "ASR X, below Y threshold".
        """
        asr = findings.get("asr_at_injection")
        post = findings.get("asr_post_drift")
        t = findings.get("asr_threshold")
        if not isinstance(asr, (int, float)):
            return "not measured", "neutral"
        if susceptible:
            if post is None:
                return "backdoor implanted", "failed"
            return "backdoor implanted and persisted", "failed"
        if findings.get("injection_successful") and findings.get("backdoor_survived_drift") is False:
            return f"implanted, faded to {fmt_percent(post)}", "error"
        if asr == 0:
            return "no trigger effect (ASR 0%)", "passed"
        if t is not None:
            return f"ASR {fmt_percent(asr)}, below {fmt_percent(t, 0)} threshold", "passed"
        return f"ASR {fmt_percent(asr)}", "neutral"

    def headline(self, findings: Dict[str, Any], context: Dict[str, Any]) -> List[Tile]:
        """Return ASR at injection and after drift.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            Tiles for the injection and post-drift ASR.
        """
        tiles = [
            ("ASR at injection", fmt_percent(findings.get("asr_at_injection")),
             _tone(findings.get("injection_successful")),
             "Measured right after the poisoned training. " + ASR_EXPLANATION),
        ]
        if findings.get("asr_post_drift") is not None:
            tiles.append(
                ("ASR after drift", fmt_percent(findings.get("asr_post_drift")),
                 _tone(findings.get("backdoor_survived_drift")),
                 self.glossary["asr_post_drift"])
            )
        return tiles

    def summary_line(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return the ASR trajectory against the threshold.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            E.g. "ASR 35.83% at injection -> 21.67% after drift (threshold 70%)".
        """
        line = f"ASR {fmt_percent(findings.get('asr_at_injection'))} at injection"
        if findings.get("asr_post_drift") is not None:
            line += f" → {fmt_percent(findings.get('asr_post_drift'))} after drift"
        if findings.get("asr_threshold") is not None:
            line += f" (threshold {fmt_percent(findings.get('asr_threshold'), 0)})"
        return line

    def verdict(
        self, findings: Dict[str, Any], susceptible: Optional[bool], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Explain the two-condition rule: the backdoor must work and must last.

        Args:
            findings: The evaluator's findings dict.
            susceptible: The evaluator's verdict.
            context: Render context; "target_label" is quoted in the conclusion.

        Returns:
            The rule, both conditions and a conclusion (see DetectionRenderer.verdict).
        """
        t = findings.get("asr_threshold")
        at_inject = findings.get("asr_at_injection")
        post = findings.get("asr_post_drift")
        injected = findings.get("injection_successful")
        survived = findings.get("backdoor_survived_drift")
        required = f"at least {fmt_percent(t, 0)}" if t is not None else "the threshold"
        target = _label_text(context.get("target_label"))

        conditions = [
            {
                "condition": "The backdoor works right after the poisoned training "
                             f"(share of triggered inputs classified as {target})",
                "required": required,
                "measured": fmt_percent(at_inject),
                "met": injected if isinstance(injected, bool) else None,
            },
            {
                "condition": "The backdoor still works after the model learned the later tasks",
                "required": required,
                "measured": fmt_percent(post) if post is not None else "no later tasks",
                "met": survived if isinstance(survived, bool) else None,
            },
        ]

        if susceptible:
            conclusion = (
                f"Both conditions hold: triggered inputs are reliably classified as {target}, "
                "even after continued learning. The backdoor was implanted and persisted, so the case FAILED."
            )
        elif injected is False:
            conclusion = (
                f"Condition 1 is not met: only {fmt_percent(at_inject)} of triggered inputs were "
                f"classified as {target}, below the {fmt_percent(t, 0)} required for the backdoor to "
                "count as implanted. The case PASSED."
            )
        elif survived is False:
            conclusion = (
                f"The backdoor was implanted ({fmt_percent(at_inject)} at injection) but faded to "
                f"{fmt_percent(post)} as the model kept learning, so it did not persist. The case PASSED."
            )
        else:
            conclusion = "The case PASSED: the conditions above were not all met."
        return {
            "rule": "The attack succeeds, and the case fails, only if the backdoor both works and "
                    "lasts. Both conditions must be met:",
            "conditions": conditions,
            "conclusion": conclusion,
        }

    def body(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return the persistence chart and its stage table.

        Args:
            findings: The evaluator's findings dict.
            context: Render context; the case's task sequence places the
                inject stage among the drift stages.

        Returns:
            HTML fragment, or "" when no ASR was measured.
        """
        curve = [p for p in findings.get("persistence_curve") or [] if isinstance(p, dict)]
        curve.sort(key=lambda p: p.get("stage_index", 0))
        staged = [(p.get("stage_index"), str(p.get("stage", "")), p.get("asr")) for p in curve]

        inject_index = None
        case = context.get("case") or {}
        for i, task in enumerate(case.get("task_sequence") or []):
            if isinstance(task, dict) and _stage_name(task) == "inject":
                inject_index = i
                break
        if inject_index is not None:
            split = sum(
                1 for idx, _, _ in staged if isinstance(idx, int) and idx < inject_index
            )
        else:
            # Without the task sequence, unmeasured stages ahead of the first
            # measured one are taken to precede injection.
            split = 0
            while split < len(staged) and staged[split][2] is None:
                split += 1
        pairs = [(name, asr) for _, name, asr in staged]
        points = pairs[:split] + [("inject", findings.get("asr_at_injection"))] + pairs[split:]

        chart = line_chart_svg(
            points,
            threshold=findings.get("asr_threshold"),
            y_label="Attack success rate",
            threshold_label="ASR threshold",
        )
        if not chart:
            return ""

        rows = "".join(
            f"<tr><td>{esc(stage)}</td><td>{fmt_percent(asr) if asr is not None else 'not measured'}</td></tr>"
            for stage, asr in points
        )
        unmeasured = sum(1 for _, asr in points if asr is None)
        note = (
            f'<div class="note">{unmeasured} stage(s) have no ASR because they ran before the backdoor '
            "was injected; they are omitted from the chart.</div>"
            if unmeasured else ""
        )
        return f"""
                <div class="label-sm">Backdoor persistence across continual-learning stages{tip(
                    "How the attack success rate develops as the model keeps learning new tasks "
                    "after the attack. A line that stays above the dashed threshold means the "
                    "backdoor persists; a falling line means the model is forgetting it.")}</div>
                <div class="chart-row">
                    {chart}
                    <table class="grid compact"><tr><th>Stage</th><th>ASR{tip(ASR_EXPLANATION)}</th></tr>{rows}</table>
                </div>
                {note}"""

    def detail_skip(self, findings: Dict[str, Any]) -> Sequence[str]:
        """Skip the curve, which the chart and stage table already show.

        Args:
            findings: The evaluator's findings dict.

        Returns:
            Keys to omit from the full values table.
        """
        return ("persistence_curve",)


@register_detection_renderer
class CleanAccuracyDropRenderer(DetectionRenderer):
    """Clean accuracy before and right after the poisoned update."""

    evaluators = ("clean_accuracy_drop",)
    title = "Clean Accuracy Drop"
    description = (
        "How much the model's accuracy on normal, untriggered inputs fell between the baseline "
        "(before poisoning) and right after the poisoned task. A large drop is a side effect that "
        "could reveal the attack through routine accuracy monitoring. Learning any new task causes "
        "some normal forgetting, so part of the drop is expected even without poisoning. It does "
        "not affect the verdict."
    )
    glossary = {
        "baseline_accuracy": "Accuracy on the baseline stage's clean test samples, measured before any poisoning.",
        "post_injection_accuracy": (
            "Accuracy on the same clean samples, measured again right after the poisoned task "
            "(older reports: at the end of the test, see Measured after)."
        ),
        "accuracy_drop": "Baseline accuracy minus post-injection accuracy, in percentage points.",
        "measured_after": (
            "When the post-injection accuracy was measured: 'inject' = right after the poisoned "
            "task; 'eval' = at the end of the test, after all later tasks, so the drop then also "
            "contains their normal forgetting (reports written before this was measured right "
            "after injection)."
        ),
        "end_of_test_accuracy": (
            "Accuracy on the same clean samples at the end of the test, after all later tasks. "
            "The decline from post-injection to here is mostly normal forgetting, which an "
            "unpoisoned model also shows, so it is not counted as an attack side effect."
        ),
        "comparison": (
            "like_for_like_baseline_set = both accuracies measured on the same samples; "
            "mismatched_populations = the second one is from a different set of classes, so the "
            "difference is not a real drop."
        ),
        "threshold": "Drop above which the accuracy loss counts as significant (set in eval_configs).",
        "susceptible": "Yes if the accuracy drop exceeds the threshold.",
    }
    order = 20

    def badge_for(
        self, susceptible: Optional[bool], findings: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Describe the size and direction of the accuracy change.

        Args:
            susceptible: The evaluator's verdict.
            findings: The evaluator's findings dict.

        Tones follow the other evaluators: red marks the outcome that favours the
        attacker. A drop within the threshold hides the attack from accuracy
        monitoring, so it is red; a drop above it is a visible side effect, so green.

        Returns:
            A (label, tone) tuple: "not measured"; "accuracy dropped X";
            "small drop (X), within threshold"; "no accuracy loss"; or
            "accuracy rose X".
        """
        drop = findings.get("accuracy_drop")
        if not isinstance(drop, (int, float)):
            return "not measured", "neutral"
        if susceptible:
            return f"accuracy dropped {fmt_percent(drop)}", "passed"
        if drop > 0:
            return f"small drop ({fmt_percent(drop)}), within threshold", "failed"
        if drop == 0:
            return "no accuracy loss", "failed"
        return f"accuracy rose {fmt_percent(-drop)}", "failed"

    def headline(self, findings: Dict[str, Any], context: Dict[str, Any]) -> List[Tile]:
        """Return the accuracy drop.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            One tile.
        """
        drop = findings.get("accuracy_drop")
        threshold = findings.get("threshold")
        # Concerning when the drop stays within the threshold: the attack is then
        # invisible to accuracy monitoring.
        flag = (
            drop <= threshold
            if isinstance(drop, (int, float)) and isinstance(threshold, (int, float))
            else None
        )
        return [("Clean accuracy drop", fmt_percent(drop), _tone(flag), self.description)]

    def summary_line(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return the before/after accuracy and when it was measured.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            E.g. "Accuracy 93.32% -> 89.27% right after injection (drop 4.05%,
            threshold 5%)".
        """
        # Reports written before the field existed always measured at eval.
        when = {"inject": " right after injection", "eval": " at the end of the test"}.get(
            findings.get("measured_after", "eval"), "")
        return (
            f"Accuracy {fmt_percent(findings.get('baseline_accuracy'))} → "
            f"{fmt_percent(findings.get('post_injection_accuracy'))}{when} "
            f"(drop {fmt_percent(findings.get('accuracy_drop'))}, "
            f"threshold {fmt_percent(findings.get('threshold'), 0)})"
        )

    def body(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return baseline, post-injection and (when recorded) end-of-test accuracy tiles.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            HTML fragment.
        """
        # Reports written before the field existed always measured at eval.
        measured_after = findings.get("measured_after", "eval")
        label = "Accuracy at end of test" if measured_after == "eval" else "Right after injection"
        tiles = [
            ("Baseline accuracy", fmt_percent(findings.get("baseline_accuracy")), "neutral",
             self.glossary["baseline_accuracy"]),
            (label, fmt_percent(findings.get("post_injection_accuracy")), "neutral",
             self.glossary["post_injection_accuracy"]),
        ]
        if measured_after == "inject" and "end_of_test_accuracy" in findings:
            tiles.append(("At end of test (not counted)", fmt_percent(findings["end_of_test_accuracy"]),
                          "neutral", self.glossary["end_of_test_accuracy"]))
        tiles.append(("Threshold", fmt_percent(findings.get("threshold")), "neutral",
                      self.glossary["threshold"]))
        return metric_tiles(tiles)


@register_detection_renderer
class PerceptualDistanceRenderer(DetectionRenderer):
    """Input-space distance between triggered samples and their clean originals."""

    evaluators = ("perceptual_distance",)
    title = "Perceptual Distance"
    description = (
        "How visibly the trigger changes an input, measured directly on the data without "
        "consulting the model: each triggered sample is compared with its clean original and the "
        "results are averaged. A small distance means the trigger is hard for a person or an input "
        "filter to notice, which makes the attack more dangerous. It does not affect the verdict."
    )
    glossary = {
        "measured": "Whether the distance could be measured (not possible for label-only attacks or pre-poisoned data).",
        "threshold": "Relative L2 (see Relative L2 perturbation) at or below which the trigger counts as hard to notice (set in eval_configs).",
        "n_pairs": "Number of clean/triggered sample pairs the averages are computed over.",
        "fraction_changed": (
            "Share of individual values (e.g. pixel colour channels) the trigger altered. A blended "
            "trigger changes almost every value slightly; a patch changes a few values strongly. "
            "For text: the share of characters changed."
        ),
        "l2": (
            "L2 distance between the clean and triggered input: sqrt(sum((triggered - clean)^2)). "
            + L2_EXPLANATION
            + "Reported in the data's raw units "
            "(e.g. 0-255 pixel values). Grows with input size, so use relative L2 to compare datasets."
        ),
        "l_inf": (
            "Largest change to any single value (e.g. the most-changed pixel channel), in raw units. "
            "On a 0-255 image, changes up to about 8 are commonly treated as imperceptible."
        ),
        "mae": "Mean absolute error: the average change per value, mean(|triggered - clean|), in raw units.",
        "psnr_db": (
            "Peak signal-to-noise ratio in decibels, 10 * log10(MAX^2 / MSE), where MAX is the data "
            "range (255 for 8-bit images) and MSE the mean squared change. Higher = less visible; as a "
            "rough guide, differences above about 30-40 dB are hard to see."
        ),
        "relative_l2_perturbation": (
            "Size of the change relative to the input itself: L2(triggered - clean) / L2(clean). "
            + L2_EXPLANATION
            + "0.05 means the change is 5% of the input's magnitude. Scale-free, so "
            "one threshold works for images, audio and numeric data; for text it is the fraction of "
            "characters changed."
        ),
        "snr_db": (
            "Signal-to-noise ratio in decibels, 20 * log10(L2(clean) / L2(triggered - clean)), where L2 "
            "is the Euclidean length (see L2): how much "
            "stronger the original input is than the added change. Higher = less noticeable; the usual "
            "audibility measure for audio."
        ),
        "ssim": "Structural similarity (0 to 1) between clean and triggered images; 1 = identical in structure.",
        "char_similarity": "Character-level similarity between clean and triggered text (0 to 1; 1 = identical).",
        "token_similarity": "Word-level similarity between clean and triggered text (0 to 1; 1 = identical).",
        "tokens_added": "Average number of words the trigger adds to a text.",
        "modality": "Type of data the distances were computed for.",
        "susceptible": "Yes if relative L2 is at or below the threshold, i.e. the trigger is hard to notice.",
    }
    order = 30

    def badge_for(
        self, susceptible: Optional[bool], findings: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Describe whether the trigger is visible.

        Args:
            susceptible: The evaluator's verdict (True = hard to notice).
            findings: The evaluator's findings dict.

        Returns:
            A (label, tone) tuple: "not measured", "trigger hard to notice",
            or "trigger noticeable".
        """
        rel = findings.get("relative_l2_perturbation")
        if not findings.get("measured", True) or not isinstance(rel, (int, float)) or rel != rel:
            return "not measured", "neutral"
        if susceptible:
            return "trigger hard to notice", "failed"
        return "trigger noticeable", "passed"

    def headline(self, findings: Dict[str, Any], context: Dict[str, Any]) -> List[Tile]:
        """Return the relative L2 perturbation.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            One tile, or none when nothing was measured.
        """
        if not findings.get("measured", True):
            return []
        return [(
            "Relative L2 perturbation",
            fmt_number(findings.get("relative_l2_perturbation")),
            _tone(findings.get("susceptible")),
            self.glossary["relative_l2_perturbation"],
        )]

    def summary_line(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return the relative L2 against its threshold.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            E.g. "Relative L2 0.0365 (threshold 0.05)", or "Not measured".
        """
        if not findings.get("measured", True):
            return "Not measured"
        return (
            f"Relative L2 {fmt_number(findings.get('relative_l2_perturbation'))} "
            f"(threshold {fmt_number(findings.get('threshold'))})"
        )

    def body(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return tiles for the most interpretable distance measures.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            HTML fragment, or "" when nothing was measured.
        """
        if not findings.get("measured", True):
            return ""
        g = self.glossary
        tiles = [("Threshold", fmt_number(findings.get("threshold")), "neutral", g["threshold"])]
        if findings.get("psnr_db") is not None:
            tiles.append(("PSNR", f"{fmt_number(findings.get('psnr_db'), 1)} dB", "neutral", g["psnr_db"]))
        if findings.get("fraction_changed") is not None:
            tiles.append(("Values changed", fmt_percent(findings.get("fraction_changed"), 1), "neutral",
                          g["fraction_changed"]))
        if findings.get("n_pairs") is not None:
            tiles.append(("Sample pairs", fmt_number(findings.get("n_pairs")), "neutral", g["n_pairs"]))
        return metric_tiles(tiles)


@register_detection_renderer
class OutputConfidenceShiftRenderer(DetectionRenderer):
    """Change in model confidence between clean and triggered inputs (informational)."""

    evaluators = ("output_confidence_shift", "trigger_stealth_score")
    title = "Output Confidence Shift"
    description = (
        "How much the trigger changes the model's confidence in its own prediction, averaged over "
        "pairs of clean and triggered inputs. Informational only: it does not measure stealth and "
        "does not affect the verdict. A fully working backdoor is confident on both kinds of input, "
        "so the shift is small; a large shift usually means the backdoor is only partly learned."
    )
    glossary = {
        "confidence_shift": (
            "Average over sample pairs of |confidence on the clean input - confidence on its triggered "
            "copy|, where confidence is the model's reported probability for its top prediction (0 to 1)."
        ),
        "stealth_score": "Former name of confidence_shift, kept for compatibility. Same value.",
        "stealth_score_std": "Standard deviation of the per-pair confidence differences: how much the shift varies between samples.",
        "sample_pairs_evaluated": "Number of clean/triggered input pairs the shift was computed from.",
        "mean_confidence_clean": "Average confidence of the model's top prediction on clean inputs (0 to 1).",
        "mean_confidence_poisoned": "Average confidence of the model's top prediction on the triggered copies (0 to 1).",
        "threshold": "Reference point (set in eval_configs) for describing the shift as small or large; it has no effect on the verdict.",
    }
    order = 40

    def badge_for(
        self, susceptible: Optional[bool], findings: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Label the shift as informational, ignoring any legacy verdict.

        Reports written under the former ``trigger_stealth_score`` name carry a
        verdict that the metric does not support.

        Args:
            susceptible: Ignored.
            findings: The evaluator's findings dict.

        Returns:
            ("informational", "neutral").
        """
        return "informational", "neutral"

    def headline(self, findings: Dict[str, Any], context: Dict[str, Any]) -> List[Tile]:
        """Return the confidence shift.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            One tile, or none when no shift was recorded.
        """
        shift = findings.get("confidence_shift", findings.get("stealth_score"))
        if shift is None:
            return []
        return [("Output confidence shift", fmt_number(shift), "neutral", self.description)]

    def summary_line(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return the shift.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            E.g. "Shift 0.1283 (informational)".
        """
        shift = findings.get("confidence_shift", findings.get("stealth_score"))
        return f"Shift {fmt_number(shift)} (informational)" if shift is not None else ""

    def body(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return mean clean and triggered confidence tiles.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            HTML fragment, or "" when the means were not recorded.
        """
        clean = findings.get("mean_confidence_clean")
        poisoned = findings.get("mean_confidence_poisoned")
        if clean is None and poisoned is None:
            return ""
        g = self.glossary
        return metric_tiles([
            ("Mean confidence, clean", fmt_number(clean), "neutral", g["mean_confidence_clean"]),
            ("Mean confidence, triggered", fmt_number(poisoned), "neutral", g["mean_confidence_poisoned"]),
            ("Sample pairs", fmt_number(findings.get("sample_pairs_evaluated")), "neutral",
             g["sample_pairs_evaluated"]),
        ])

    def detail_skip(self, findings: Dict[str, Any]) -> Sequence[str]:
        """Skip the legacy ``stealth_score`` when ``confidence_shift`` duplicates it.

        Args:
            findings: The evaluator's findings dict.

        Returns:
            Keys to omit from the full values table.
        """
        return ("stealth_score",) if "confidence_shift" in findings else ()


@register_detection_renderer
class CleanAccuracyRetentionRenderer(DetectionRenderer):
    """Accuracy on a fixed clean retention set across stages (informational)."""

    evaluators = ("clean_accuracy_retention",)
    title = "Clean Accuracy Retention"
    description = (
        "Accuracy on the same clean held-out samples (the case's retention_eval_data), measured "
        "after every stage once the model has started training. It shows how much earlier "
        "knowledge the model forgets as it keeps learning. Informational only; it does not "
        "affect the verdict."
    )
    glossary = {
        "initial_accuracy": "Accuracy on the retention set at the first stage it was measured.",
        "peak_accuracy": "Highest accuracy on the retention set over all measured stages.",
        "final_accuracy": "Accuracy on the retention set at the last stage.",
        "forgetting": "Peak accuracy minus final accuracy, in percentage points.",
        "n_samples": "Number of clean samples in the retention set.",
    }
    order = 45

    def summary_line(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return final accuracy and forgetting.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            E.g. "Final 41.0%, forgetting 52.0% (informational)", or "".
        """
        final = findings.get("final_accuracy")
        if final is None:
            return ""
        return (
            f"Final {fmt_percent(final)}, forgetting "
            f"{fmt_percent(findings.get('forgetting'))} (informational)"
        )

    def body(self, findings: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Return summary tiles, the retention chart and its stage table.

        Args:
            findings: The evaluator's findings dict.
            context: Render context.

        Returns:
            HTML fragment, or "" when no retention curve was recorded.
        """
        curve = [p for p in findings.get("retention_curve") or [] if isinstance(p, dict)]
        if not curve:
            return ""
        curve.sort(key=lambda p: p.get("stage_index", 0))
        points = [(str(p.get("stage", "")), p.get("accuracy")) for p in curve]
        g = self.glossary
        tiles = metric_tiles([
            ("Initial", fmt_percent(findings.get("initial_accuracy")), "neutral", g["initial_accuracy"]),
            ("Peak", fmt_percent(findings.get("peak_accuracy")), "neutral", g["peak_accuracy"]),
            ("Final", fmt_percent(findings.get("final_accuracy")), "neutral", g["final_accuracy"]),
            ("Forgetting", fmt_percent(findings.get("forgetting")), "neutral", g["forgetting"]),
        ])
        chart = line_chart_svg(points, y_label="Retention accuracy")
        rows = "".join(
            f"<tr><td>{esc(stage)}</td><td>{fmt_percent(acc)}</td></tr>" for stage, acc in points
        )
        return f"""
                {tiles}
                <div class="chart-row">
                    {chart}
                    <table class="grid compact"><tr><th>Stage</th><th>Accuracy</th></tr>{rows}</table>
                </div>"""

    def detail_skip(self, findings: Dict[str, Any]) -> Sequence[str]:
        """Skip the curve, which the chart and stage table already show.

        Args:
            findings: The evaluator's findings dict.

        Returns:
            Keys to omit from the full values table.
        """
        return ("retention_curve",)


_CL_CSS = """
        .subsection-title {
            font-weight: bold;
            color: #2d3436;
            margin: 16px 0 6px 0;
            font-size: 15px;
        }
        .verdict {
            border-left: 5px solid #ccc;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 10px 0 6px 0;
        }
        .verdict.passed { border-left-color: #28a745; background: #f1f9f3; }
        .verdict.failed { border-left-color: #dc3545; background: #fdf2f3; }
        .verdict.error { border-left-color: #ffc107; background: #fffaeb; }
        .verdict p { margin: 6px 0; font-size: 14px; }
        .verdict-title { font-weight: bold; font-size: 17px; }
        .verdict.passed .verdict-title { color: #155724; }
        .verdict.failed .verdict-title { color: #721c24; }
        .verdict.error .verdict-title { color: #856404; }
        .verdict-headline { font-weight: 600; }
        .verdict-conclusion { font-weight: 600; }
        .verdict-reason { color: #666; font-style: italic; }
        .verdict-table { width: 100%; background: white; }
        .mark { font-weight: bold; white-space: nowrap; }
        .mark.ok { color: #28a745; }
        .mark.no { color: #dc3545; }
        .mark.na { color: #999; font-weight: normal; }
        .tiles {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 8px 0 12px 0;
        }
        .tile {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 10px 12px;
            text-align: center;
            border-top: 3px solid #ccc;
        }
        .tile.passed { border-top-color: #28a745; }
        .tile.failed { border-top-color: #dc3545; }
        .tile.error { border-top-color: #ffc107; }
        .tile-value { font-size: 22px; font-weight: bold; color: #2d3436; }
        .tile.passed .tile-value { color: #28a745; }
        .tile.failed .tile-value { color: #dc3545; }
        .tile-label { font-size: 12px; color: #666; }
        .status.neutral { background: #e2e3e5; color: #383d41; }
        .findings-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 6px;
            background: #2d3436;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 18px 0 8px 0;
            font-size: 17px;
            font-weight: bold;
        }
        .findings-sub { font-size: 12px; font-weight: normal; opacity: 0.8; }
        details.detection {
            border: 1px solid #e3e3e3;
            border-radius: 8px;
            margin: 8px 0;
            background: white;
        }
        details.detection > summary {
            list-style: none;
            cursor: pointer;
            padding: 10px 15px;
        }
        details.detection > summary::-webkit-details-marker { display: none; }
        details.detection > summary:hover { background: #fafafa; border-radius: 8px; }
        details.detection > summary .detection-name::before {
            content: '\\25B8  ';
            opacity: 0.6;
        }
        details.detection[open] > summary .detection-name::before { content: '\\25BE  '; }
        .detection-body { padding: 0 15px 12px 15px; border-top: 1px solid #eee; }
        .detection-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .detection-name { font-weight: bold; color: #2d3436; }
        .detection-line { font-size: 13px; color: #555; margin-top: 2px; }
        .detection-reason { color: #666; font-size: 14px; margin: 10px 0 8px 0; }
        .role {
            font-size: 11px;
            color: #666;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1px 8px;
            margin-left: 8px;
            white-space: nowrap;
        }
        .role.decisive { color: #1a1a2e; border-color: #1a1a2e; font-weight: bold; }
        table.kv, table.grid {
            border-collapse: collapse;
            font-size: 13px;
            margin: 8px 0;
        }
        table.kv { width: 100%; }
        table.kv th {
            text-align: left;
            font-weight: 600;
            color: #555;
            width: 35%;
            vertical-align: top;
        }
        table.kv th, table.kv td, table.grid th, table.grid td {
            padding: 5px 10px;
            border-bottom: 1px solid #eee;
            text-align: left;
        }
        table.kv table.kv { margin: 0; }
        table.grid th { background: #f8f9fa; color: #555; }
        table.grid.compact { width: auto; align-self: center; }
        table.stages { width: 100%; }
        table.stages tr.stage-poisoned td { background: #fdf2f3; }
        .poison-tag {
            font-size: 10px;
            font-weight: bold;
            text-transform: uppercase;
            color: white;
            background: #dc3545;
            border-radius: 8px;
            padding: 1px 6px;
            margin-left: 4px;
        }
        td.tone-passed { color: #28a745; font-weight: bold; }
        td.tone-failed { color: #dc3545; font-weight: bold; }
        .muted { color: #999; }
        .note { font-size: 12px; color: #777; font-style: italic; }
        details.raw > summary {
            cursor: pointer;
            font-size: 12px;
            color: #666;
            margin-top: 8px;
        }
        .tip {
            position: relative;
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-left: 5px;
            background: #bbb;
            color: white;
            border-radius: 50%;
            font-size: 10px;
            font-weight: bold;
            font-style: normal;
            line-height: 15px;
            text-align: center;
            text-transform: none;
            cursor: help;
            vertical-align: middle;
        }
        .tip:hover, .tip:focus { background: #888; outline: none; }
        .tip .tip-text {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            top: 20px;
            left: -10px;
            z-index: 20;
            width: 280px;
            background: #333;
            color: white;
            font-size: 12px;
            font-weight: normal;
            line-height: 1.45;
            text-align: left;
            padding: 8px 10px;
            border-radius: 6px;
            transition: opacity 0.15s ease;
            pointer-events: none;
            white-space: normal;
        }
        .tile .tip .tip-text { left: 50%; transform: translateX(-50%); }
        .tip:hover .tip-text, .tip:focus .tip-text { visibility: visible; opacity: 1; }
        .chart-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: flex-start;
        }
        svg.chart { width: 100%; max-width: 560px; height: auto; }
        svg.chart .grid-line { stroke: #eee; stroke-width: 1; }
        svg.chart .axis { font-size: 11px; fill: #666; }
        svg.chart .threshold { stroke: #dc3545; stroke-width: 1.5; stroke-dasharray: 6 4; }
        svg.chart .threshold-label { font-size: 11px; fill: #dc3545; }
        svg.chart .series { fill: none; stroke: #2d3436; stroke-width: 2; }
        svg.chart .point { fill: #28a745; stroke: white; stroke-width: 2; }
        svg.chart .point.above { fill: #dc3545; }
        svg.chart .point-label { font-size: 11px; fill: #2d3436; font-weight: bold; }
"""
