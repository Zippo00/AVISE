"""AI summary profile for continual learning (CL) SETs.

The facts given to the model are built from the same interpretation layer the
HTML report uses (verdict explanation, outcome labels, evaluator descriptions
in ``avise.reportgen.reporters.html_renderers.continuallearning``), so the
summary works from pre-interpreted statements instead of raw numbers and
cannot contradict the report on what a metric means. Evaluators without a
detection renderer contribute their scalar findings and reason string.

SET-specific background and remediation options are registered per SET name
with ``register_cl_set_summary``; an unregistered CL SET still gets a summary
from its evaluator findings alone.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...reporters.html_renderers.components import fmt_number, fmt_percent, humanize
from ...reporters.html_renderers.continuallearning import (
    CONFIG_VALUE_FORMATTERS,
    get_detection_renderer,
    split_detection,
    verdict_basis,
    _ordered_detections,
    _stage_name,
)
from .base import SummaryProfile, register_summary_profile


@dataclass
class RemediationOption:
    """A mitigation the summary may recommend, and when it applies.

    Attributes:
        text:       What is recommended, in plain words.
        applies_to: Situation tags (see ``situation_tags``) that make it relevant;
                    "always" applies to every completed run with a measured effect.
        because:    Why it fits, phrased to follow "because".
    """

    text: str
    applies_to: Tuple[str, ...]
    because: str


@dataclass
class CLSetSummaryInfo:
    """SET-specific context for the CL summary prompts.

    Attributes:
        description:         Plain-language explanation of what the SET tests.
        remediation_options: Mitigations in priority order; the applicable ones
                             are selected from the measured results.
    """

    description: str
    remediation_options: List[RemediationOption] = field(default_factory=list)


_SET_INFO: Dict[str, CLSetSummaryInfo] = {}

GENERIC_CL_DESCRIPTION = (
    "A security evaluation test against a continual-learning (CL) system: a machine-learning "
    "system that keeps learning from new batches of training data (tasks) after deployment."
)


def register_cl_set_summary(set_name: str, info: CLSetSummaryInfo) -> None:
    """Register the summary context for a CL SET.

    Args:
        set_name: The SET's ``name`` as written in its reports (e.g.
            "Continual Learning Backdoor").
        info: Description and remediation options for the SET.
    """
    _SET_INFO[set_name] = info


register_cl_set_summary(
    "Continual Learning Backdoor",
    CLSetSummaryInfo(
        description=(
            "Backdoor poisoning test against a continual-learning (CL) system, i.e. a model that keeps "
            "learning from new batches of training data (tasks) after deployment. The test slips "
            "poisoned samples into one training task: each carries a trigger (a pattern added to the "
            "input) and is labelled as the attacker's chosen target class. A successful backdoor "
            "leaves the model working normally on clean inputs, while any input carrying the trigger "
            "is classified as the target class. The test then lets the model learn further benign "
            "tasks to see whether the backdoor persists."
        ),
        remediation_options=[
            RemediationOption(
                "inspecting the model for hidden triggers after each update, for example with "
                "trigger reverse-engineering methods such as Neural Cleanse",
                ("implanted",),
                "a backdoor was implanted in the model",
            ),
            RemediationOption(
                "reviewing what the system retains between tasks (stored training examples or "
                "learned class statistics used to prevent forgetting)",
                ("persisted",),
                "the backdoor survived the later training tasks, and the mechanisms that preserve "
                "old knowledge can also preserve a backdoor",
            ),
            RemediationOption(
                "screening each incoming batch of training data for poisoned samples before "
                "training, for example with activation clustering or spectral-signature analysis",
                ("implanted", "partial"),
                "poisoned samples in the training data are the source of the measured attack effect",
            ),
            RemediationOption(
                "relying on automated screening rather than manual inspection of training data",
                ("hard_to_notice",),
                "the trigger was hard to see, so manual inspection would likely miss it",
            ),
            RemediationOption(
                "verifying the origin and integrity of every batch of training data and accepting "
                "updates only from trusted sources",
                ("implanted", "partial"),
                "poisoned data can only affect the model if it enters the training data",
            ),
            RemediationOption(
                "re-checking the model on a trusted held-out test set after every update and "
                "watching for predictions shifting toward a single class",
                ("always",),
                "such a shift is how backdoor activity shows up in routine monitoring",
            ),
        ],
    ),
)
# Reports written before the SET was renamed carry its old name.
_SET_INFO["Backdoor"] = _SET_INFO["Continual Learning Backdoor"]


def _scalar_digest(findings: Dict[str, Any], limit: int = 8) -> str:
    """Summarise scalar findings for evaluators without a detection renderer.

    Args:
        findings: The evaluator's findings dict.
        limit: Maximum number of values listed.

    Returns:
        "Key: value; ..." text.
    """
    items = [
        f"{humanize(k)}: {fmt_number(v)}"
        for k, v in findings.items()
        if k not in ("reason", "evaluator") and isinstance(v, (int, float, str, bool))
    ]
    return "; ".join(items[:limit])


def _config_lines(configuration: Dict[str, Any]) -> List[str]:
    """Describe the attack configuration in plain text.

    Args:
        configuration: The report configuration.

    Returns:
        Lines describing the configured settings that are present.
    """
    lines = []
    get = configuration.get
    if get("target_modality"):
        lines.append(f"Type of data: {get('target_modality')}")
    if get("trigger_type"):
        params = get("trigger_config") or {}
        detail = ", ".join(f"{humanize(k)} {fmt_number(v)}" for k, v in params.items())
        lines.append(
            f"Trigger type: {get('trigger_type')}" + (f" ({detail})" if detail else "")
        )
    if get("poison_rate") is not None:
        lines.append(
            f"Poisoned share: {fmt_percent(get('poison_rate'), 0)} of the training samples that "
            "belong to the attacked classes in each poisoned task (not of all training data)"
        )
    if "target_label" in configuration:
        lines.append(f"Attacker's target class: {get('target_label')}")
    if "source_label" in configuration:
        fmt = CONFIG_VALUE_FORMATTERS.get("source_label")
        value = fmt(get("source_label")) if fmt else get("source_label")
        lines.append(f"Attacked classes: {value}")
    if get("poison_stages"):
        lines.append(f"Poisoned stages: {get('poison_stages')}")
    return lines


def _sequence_text(
    case: Optional[Dict[str, Any]], configuration: Dict[str, Any]
) -> str:
    """Render a case's task sequence as an arrow-separated line.

    Args:
        case: The case's entry of configuration["set_cases"], or None.
        configuration: The report configuration (read for poison_stages).

    Returns:
        E.g. "drift -> baseline -> inject [poisoned] -> drift -> eval", or "".
    """
    tasks = [t for t in (case or {}).get("task_sequence") or [] if isinstance(t, dict)]
    if not tasks:
        return ""
    all_train = configuration.get("poison_stages") == "all_train"
    after, parts = False, []
    for t in tasks:
        stage = _stage_name(t)
        poisoned = stage == "inject" or (all_train and after and stage == "drift")
        parts.append(stage + (" [poisoned]" if poisoned else ""))
        after = after or stage == "inject"
    return " -> ".join(parts)


def _drifts_after_inject(case: Optional[Dict[str, Any]]) -> Optional[int]:
    """Count the drift stages that follow the inject stage.

    Args:
        case: The case's entry of configuration["set_cases"], or None.

    Returns:
        The count, or None when the task sequence is unknown.
    """
    tasks = [t for t in (case or {}).get("task_sequence") or [] if isinstance(t, dict)]
    if not tasks:
        return None
    stages = [_stage_name(t) for t in tasks]
    if "inject" not in stages:
        return None
    return sum(1 for s in stages[stages.index("inject") + 1 :] if s == "drift")


def _case(configuration: Dict[str, Any], set_id: Any) -> Optional[Dict[str, Any]]:
    """Find a case's entry in configuration["set_cases"].

    Args:
        configuration: The report configuration.
        set_id: The case id.

    Returns:
        The entry, or None.
    """
    for c in configuration.get("set_cases") or []:
        if isinstance(c, dict) and c.get("id") == set_id:
            return c
    return None


def _backdoor(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a result's backdoor_injection_success findings.

    Args:
        result: A result in dict form.

    Returns:
        The findings dict, or {} when absent.
    """
    entry = (result.get("detections") or {}).get("backdoor_injection_success")
    return split_detection(entry)[1] if entry is not None else {}


def _observations(result: Dict[str, Any], configuration: Dict[str, Any]) -> List[str]:
    """Derive plain-language interpretations of one case's backdoor results.

    The same rules feed the summary facts and the deterministic notes, so the
    model is told the interpretation instead of inferring it.

    Args:
        result: A result in dict form.
        configuration: The report configuration.

    Returns:
        Observation sentences (without the case id); empty when none apply.
    """
    out = []
    f = _backdoor(result)
    asr, t = f.get("asr_at_injection"), f.get("asr_threshold")
    status = result.get("status")
    if (
        status == "passed"
        and isinstance(asr, (int, float))
        and isinstance(t, (int, float))
    ):
        if asr == 0:
            out.append(
                "The trigger had no measurable effect: the attack success rate was 0%."
            )
        elif asr < t:
            out.append(
                f"The attack success rate stayed below the threshold ({fmt_percent(asr)} < "
                f"{fmt_percent(t, 0)}) but was not zero. A small value can partly come from ordinary "
                "misclassifications into the target class; a larger one indicates a partial backdoor "
                "that a stronger attack could complete."
            )
    if f.get("injection_successful") and f.get("backdoor_survived_drift") is False:
        n = _drifts_after_inject(_case(configuration, result.get("set_id")))
        later = f" over {n} later task(s)" if n is not None else ""
        out.append(
            f"The backdoor was implanted but faded below the threshold{later}. Whether it fades "
            "depends on how much the model learns afterwards; a target that receives fewer updates "
            "could keep it longer."
        )
    if status == "failed" and f.get("backdoor_survived_drift") is True:
        out.append(
            "The backdoor survived every later training task that followed the attack."
        )
    pd_entry = (result.get("detections") or {}).get("perceptual_distance")
    if pd_entry is not None:
        pd_verdict, pd_findings = split_detection(pd_entry)
        if pd_verdict is True:
            out.append(
                "The trigger was below the perceptual threshold, so manual inspection of training "
                "data would be unlikely to reveal it."
            )
        elif pd_verdict is False and pd_findings.get("measured", True):
            out.append(
                "The trigger was above the perceptual threshold, i.e. noticeable: it is not "
                "stealthy, and inspection of the training data could reveal it."
            )
    return out


def situation_tags(results: List[Dict[str, Any]]) -> set:
    """Classify the completed results into situation tags for remediation selection.

    Args:
        results: All results in dict form.

    Returns:
        A set drawn from: "failed", "implanted" (ASR reached the threshold at
        injection), "persisted" (it stayed there), "faded", "partial" (passed
        with a non-zero ASR), "no_effect" (every completed case had ASR 0),
        "hard_to_notice", "noticeable" and "always" (any measured effect).
    """
    tags = set()
    completed = [r for r in results if r.get("status") in ("passed", "failed")]
    effects = []
    for r in completed:
        f = _backdoor(r)
        asr = f.get("asr_at_injection")
        if r.get("status") == "failed":
            tags.add("failed")
        if f.get("injection_successful"):
            tags.add("implanted")
            if f.get("backdoor_survived_drift") is False:
                tags.add("faded")
            elif r.get("status") == "failed":
                tags.add("persisted")
        elif isinstance(asr, (int, float)) and asr > 0:
            tags.add("partial")
        effects.append(isinstance(asr, (int, float)) and asr > 0)
        pd_entry = (r.get("detections") or {}).get("perceptual_distance")
        if pd_entry is not None:
            verdict, findings = split_detection(pd_entry)
            if verdict is True:
                tags.add("hard_to_notice")
            elif verdict is False and findings.get("measured", True):
                tags.add("noticeable")
    if completed and not any(effects):
        tags.add("no_effect")
    if any(effects):
        tags.add("always")
    return tags


def select_remediations(
    info: Optional[CLSetSummaryInfo], tags: set, limit: int = 4
) -> List[RemediationOption]:
    """Pick the applicable mitigations in priority order.

    Args:
        info: The SET's registered summary context, or None.
        tags: Output of ``situation_tags``.
        limit: Maximum number of mitigations returned.

    Returns:
        The selected options.
    """
    if not info:
        return []
    chosen = [o for o in info.remediation_options if tags & set(o.applies_to)]
    return chosen[:limit]


def urgency_sentence(tags: set) -> str:
    """Return the deterministic statement of how urgent remediation is.

    Args:
        tags: Output of ``situation_tags``.

    Returns:
        One sentence.
    """
    if "failed" in tags:
        return (
            "The target was found vulnerable, so these measures should be prioritised, and the "
            "test should be repeated after they are in place."
        )
    if tags & {"partial", "faded"}:
        return (
            "The target passed, so no urgent fix is required, but the measured effect warrants "
            "monitoring and periodic re-testing."
        )
    return "Periodic re-testing is recommended, especially after changes to the training pipeline."


@register_summary_profile
class ContinualLearningSummaryProfile(SummaryProfile):
    """Summarise CL results from their interpreted evaluator findings."""

    name = "continuallearning"
    priority = 10

    def matches(self, result: Dict[str, Any]) -> bool:
        """Report whether the result is a serialised CL EvaluationResult.

        Args:
            result: A result in dict form.

        Returns:
            True for dicts with "baseline_metrics" but no "prompt".
        """
        return "baseline_metrics" in result and "prompt" not in result

    def facts(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        configuration: Dict[str, Any],
        set_name: str = "",
    ) -> str:
        """Describe the test, its configuration and each case's interpreted results.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            configuration: The report configuration.
            set_name: The SET's name, used to look up its registered description.

        Returns:
            Plain text for the prompts.
        """
        info = _SET_INFO.get(set_name)
        self._tags = situation_tags(results)
        self._brief = self._situation_brief(results, summary_stats, configuration)
        lines = [
            "ABOUT THE TEST",
            info.description if info else GENERIC_CL_DESCRIPTION,
            "",
        ]

        config_lines = _config_lines(configuration)
        if config_lines:
            lines += ["ATTACK CONFIGURATION"] + [f"- {l}" for l in config_lines] + [""]

        lines += [
            "OVERALL RESULT",
            f"- Test cases: {summary_stats.get('total_set_cases', len(results))}; "
            f"passed (target withstood the attack): {summary_stats.get('passed', 0)}; "
            f"failed (target vulnerable): {summary_stats.get('failed', 0)}; "
            f"inconclusive (could not be completed): {summary_stats.get('error', 0)}",
            "",
        ]

        definitions: Dict[str, str] = {}
        for r in results:
            set_id = r.get("set_id", "unknown")
            status = r.get("status", "error")
            case = _case(configuration, set_id)
            ctx = {
                "configuration": configuration,
                "set_id": set_id,
                "case": case,
                "target_label": configuration.get("target_label"),
            }
            lines.append(f"CASE {set_id}: {status.upper()}")
            seq = _sequence_text(case, configuration)
            if seq:
                lines.append(f"- Task sequence: {seq}")
            if status not in ("passed", "failed"):
                lines.append(
                    f"- The case could not be completed: {r.get('reason', '')}"
                )
                lines.append("")
                continue

            detections = r.get("detections") or {}
            basis = verdict_basis(r)
            for name in basis:
                if name not in detections:
                    continue
                renderer = get_detection_renderer(name)
                susceptible, findings = split_detection(detections[name])
                explained = renderer.verdict(findings, susceptible, ctx)
                if explained:
                    lines.append(f"- Verdict rule: {explained.get('rule', '')}")
                    for i, c in enumerate(explained.get("conditions") or [], 1):
                        met = {True: "met", False: "not met"}.get(
                            c.get("met"), "not applicable"
                        )
                        lines.append(
                            f"  Condition {i}: {c.get('condition')} - required {c.get('required')}, "
                            f"measured {c.get('measured')}: {met}"
                        )
                    lines.append(
                        f"- Verdict explanation: {explained.get('conclusion', '')}"
                    )
                else:
                    lines.append(f"- Verdict reason: {r.get('reason', '')}")
            if not basis:
                lines.append(f"- Verdict reason: {r.get('reason', '')}")

            lines.append("- Measurements:")
            for name, entry, renderer in _ordered_detections(detections):
                susceptible, findings = split_detection(entry)
                title = renderer.title or humanize(name)
                role = (
                    "decides the verdict"
                    if name in basis
                    else "supporting, does not affect the verdict"
                )
                summary = renderer.summary_line(findings, ctx) or _scalar_digest(
                    findings
                )
                outcome, _ = renderer.badge_for(susceptible, findings)
                lines.append(f"  * {title} ({role}): {summary}. Outcome: {outcome}.")
                if not renderer.summary_line(findings, ctx) and findings.get("reason"):
                    lines.append(f"    Evaluator note: {findings['reason']}")
                if renderer.description:
                    definitions[title] = renderer.description
            observations = _observations(r, configuration)
            if observations:
                lines.append("- Observations:")
                lines += [f"  * {o}" for o in observations]
            lines.append("")

        if definitions:
            lines.append("WHAT THE MEASUREMENTS MEAN")
            lines += [f"- {k}: {v}" for k, v in definitions.items()]
            lines.append(
                "- ASR (attack success rate): the share of triggered test inputs that the model "
                "wrongly classified as the attacker's target class instead of their true class. A high "
                "ASR is bad for the defender: it means the trigger takes control of the prediction."
            )
        return "\n".join(lines).strip()

    _tags: set = set()
    _brief: str = ""

    def _situation_brief(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        configuration: Dict[str, Any],
    ) -> str:
        """Summarise the outcome in a few interpreted lines for the remediation prompt.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            configuration: The report configuration.

        Returns:
            Plain-text lines.
        """
        lines = [
            f"- {summary_stats.get('passed', 0)} case(s) passed (target withstood the attack), "
            f"{summary_stats.get('failed', 0)} failed (target vulnerable)."
        ]
        for r in results:
            if r.get("status") not in ("passed", "failed"):
                continue
            renderer = get_detection_renderer("backdoor_injection_success")
            entry = (r.get("detections") or {}).get("backdoor_injection_success")
            if entry is not None:
                outcome, _ = renderer.badge_for(*split_detection(entry))
                lines.append(
                    f"- Case {r.get('set_id')}: {r['status']}; backdoor result: {outcome}."
                )
            lines += [f"  {o}" for o in _observations(r, configuration)]
        return "\n".join(lines)

    def issue_prompt(self, facts: str, set_name: str = "") -> str:
        """Build the issue summary request.

        Args:
            facts: Output of ``facts()``.
            set_name: Unused; the SET description is already in the facts.

        Returns:
            The user prompt.
        """
        return f"""Write the "Issue Summary" section of a security report for the test described below. The readers are security professionals who may not be machine-learning experts.

{facts}

Write ONE paragraph of 4 to 7 sentences that:
1. States the overall result: whether the system was found vulnerable to this attack, and why, following the verdict explanation.
2. Gives the decisive measurements with their numbers copied exactly from the data, and says in plain words what they mean.
3. States whether the effect persisted as the model kept learning, when the data covers it.
4. Includes the observations listed for each case, in your own words.
5. Mentions the supporting measurements briefly and makes clear they do not change the verdict.

Rules:
- Copy every number exactly as written in the data. Do not calculate, round or invent numbers.
- Do not describe a result below the threshold as a vulnerability, and do not describe a non-zero attack success rate as "no effect".
- Triggered inputs classified as the target class are misclassified; never call them "correctly" classified.
- Perceptual distance only measures how visible the trigger is. Never say the trigger was "detected"; nothing in this test tried to detect it.
- Call the trigger stealthy or hard to see only if the observations say so.
- Do not mention anything that is not in the data (no other attacks, datasets, models or tools).
- Write in an impersonal, factual tone. Do not address the reader.
- Plain text only: no headings, lists, bold text or markdown."""

    def remediation_prompt(self, facts: str, set_name: str = "") -> str:
        """Build the remediation request from the pre-selected mitigations.

        The mitigations and the urgency are chosen deterministically from the
        results (``situation_tags``, ``select_remediations``); the model only
        phrases them. ``facts`` is not used because the full facts led the
        model to contradict the verdict.

        Args:
            facts: Unused (kept for the SummaryProfile interface).
            set_name: The SET's name, used to look up its mitigations.

        Returns:
            The user prompt.
        """
        tags = self._tags
        chosen = select_remediations(_SET_INFO.get(set_name), tags)
        brief = self._brief
        measures = (
            "\n".join(
                f"{i}. Recommend {o.text}, because {o.because}."
                for i, o in enumerate(chosen, 1)
            )
            or "1. Recommend periodic re-testing of the system."
        )
        return f"""Write the "Recommended Remediations" section of a security report.

SITUATION (already decided from the measurements; do not reinterpret it)
{brief}

MEASURES TO RECOMMEND (already selected; use exactly these, in this order, and add no others)
{measures}

FINAL SENTENCE (use it as the last sentence, unchanged)
{urgency_sentence(tags)}

Write ONE paragraph: one sentence for each measure, saying what is recommended and the reason given, then the final sentence.

Rules:
- Use an impersonal, declarative tone ("... is recommended because ..."). Do not give orders and do not address the reader.
- Do not mention numbers, and do not add claims that are not in the situation or the reasons.
- None of these measures was used or tested during the evaluation; never say or imply that one was effective.
- Plain text only: no headings, lists, bold text or markdown."""

    def postprocess(self, text: str) -> str:
        """Force the plain single-paragraph format the prompts ask for.

        The summary model does not reliably keep to one paragraph or avoid
        markdown, so paragraphs are joined and markdown markers removed.

        Args:
            text: The model's answer.

        Returns:
            One paragraph of plain text.
        """
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", text, flags=re.MULTILINE)
        text = text.replace("**", "").replace("__", "")
        return re.sub(r"\s*\n\s*", " ", text).strip()

    def canned_issue(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return a fixed statement when no case completed.

        Args:
            results: All results in dict form.

        Returns:
            The fixed text, or None when there are completed cases to summarise.
        """
        if not any(r.get("status") in ("passed", "failed") for r in results):
            return "No test case could be completed, so no security assessment can be made from this run."
        return None

    def canned_remediation(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return fixed remediation text when no case completed or no effect was measured.

        Args:
            results: All results in dict form.

        Returns:
            The fixed text, or None to query the model.
        """
        if not any(r.get("status") in ("passed", "failed") for r in results):
            return (
                "No remediation can be recommended because no test case was completed."
            )
        if "no_effect" in situation_tags(results):
            return (
                "No attack effect was measured, so no specific remediation is required. Periodic "
                "re-testing is recommended, especially after changes to the training pipeline or "
                "to the data sources the system learns from."
            )
        return None

    def notes(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        subcategory_runs: Dict[str, int],
        configuration: Dict[str, Any],
    ) -> List[str]:
        """Return deterministic notes about reliability and partial effects.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            subcategory_runs: Unused by this profile.
            configuration: The report configuration.

        Returns:
            Note strings.
        """
        notes = [
            "This summary was written by a language model from the measured results. Where it "
            "differs from the measurements in the report, the measurements take precedence."
        ]
        completed = [r for r in results if r.get("status") in ("passed", "failed")]
        if completed:
            notes.append(
                f"Each test case trained the target system once ({len(completed)} completed "
                "case(s)). Training a continual-learning system can be non-deterministic, so attack "
                "success rates may vary between repeated runs; repeating the test is recommended "
                "before drawing firm conclusions."
            )
        for r in completed:
            for o in _observations(r, configuration):
                if o.startswith(("The backdoor survived", "The trigger was above")):
                    continue  # stated by the verdict / not a caveat
                notes.append(f"In case **{r.get('set_id', '')}**: {o}")
        errors = [r for r in results if r.get("status") not in ("passed", "failed")]
        if errors:
            notes.append(
                f"{len(errors)} test case(s) could not be completed and are not reflected in the "
                "summary."
            )
        notes.append(
            "Automated tests may produce false positives or negatives; human review is advised for "
            "critical evaluations."
        )
        return notes
