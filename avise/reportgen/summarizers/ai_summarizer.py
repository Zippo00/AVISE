"""AI summarizer for security evaluation test results."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .profiles import SummaryProfile, select_summary_profile

logger = logging.getLogger(__name__)


@dataclass
class AISummary:
    """Structured AI summary for security evaluation results."""

    issue_summary: str
    recommended_remediations: str
    notes: List[str]


class AISummarizer:
    """Generates AI-based summaries of security evaluation test results."""

    def __init__(
        self,
        evaluation_model_name: str = "mistralai/Ministral-3-3B-Instruct-2512",
        max_new_tokens: int = 512,
        reuse_model=None,
    ):
        """Initialize the AI summarizer with evaluation model.

        Args:
            evaluation_model_name: Name of the HuggingFace model to use
            max_new_tokens: Maximum tokens to generate
            reuse_model: Optional existing model instance to reuse
        """
        from avise.models.evaluation_lm import EvaluationLanguageModel

        if reuse_model is not None:
            self.model = reuse_model
        else:
            logger.info("Loading AI summarizer model...")
            self.model = EvaluationLanguageModel(
                model_name=evaluation_model_name,
                max_new_tokens=max_new_tokens,
            )

    def del_model(self):
        """Delete the model from GPU memory."""
        if self.model:
            self.model.del_model()

    def generate_summary(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        subcategory_runs: Optional[Dict[str, int]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        set_name: str = "",
        profile: Optional[SummaryProfile] = None,
    ) -> AISummary:
        """Generate an AI summary of the security evaluation test results.

        The domain-specific input, prompts and notes come from a SummaryProfile
        (see ``avise.reportgen.summarizers.profiles``), selected from the
        results unless one is given.

        Args:
            results: Evaluation results in dict form (as serialised to JSON).
            summary_stats: Summary statistics (total_set_cases, passed, failed, etc.).
            subcategory_runs: Optional mapping of subcategory -> number of runs.
            configuration: Optional report configuration; CL profiles read the
                attack settings and task sequences from it.
            set_name: The SET's name, used to look up SET-specific prompt context.
            profile: Profile to use instead of selecting one from the results.

        Returns:
            AISummary: Structured summary with issue summary, remediations, and notes.
        """
        profile = profile or select_summary_profile(results)
        configuration = configuration or {}
        facts = profile.facts(results, summary_stats, configuration, set_name=set_name)

        logger.info("Starting to generate AI summary of the evaluation results...")
        issue_summary = profile.canned_issue(results) or profile.postprocess(self._ask(
            profile.issue_prompt(facts, set_name=set_name),
            profile.system_prompt,
            fallback="Unable to generate summary.",
        ))
        logger.info("Starting to generate recommended remediations...")
        recommended_remediations = profile.canned_remediation(results) or profile.postprocess(
            self._ask(
                profile.remediation_prompt(facts, set_name=set_name),
                profile.system_prompt,
                fallback="Unable to generate recommendations.",
            )
        )
        notes = profile.notes(results, summary_stats, subcategory_runs or {}, configuration)

        return AISummary(
            issue_summary=issue_summary,
            recommended_remediations=recommended_remediations,
            notes=notes,
        )

    def _ask(self, prompt: str, system_prompt: str, fallback: str) -> str:
        """Query the model with a summary-specific system prompt.

        The model's own system prompt (e.g. the evaluation-judge prompt of a
        reused evaluation model) is swapped out for the request and restored
        afterwards.

        Args:
            prompt: The user prompt.
            system_prompt: System prompt for this request.
            fallback: Text returned when the model gives no usable answer.

        Returns:
            The model's answer with surrounding whitespace removed, or the
            fallback (also on errors, which are logged).
        """
        original = getattr(self.model, "system_prompt", None)
        try:
            self.model.system_prompt = {"role": "system", "content": system_prompt}
            response = self.model.generate(prompt)
        except Exception as e:
            logger.error(f"Failed to generate AI summary text: {e}")
            return fallback
        finally:
            self.model.system_prompt = original

        if not response:
            return fallback
        last_item = response[-1]
        if isinstance(last_item, dict):
            text = last_item.get("content", "")
        elif isinstance(last_item, (list, tuple)):
            text = str(last_item[0]) if last_item else ""
        else:
            text = str(last_item)
        return text.strip() or fallback


def format_json_ai_summary(ai_summary: AISummary) -> Dict[str, Any]:
    """Format AI summary for JSON report output.

    Args:
        ai_summary: The AISummary object to format

    Returns:
        Dict ready to be appended to JSON report
    """
    return {
        "ai_summary": {
            "issue_summary": ai_summary.issue_summary,
            "recommended_remediations": ai_summary.recommended_remediations,
            "notes": ai_summary.notes,
        }
    }


def format_html_ai_summary(ai_summary: AISummary) -> str:
    """Format AI summary for HTML report output.

    Args:
        ai_summary: The AISummary object to format

    Returns:
        HTML string for the AI summary section
    """
    notes_html = "".join(f"<li>{note}</li>" for note in ai_summary.notes)

    return f"""
    <div class="category">
        <div class="category-header">
            <h2>AI Security Evaluation Summary</h2>
        </div>
        <div class="set-item">
            <h3>Issue Summary</h3>
            <p>{ai_summary.issue_summary}</p>
        </div>
        <div class="set-item">
            <h3>Recommended Remediations</h3>
            <p>{ai_summary.recommended_remediations}</p>
        </div>
        <div class="set-item">
            <h3>Notes</h3>
            <ul>
                {notes_html}
            </ul>
        </div>
    </div>
"""


def format_markdown_ai_summary(ai_summary: AISummary) -> str:
    """Format AI summary for Markdown report output.

    Args:
        ai_summary: The AISummary object to format

    Returns:
        Markdown string for the AI summary section
    """
    notes_md = "\n".join(f"- {note}" for note in ai_summary.notes)

    return f"""---

## AI Security Evaluation Summary

### Issue Summary

{ai_summary.issue_summary}

### Recommended Remediations

{ai_summary.recommended_remediations}

### Notes

{notes_md}
"""
