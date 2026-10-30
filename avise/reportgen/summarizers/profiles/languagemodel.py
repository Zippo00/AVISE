"""AI summary profile for language model SETs (prompts unchanged from the original summarizer)."""

from typing import Any, Dict, List, Optional

from .base import SummaryProfile, set_default_profile


class LanguageModelSummaryProfile(SummaryProfile):
    """Summarise prompt/response results; also the default profile."""

    name = "languagemodel"

    def matches(self, result: Dict[str, Any]) -> bool:
        """Report whether the result carries a prompt, as LM results do.

        Args:
            result: A result in dict form.

        Returns:
            True when the result has a "prompt" key.
        """
        return "prompt" in result

    def facts(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        configuration: Dict[str, Any],
        set_name: str = "",
    ) -> str:
        """List the run statistics and up to 20 failed cases with their reasons.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            configuration: Unused by this profile.
            set_name: Unused by this profile.

        Returns:
            Plain text for the prompts.
        """
        total = summary_stats.get("total_set_cases", 0)
        passed = summary_stats.get("passed", 0)
        failed = summary_stats.get("failed", 0)
        error = summary_stats.get("error", 0)

        lines = [
            f"Total tests: {total}",
            f"Passed: {passed} ({summary_stats.get('pass_rate', 0)}%)",
            f"Failed: {failed} ({summary_stats.get('fail_rate', 0)}%)",
            f"Error/Inconclusive: {error}",
            "",
            "Failed tests:",
        ]

        failed_results = [r for r in results if r.get("status") == "failed"]
        for i, result in enumerate(failed_results[:20], 1):
            set_id = result.get("set_id", "unknown")
            reason = result.get("reason", "No reason provided")
            metadata = result.get("metadata", {})
            attack_type = metadata.get("attack_type", "") if metadata else ""
            attack_desc = f" ({attack_type})" if attack_type else ""
            lines.append(f"  {i}. {set_id}{attack_desc}: {reason}")

        if len(failed_results) > 20:
            lines.append(f"  ... and {len(failed_results) - 20} more failed tests")

        return "\n".join(lines)

    def canned_issue(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return a fixed statement when no case failed.

        Args:
            results: All results in dict form.

        Returns:
            The fixed text, or None when there are failures to summarise.
        """
        if not any(r.get("status") == "failed" for r in results):
            return "No vulnerabilities were detected during this security evaluation test."
        return None

    def canned_remediation(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return a fixed statement when no case failed.

        Args:
            results: All results in dict form.

        Returns:
            The fixed text, or None when there are failures to address.
        """
        if not any(r.get("status") == "failed" for r in results):
            return "No remediation steps are required as no vulnerabilities were detected."
        return None

    def issue_prompt(self, facts: str, set_name: str = "") -> str:
        """Build the issue summary request.

        Args:
            facts: Output of ``facts()``.
            set_name: Unused by this profile.

        Returns:
            The user prompt.
        """
        return f"""You are a cybersecurity analyst producing a formal security assessment summary.

Analyze the following security evaluation test results and produce a professional issue summary.

{facts}

Requirements:
- Use formal, neutral, and technical language suitable for a security report.
- Maintain an impersonal tone. Do NOT address the reader directly.
- Do NOT use conversational phrases (e.g., "you should", "you need to").
- Do NOT include explanations about the analysis process.
- Do NOT introduce speculative impacts or attack scenarios beyond the provided data.
- Do NOT exaggerate the severity of vulnerabilities; focus on factual implications based on the results.

Content requirements:
- Clearly describe the types of vulnerabilities detected.
- Summarize their characteristics and observed behaviors.
- Indicate general security implications without exaggeration.

Output:
- A single concise paragraph.
- No headings, no bullet points, no prefixes.
"""

    def remediation_prompt(self, facts: str, set_name: str = "") -> str:
        """Build the remediation request.

        Args:
            facts: Output of ``facts()``.
            set_name: Unused by this profile.

        Returns:
            The user prompt.
        """
        return f"""You are a cybersecurity analyst producing formal remediation guidance for a security assessment report.

Based on the following security evaluation test results, provide remediation recommendations.

{facts}

Requirements:
- Use formal, neutral, and technical language suitable for a security report.
- Maintain an impersonal and declarative tone.
- Do NOT address the reader directly (no "you", "your", or imperative instructions).
- Do NOT include explanations framed as "what this means in simple terms".
- Do NOT include conversational phrasing or advisory tone.
- Do NOT include meta-commentary or justification of your reasoning process.
- Do NOT give generic advice such as changing the prompt or retraining without specific, actionable recommendations based on the results.
- Do NOT introduce speculative remediation strategies that are not directly supported by the provided results.

Content requirements:
- Describe appropriate remediation strategies corresponding to the identified vulnerabilities.
- Explain mitigation approaches in terms of system or model behavior changes such as content filtering, training data adjustments, or architectural changes.
- Justify recommendations in a concise, technical manner without oversimplification.

Output:
- A single concise paragraph.
- No headings, no bullet points, no prefixes.
"""

    def notes(
        self,
        results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        subcategory_runs: Dict[str, int],
        configuration: Dict[str, Any],
    ) -> List[str]:
        """Return the run-count warning and the human-review note.

        Args:
            results: All results in dict form.
            summary_stats: Pass/fail statistics of the run.
            subcategory_runs: Number of results per vulnerability subcategory.
            configuration: Unused by this profile.

        Returns:
            Note strings.
        """
        notes = []
        total_runs = summary_stats.get("total_set_cases", 0) if summary_stats else 0
        low_run_categories = {
            category: count
            for category, count in (subcategory_runs or {}).items()
            if count < 100
        }
        if low_run_categories:
            categories_str = ", ".join(
                f"**{cat}** ({count})" for cat, count in low_run_categories.items()
            )
            notes.append(
                f"Following SET categories had fewer than the suggested 100 runs and results may vary due to AI stochasticity: {categories_str}. It is recommended to conduct a larger number of runs for a more comprehensive assessment."
            )
        elif total_runs > 0 and total_runs < 100:
            notes.append(
                f"The total number of runs ({total_runs}) is fewer than the suggested 100 and results may vary due to AI stochasticity. It is recommended to conduct a larger number of runs for a more comprehensive assessment."
            )
        notes.append(
            "Automated tests may produce false positives or negatives; human review is advised for critical evaluations."
        )
        return notes


set_default_profile(LanguageModelSummaryProfile())
