"""HTML result renderer for language model SETs (prompt / response results)."""

from typing import Any, Dict

from ..base import BaseReporter
from .base import ResultRenderer, set_default_renderer
from ....pipelines.languagemodel import EvaluationResult
from ....pipelines.languagemodel.schema import VERDICT_SOURCE_LABELS


class LanguageModelResultRenderer(ResultRenderer):
    """Render prompt/response (or full conversation) results of LM SETs.

    Registered as the default renderer, so it also draws any result no other
    renderer claims.
    """

    name = "languagemodel"

    def matches(self, result: Any) -> bool:
        """Report whether the result carries a prompt, as LM results do.

        Args:
            result: An EvaluationResult object or its dict form.

        Returns:
            True for LM EvaluationResults and dicts with a "prompt" key.
        """
        if isinstance(result, EvaluationResult):
            return True
        return isinstance(result, dict) and "prompt" in result

    @staticmethod
    def to_set_dict(result: Any) -> Dict[str, Any]:
        """Flatten a result into the dict shape ``render_item`` reads.

        Args:
            result: An LM EvaluationResult, another object with ``to_dict()``,
                or a dict.

        Returns:
            Dict with set_id, prompt, response, status, reason, attack_type,
            detections, full_conversation, description and, when present,
            elm_evaluation and verdict_source.
        """
        if isinstance(result, EvaluationResult):
            set_ = {
                "set_id": result.set_id,
                "prompt": result.prompt,
                "response": result.response,
                "status": result.status,
                "reason": result.reason,
                "attack_type": result.metadata.get("attack_type", ""),
                "detections": result.detections,
                "full_conversation": result.metadata.get("full_conversation", []),
                "description": result.metadata.get("description", ""),
            }
            if result.elm_evaluation:
                set_["elm_evaluation"] = result.elm_evaluation
            if result.verdict_source:
                set_["verdict_source"] = result.verdict_source
            return set_
        return result.to_dict() if hasattr(result, "to_dict") else result

    def render_item(self, result: Any, context: Dict[str, Any]) -> str:
        """Render one LM result as a collapsible block.

        Args:
            result: An LM EvaluationResult or its dict form.
            context: Report-wide information (unused by this renderer).

        Returns:
            HTML fragment.
        """
        set_ = self.to_set_dict(result)
        escape_html = BaseReporter.escape_html

        set_label = set_.get("attack_type") or set_.get("description") or ""
        if set_label:
            set_label = f" - {set_label}"

        elm_html = ""
        if "elm_evaluation" in set_:
            elm_html = f"""
                <div class="elm-eval">
                    <div class="label-sm">ELM Evaluation</div>
                    {escape_html(set_["elm_evaluation"])}
                </div>"""

        verdict_source_html = ""
        if set_.get("verdict_source"):
            source = set_["verdict_source"]
            verdict_source_html = f"""
            <div class="label-sm">Verdict decided by: {escape_html(VERDICT_SOURCE_LABELS.get(source, source))}</div>"""

        # Check for conversation format (memory test)
        conversation_html = ""
        if set_.get("full_conversation"):
            conversation_html = """
                <div class="label-sm">Conversation</div>
                <div class="conversation">"""
            for msg in set_["full_conversation"]:
                role = msg.get("role", "user")
                content = escape_html(msg.get("content", ""))
                conversation_html += f"""
                    <div class="message {role}"><strong>{role}:</strong> {content}</div>"""
            conversation_html += """
                </div>"""

        # Use prompt/response for non-conversation SETs
        prompt_response_html = ""
        if not conversation_html:
            prompt_response_html = f"""
            <div class="label-sm">Prompt</div>
            <div class="prompt">{escape_html(set_.get("prompt", ""))}</div>
            <div class="label-sm">Response</div>
            <div class="response">{escape_html(set_.get("response", ""))}</div>"""

        return f"""
        <details class="set-item">
            <summary>
                <div class="set-header">
                    <span class="set-id">{set_["set_id"]}{set_label}</span>
                    <span class="status {set_["status"]}">{set_["status"]}</span>
                </div>
            </summary>
            {prompt_response_html}
            {conversation_html}
            {verdict_source_html}
            <div class="reason">{escape_html(set_.get("reason", ""))}</div>
            {elm_html}
        </details>
"""


set_default_renderer(LanguageModelResultRenderer())
