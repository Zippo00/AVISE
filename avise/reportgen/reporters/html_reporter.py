"""HTML report writer."""

import re
from pathlib import Path
from typing import Dict, Any

from .base import BaseReporter
from .html_renderers import ResultRenderer, select_result_renderer
from ...pipelines.languagemodel import ReportData


class HTMLReporter(BaseReporter):
    """Writes reports in styled HTML format.

    Page chrome (styles, header, summary cards, AI summary, grouping) is shared
    by every SET. Each result is drawn by the ResultRenderer registered for its
    domain; see ``avise.reportgen.reporters.html_renderers``.
    """

    format_name = "html"
    file_extension = ".html"
    group_results: bool = True  # Can be overridden in SET scripts

    # Status colors for styling
    STATUS_COLORS = {"passed": "#28a745", "failed": "#dc3545", "error": "#ffc107"}

    def write(self, report_data: ReportData, output_path: Path) -> None:
        """Write report data as styled HTML file.

        Args:
            report_data: The report data to write
            output_path: Path to the output file / directory
        """
        html = self._generate_html(report_data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _report_renderer(self, report_data: ReportData) -> ResultRenderer:
        """Choose the renderer for report-level sections from the first result.

        Args:
            report_data: The report being written.

        Returns:
            The ResultRenderer for the report's domain (the default renderer
            when the report has no results).
        """
        results = getattr(report_data, "results", None) or []
        return select_result_renderer(results[0] if results else None)

    def _render_context(self, report_data: ReportData) -> Dict[str, Any]:
        """Build the context dict passed to every ``render_item`` call.

        Args:
            report_data: The report being written.

        Returns:
            Dict with "total_results" and "configuration".
        """
        return {
            "total_results": len(getattr(report_data, "results", None) or []),
            "configuration": getattr(report_data, "configuration", {}) or {},
        }

    def _generate_html(self, report_data: ReportData) -> str:
        """Generate complete HTML report."""
        self._context = self._render_context(report_data)
        html = self._get_html_header(report_data)
        html += self._get_summary_section(report_data)

        # Add AI Summary section
        if report_data.ai_summary:
            html += self._get_ai_summary(report_data.ai_summary)

        html += self._report_renderer(report_data).render_sections(report_data)

        # Use grouping if enabled in reporter or report_data
        use_grouping = getattr(self, "group_results", True) and getattr(
            report_data, "group_results", True
        )

        if use_grouping:
            html += self._get_results_grouped(report_data)
        else:
            html += self._get_results(report_data.results)
        html += "</body>\n</html>"
        return html

    def _get_results_grouped(self, report_data: ReportData) -> str:
        """Generate grouped results by set_category."""
        grouped = report_data.group_by_vulnerability()
        html = """
    <div class="category">
        <div class="category-header">
            <h2>Security Evaluation Test Results</h2>
        </div>
"""
        for group_name, results in sorted(grouped.items()):
            group_stats = self._calculate_group_stats(results)
            html += f"""
        <div class="set-item" style="background: #f0f0f0;">
            <div class="set-header">
                <span class="set-id">{group_name}</span>
                <span class="status passed">{group_stats["passed"]} passed</span>
                <span class="status failed">{group_stats["failed"]} failed</span>
                <span class="status error">{group_stats["error"]} error</span>
            </div>
        </div>
"""
            for result in results:
                html += self._render_result(result)
        html += "    </div>\n"
        return html

    def _calculate_group_stats(self, results: list) -> dict:
        """Calculate stats for a group of results."""
        statuses = [
            r.get("status") if isinstance(r, dict) else getattr(r, "status", None)
            for r in results
        ]
        passed = statuses.count("passed")
        failed = statuses.count("failed")
        error = statuses.count("error")
        return {"passed": passed, "failed": failed, "error": error}

    def _get_ai_summary(self, ai_summary: Dict[str, Any]) -> str:
        """Generate AI summary section for HTML report."""
        notes_html = "".join(
            f"<li>{self._markdown_to_html(note)}</li>"
            for note in ai_summary.get("notes", [])
        )
        return f"""
    <details class="category" open>
        <summary>
            <div class="category-header">
                <h2>AI Security Evaluation Summary</h2>
            </div>
        </summary>
        <div class="set-item">
            <h3>Issue Summary</h3>
            <div class="ai-content">{self._markdown_to_html(ai_summary.get("issue_summary", ""))}</div>
        </div>
        <div class="set-item">
            <h3>Recommended Remediations</h3>
            <div class="ai-content">{self._markdown_to_html(ai_summary.get("recommended_remediations", ""))}</div>
        </div>
        <div class="set-item">
            <h3>Notes</h3>
            <ul>
                {notes_html}
            </ul>
        </div>
    </details>
"""

    def _markdown_to_html(self, text: str) -> str:
        """Convert basic markdown to HTML."""
        if not text:
            return ""
        html = text
        html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        html = re.sub(r"^### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"^\* (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"\n\n", r"</p><p>", html)
        html = f"<p>{html}</p>"
        html = re.sub(r"<p></p>", r"", html)
        html = re.sub(r"<p>(<h[234]>)", r"\1", html)
        html = re.sub(r"(</h[234]>)<p>", r"\1", html)
        html = re.sub(r"(</h[234]>)</p>", r"\1", html)
        html = re.sub(r"<p>(<li>)", r"\1", html)
        html = re.sub(r"(</li>)<p>", r"\1", html)
        html = re.sub(r"(</li>)</p>", r"\1", html)
        return html

    def _get_html_header(self, report_data: ReportData) -> str:
        """Generate HTML head and opening body."""
        config = report_data.configuration
        target_info = ""
        if "target_model" in config:
            target_info = f"Target: {config['target_model']} |"
        elif "target_system" in config:
            target_info = f"Target: {config['target_system']} |"
        elif "target" in config:
            target_info = f"Target: {config['target']} |"

        renderer = self._report_renderer(report_data)
        for item in renderer.header_meta(report_data):
            target_info += f" {self.escape_html(str(item))} |"
        extra_css = renderer.extra_css()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AVISE Report - {report_data.set_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #1a1a2e;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            position: relative;
        }}
        .card .number {{ font-size: 36px; font-weight: bold; }}
        .card .label {{ color: #666; font-size: 14px; }}
        .card.passed .number {{ color: {self.STATUS_COLORS["passed"]}; }}
        .card.failed .number {{ color: {self.STATUS_COLORS["failed"]}; }}
        .card.error .number {{ color: {self.STATUS_COLORS["error"]}; }}
        .card .tooltip-trigger {{
            position: absolute;
            top: 8px;
            right: 10px;
            width: 16px;
            height: 16px;
            background: #ccc;
            color: white;
            border-radius: 50%;
            font-size: 11px;
            font-weight: bold;
            line-height: 16px;
            text-align: center;
            cursor: default;
        }}

        .card .tooltip-trigger .tooltip-text {{
            visibility: hidden;
            opacity: 0;
            background: #333;
            color: white;
            font-size: 12px;
            font-weight: normal;
            text-align: left;
            padding: 6px 10px;
            border-radius: 6px;
            width: 160px;
            position: absolute;
            top: 22px;   /* appears just below the ? */
            right: 0;
            z-index: 10;
            transition: opacity 0.2s ease;
            pointer-events: none;
            white-space: normal;
        }}

        .card .tooltip-trigger:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}
        .category {{
            background: white;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .category-header {{
            background: #2d3436;
            color: white;
            padding: 15px 20px;
            border-radius: 10px 10px 0 0;
        }}
        .category-header h2 {{ margin: 0; font-size: 18px; }}
        .set-item {{
            border-bottom: 1px solid #eee;
            padding: 15px 20px;
        }}
        .set-item:last-child {{ border-bottom: none; }}
        .set-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .set-id {{ font-weight: bold; color: #2d3436; }}
        .status {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status.passed {{ background: #d4edda; color: #155724; }}
        .status.failed {{ background: #f8d7da; color: #721c24; }}
        .status.error {{ background: #fff3cd; color: #856404; }}
        .prompt, .response {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .label-sm {{ font-size: 12px; color: #666; margin-bottom: 5px; }}
        .reason {{ color: #666; font-style: italic; margin-top: 10px; }}
        .elm-eval {{
            background: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 10px;
            margin-top: 10px;
        }}
        .conversation {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .conversation .message {{
            margin: 5px 0;
            padding: 5px 10px;
            border-radius: 5px;
        }}
        .conversation .user {{ background: #e3f2fd; }}
        .conversation .assistant {{ background: #e8f5e9; }}
        .conversation .system {{ background: #fff3e0; }}
        .ai-content {{ white-space: pre-wrap; }}
        .ai-content h3, .ai-content h4 {{ margin: 15px 0 10px 0; }}
        .ai-content li {{ margin: 5px 0; }}
        details.category > summary {{
            list-style: none;
            cursor: pointer;
            user-select: none;
        }}
        details.category > summary::-webkit-details-marker {{
            display: none;
        }}
        details.category > summary .category-header {{
            border-radius: 10px;
        }}
        details.category[open] > summary .category-header {{
            border-radius: 10px 10px 0 0;
        }}
        details.category > summary .category-header h2::after {{
            content: '       ▸';
            font-size: 20px;
            opacity: 0.7;
        }}
        details.category[open] > summary .category-header h2::after {{
            content: '       ▾';
        }}
        details.set-item > summary {{
            list-style: none;
            cursor: pointer;
            user-select: none;
        }}
        details.set-item > summary::-webkit-details-marker {{
            display: none;
        }}
        details.set-item > summary .set-header {{
            margin-bottom: 0;
        }}
        details.set-item[open] > summary .set-header {{
            margin-bottom: 10px;
        }}
        details.set-item > summary .set-header .set-id::after {{
            content: '       ▸';
            font-size: 20px;
            opacity: 0.7;
        }}
        details.set-item[open] > summary .set-header .set-id::after {{
            content: '       ▾';
        }}
{extra_css}
    </style>
</head>
<body>
    <div class="header">
        <h1>AVISE Security Report</h1>
        <div class="meta">
            Security Evaluation Test: {report_data.set_name} |
            {target_info}
            Generated: {report_data.timestamp} |
            Duration: {report_data.execution_time_seconds}s
            {" | ELM Evaluation: Yes" if config.get("elm_evaluation_used") else ""}
        </div>
    </div>
"""

    def _get_summary_section(self, report_data: ReportData) -> str:
        """Generate summary cards section."""
        summary = report_data.summary
        return f"""
    <div class="summary-cards">
        <div class="card">
            <div class="number">{summary["total_set_cases"]}</div>
            <div class="label">Total Security Evaluation Test Cases</div>
        </div>
        <div class="card passed">
            <div class="number">{summary["passed"]}</div>
            <div class="label">Passed ({summary["pass_rate"]}%)</div>
        </div>
        <div class="card failed">
            <div class="number">{summary["failed"]}</div>
            <div class="label">Failed ({summary["fail_rate"]}%)</div>
        </div>
        <div class="card error">
            <div class="number">{summary["error"]}</div>
            <div class="label">Inconclusive</div>
        </div>
        <div class="card">
            <span class="tooltip-trigger">?
            <span class="tooltip-text">95% confidence interval for the true pass rate, calculated using the Wilson score method. Accounts for sample size — wider when fewer tests are run.</span>
            </span>
            <div class="number" style="font-size: 28px">[{summary["ci_lower_bound"]:.2f}, {summary["ci_upper_bound"]:.2f}]</div>
            <div class="label">Pass Rate 95% Confidence Interval</div>
        </div>
    </div>{self._get_verdict_source_line(summary)}
"""

    @staticmethod
    def _get_verdict_source_line(summary: Dict[str, Any]) -> str:
        """Build a line stating how many case verdicts each source decided.

        Args:
            summary: The report's summary statistics.

        Returns:
            HTML fragment, or an empty string if the summary has no
            "verdict_sources" entry.
        """
        sources = summary.get("verdict_sources")
        if not sources:
            return ""
        return f"""
    <div class="label-sm" style="margin: -10px 0 20px 0;">
        Verdicts decided by Evaluators: {sources.get("evaluators", 0)} |
        decided by Evaluation Language Model: {sources.get("elm", 0)} |
        unresolved: {sources.get("unresolved", 0)}
    </div>"""

    def _get_results(self, results: list) -> str:
        """Generate list of results."""
        html = """
    <div class="category">
        <div class="category-header">
            <h2>Security Evaluation Test Results</h2>
        </div>
"""
        for result in results:
            html += self._render_result(result)
        html += "    </div>\n"
        return html

    def _render_result(self, result: Any) -> str:
        """Render one result with the renderer registered for its domain.

        Args:
            result: An EvaluationResult object or its dict form.

        Returns:
            HTML fragment.
        """
        context = getattr(self, "_context", None) or {}
        return select_result_renderer(result).render_item(result, context)

    def _get_set_item(self, set_: Dict[str, Any]) -> str:
        """Render a single result dict (kept for callers of the former API).

        Args:
            set_: A result in dict form.

        Returns:
            HTML fragment.
        """
        return self._render_result(set_)
