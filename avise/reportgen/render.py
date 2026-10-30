"""Regenerate an HTML report from a JSON report.

Usage::

    python -m avise.reportgen.render report.json [-o report.html]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .loader import load_report_data
from .reporters import HTMLReporter


def main(argv: Optional[List[str]] = None) -> int:
    """Render each given JSON report to HTML.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 if any report failed.
    """
    parser = argparse.ArgumentParser(description="Render AVISE JSON reports as HTML.")
    parser.add_argument("reports", nargs="+", help="JSON report file(s).")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (single report) or directory (several reports). "
        "Defaults to the JSON path with an .html extension.",
    )
    args = parser.parse_args(argv)

    status = 0
    for src in map(Path, args.reports):
        if args.output and len(args.reports) == 1:
            dst = Path(args.output)
        elif args.output:
            dst = Path(args.output) / src.with_suffix(".html").name
        else:
            dst = src.with_suffix(".html")
        try:
            report_data = load_report_data(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            HTMLReporter().write(report_data, dst)
            print(f"{src} -> {dst}")
        except (OSError, ValueError) as e:
            print(f"error: {src}: {e}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
