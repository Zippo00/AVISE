"""Per-domain profiles for the AI summarizer.

Importing this package registers the built-in profiles: language model (also
the default) and continual learning.
"""

from .base import (
    DEFAULT_SYSTEM_PROMPT,
    SummaryProfile,
    register_summary_profile,
    select_summary_profile,
    set_default_profile,
)
from .languagemodel import LanguageModelSummaryProfile
from .continuallearning import (
    CLSetSummaryInfo,
    ContinualLearningSummaryProfile,
    RemediationOption,
    register_cl_set_summary,
    select_remediations,
    situation_tags,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SummaryProfile",
    "register_summary_profile",
    "select_summary_profile",
    "set_default_profile",
    "LanguageModelSummaryProfile",
    "CLSetSummaryInfo",
    "RemediationOption",
    "select_remediations",
    "situation_tags",
    "ContinualLearningSummaryProfile",
    "register_cl_set_summary",
]
