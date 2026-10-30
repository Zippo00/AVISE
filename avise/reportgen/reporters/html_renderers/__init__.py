"""Per-domain result renderers used by the HTML reporter.

Importing this package registers the built-in renderers: the language model
renderer (also the default) and the continual learning renderer.
"""

from .base import (
    RENDERER_HINT_KEY,
    ResultRenderer,
    get_result_renderer,
    register_result_renderer,
    select_result_renderer,
    set_default_renderer,
)
from .languagemodel import LanguageModelResultRenderer
from .continuallearning import (
    ContinualLearningResultRenderer,
    DetectionRenderer,
    get_detection_renderer,
    register_detection_renderer,
)

__all__ = [
    "RENDERER_HINT_KEY",
    "ResultRenderer",
    "get_result_renderer",
    "register_result_renderer",
    "select_result_renderer",
    "set_default_renderer",
    "LanguageModelResultRenderer",
    "ContinualLearningResultRenderer",
    "DetectionRenderer",
    "get_detection_renderer",
    "register_detection_renderer",
]
