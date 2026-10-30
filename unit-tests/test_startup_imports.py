"""Unit tests that keep AVISE's startup fast by not importing heavy libraries at import time."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Libraries that take seconds to import; they must only be imported where they are used.
HEAVY_MODULES = [
    "torch",
    "transformers",
    "huggingface_hub",
    "mistral_common",
    "accelerate",
    "openai",
    "ollama",
    "scipy",
]

# Imports the CLI and engine, which registers every SET, connector and evaluator.
_PROBE = """
import json, sys
import avise.cli, avise.engine
from avise.registry import set_registry, connector_registry, evaluator_registry
print(json.dumps({
    "loaded": sorted(m for m in sys.modules if m.split(".")[0] in HEAVY),
    "registered": [len(set_registry.list()), len(connector_registry.list()), len(evaluator_registry.list())],
}))
"""


@pytest.fixture(scope="module")
def startup_state():
    """Import AVISE in a fresh interpreter and report what it loaded.

    Returns:
        Dict with "loaded" (heavy modules present in sys.modules) and
        "registered" (counts of registered SETs, connectors and evaluators).
    """
    import json

    code = f"HEAVY = {set(HEAVY_MODULES)!r}\n{_PROBE}"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_avise_registers_components(startup_state):
    # Guards the test below: lazy imports must not stop components from registering
    sets_, connectors, evaluators = startup_state["registered"]
    assert sets_ > 0 and connectors > 0 and evaluators > 0


@pytest.mark.parametrize("module", HEAVY_MODULES)
def test_importing_avise_does_not_load_heavy_module(startup_state, module):
    loaded = [m for m in startup_state["loaded"] if m.split(".")[0] == module]
    assert not loaded, (
        f"Importing AVISE loaded '{module}', which slows down every `avise` command "
        f"(e.g. `avise --SET-list`). Import it inside the function or method that "
        f"uses it instead of at module level."
    )
