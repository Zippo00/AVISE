"""Unit tests for grouped registry listings (--SET-list and --connector-list)."""

import io
import sys
import types

import pytest

from avise import cli
from avise import connectors, sets  # noqa: F401  (imports register all SETs and connectors)
from avise.registry import Registry, connector_registry, set_registry
from avise.utils import (
    TerminalStyle,
    ansi_colors,
    detect_terminal_style,
    list_categories,
    render_registry_tree,
)


def _category_packages(registry, root_package):
    """List the packages that appear as categories in a registry's listing.

    Groups that come from a ``registry_category`` label or the "Other" group
    are not packages and are skipped.

    Args:
        registry: Registry to group.
        root_package: Package the categories are relative to, e.g. "avise.sets".

    Returns:
        Dotted package paths, depth first.
    """
    packages = []

    def walk(group, package):
        """Collect package-backed subgroups of a group, recursing into them.

        Args:
            group: Group whose subgroups are checked.
            package: Dotted package path of ``group``.
        """
        for sub in group.sorted_groups():
            sub_package = f"{package}.{sub.key}"
            if sub_package in sys.modules:
                packages.append(sub_package)
                walk(sub, sub_package)

    walk(registry.grouped(root_package), root_package)
    return packages


CATEGORY_PACKAGES = _category_packages(set_registry, "avise.sets") + _category_packages(
    connector_registry, "avise.connectors"
)


def _make_class(name, module, **attrs):
    """Create a class that appears to be defined in the given module.

    Args:
        name: Class name.
        module: Value for the class's ``__module__``.
        **attrs: Extra class attributes.

    Returns:
        The new class.
    """
    return type(name, (), {"__module__": module, **attrs})


@pytest.fixture
def registry(monkeypatch):
    """Build a registry with fake SETs across nested and foreign packages.

    The fake packages are inserted into sys.modules so DISPLAY_NAME lookups work.

    Returns:
        The populated Registry.
    """
    packages = {
        "fakeroot": None,
        "fakeroot.languagemodel": "Language Model",
        "fakeroot.languagemodel.single_turn": "Single-turn",
        "fakeroot.languagemodel.multi_turn": None,
        "fakeroot.new_domain": None,
    }
    for package, display_name in packages.items():
        module = types.ModuleType(package)
        if display_name:
            module.DISPLAY_NAME = display_name
        monkeypatch.setitem(__import__("sys").modules, package, module)

    reg = Registry("sets")
    classes = {
        "zeta_jailbreak": ("fakeroot.languagemodel.single_turn.jb", "Jailbreak SET"),
        "alpha_injection": ("fakeroot.languagemodel.single_turn.pi", "Prompt\n   injection"),
        "red_queen": ("fakeroot.languagemodel.multi_turn.rq", "Red Queen"),
        "new_set": ("fakeroot.new_domain.thing", "New domain SET"),
        "plugin_set": ("someones_plugin.sets", "Third-party SET"),
    }
    for name, (module, description) in classes.items():
        reg.register(name)(_make_class(name, module, description=description))
    reg.register("pinned")(
        _make_class(
            "Pinned",
            "fakeroot.languagemodel.single_turn.x",
            registry_category=["Language Model", "Special"],
        )
    )
    return reg


# ── Registry.grouped ──────────────────────────────────────────────────────


def test_grouped_follows_package_paths(registry):
    root = registry.grouped("fakeroot")

    lm = root.groups["languagemodel"]
    assert lm.display_name == "Language Model"
    assert sorted(lm.groups["single_turn"].items) == ["alpha_injection", "zeta_jailbreak"]
    assert lm.groups["multi_turn"].items == ["red_queen"]
    assert root.count() == 6


def test_grouped_falls_back_to_package_name(registry):
    root = registry.grouped("fakeroot")

    assert root.groups["new_domain"].display_name == "New domain"
    assert root.groups["languagemodel"].groups["multi_turn"].display_name == "Multi turn"


def test_grouped_puts_foreign_classes_under_other(registry):
    root = registry.grouped("fakeroot")

    assert root.groups["Other"].items == ["plugin_set"]
    assert root.sorted_groups()[-1].key == "Other"


def test_registry_category_overrides_package_path(registry):
    root = registry.grouped("fakeroot")

    # The "Language Model" label joins the languagemodel package's group
    assert "Language Model" not in root.groups
    assert root.groups["languagemodel"].groups["Special"].items == ["pinned"]
    assert "pinned" not in root.groups["languagemodel"].groups["single_turn"].items


def test_registry_category_registered_first_merges_with_package_group():
    import sys
    import types

    reg = Registry("sets")
    reg.register("pinned")(
        _make_class("Pinned", "elsewhere.mod", registry_category="Language Model")
    )
    reg.register("pi")(_make_class("PI", "fakeroot2.languagemodel.single_turn.pi"))
    module = types.ModuleType("fakeroot2.languagemodel")
    module.DISPLAY_NAME = "Language Model"
    sys.modules["fakeroot2.languagemodel"] = module
    try:
        root = reg.grouped("fakeroot2")
    finally:
        del sys.modules["fakeroot2.languagemodel"]

    assert list(root.groups) == ["languagemodel"]
    assert root.groups["languagemodel"].items == ["pinned"]
    assert root.groups["languagemodel"].groups["single_turn"].items == ["pi"]


# ── DISPLAY_NAME of real category packages ────────────────────────────────


def test_category_packages_are_found():
    # Guards the parametrized test below against silently checking nothing
    assert any(p.startswith("avise.sets.") for p in CATEGORY_PACKAGES)
    assert any(p.startswith("avise.connectors.") for p in CATEGORY_PACKAGES)


@pytest.mark.parametrize("package", CATEGORY_PACKAGES)
def test_category_package_defines_display_name(package):
    module = sys.modules[package]
    display_name = getattr(module, "DISPLAY_NAME", None)

    assert isinstance(display_name, str) and display_name.strip(), (
        f"{module.__file__} is listed as a category by `avise --SET-list` / "
        "`avise --connector-list` but defines no DISPLAY_NAME. Add a line like "
        '`DISPLAY_NAME = "Language Model"` to it.'
    )


def test_category_packages_skip_labels_and_other(registry):
    packages = _category_packages(registry, "fakeroot")

    assert packages == [
        "fakeroot.languagemodel",
        "fakeroot.languagemodel.multi_turn",
        "fakeroot.languagemodel.single_turn",
        "fakeroot.new_domain",
    ]


# ── RegistryGroup.filter ──────────────────────────────────────────────────


@pytest.mark.parametrize("category", ["single_turn", "Single-turn", "SINGLE TURN", "singleturn"])
def test_filter_matches_key_or_display_name_loosely(registry, category):
    filtered = registry.grouped("fakeroot").filter(category)

    assert list(filtered.groups) == ["languagemodel"]
    assert list(filtered.groups["languagemodel"].groups) == ["single_turn"]
    assert filtered.count() == 2


def test_filter_without_match_returns_none(registry):
    assert registry.grouped("fakeroot").filter("vision") is None


def test_list_categories(registry):
    categories = list_categories(registry.grouped("fakeroot"))

    assert "single_turn (Single-turn)" in categories
    assert "Other" in categories
    assert categories[-1] == "Other"


# ── render_registry_tree ──────────────────────────────────────────────────


def test_render_plain_tree(registry):
    root = registry.grouped("fakeroot").filter("languagemodel")
    descriptions = {name: getattr(registry.get(name), "description", "") for name in registry.list()}

    output = render_registry_tree(root, "Available SETs", descriptions, hint="tip")

    assert output == (
        "Available SETs (4)\n"
        "\n"
        "Language Model (4)\n"
        "├── Multi turn (1)\n"
        "│   └── red_queen        Red Queen\n"
        "├── Single-turn (2)\n"
        "│   ├── alpha_injection  Prompt injection\n"
        "│   └── zeta_jailbreak   Jailbreak SET\n"
        "└── Special (1)\n"
        "    └── pinned\n"
        "\n"
        "tip\n"
    )


def test_render_ascii_and_truncation(registry):
    root = registry.grouped("fakeroot").filter("single_turn")
    descriptions = {"alpha_injection": "x" * 100}
    style = TerminalStyle(unicode=False, width=50)

    output = render_registry_tree(root, "SETs", descriptions, style)

    line = next(l for l in output.splitlines() if "alpha_injection" in l)
    assert line.startswith("    |-- alpha_injection")
    assert len(line) == 50 and line.endswith("...")
    assert "`-- zeta_jailbreak" in output
    assert "├" not in output


def test_render_drops_descriptions_on_narrow_terminals(registry):
    root = registry.grouped("fakeroot").filter("single_turn")

    output = render_registry_tree(
        root, "SETs", {"alpha_injection": "desc"}, TerminalStyle(width=30)
    )

    assert "desc" not in output
    assert "alpha_injection" in output


def test_render_color(registry):
    root = registry.grouped("fakeroot").filter("multi_turn")

    output = render_registry_tree(root, "SETs", style=TerminalStyle(color=True))

    assert f"{ansi_colors['bold_cyan']}Language Model{ansi_colors['reset']}" in output
    assert f"{ansi_colors['bold']}red_queen{ansi_colors['reset']}" in output


# ── detect_terminal_style ─────────────────────────────────────────────────


class _FakeStream(io.StringIO):
    """StringIO with configurable TTY status and encoding."""

    def __init__(self, tty, encoding):
        """Store the fake TTY status and encoding.

        Args:
            tty: Value returned by isatty().
            encoding: Value of the ``encoding`` attribute.
        """
        super().__init__()
        self._tty = tty
        self._encoding = encoding

    @property
    def encoding(self):
        """Return the fake encoding."""
        return self._encoding

    def isatty(self):
        """Return the fake TTY status."""
        return self._tty


def test_detect_style_for_pipe():
    style = detect_terminal_style(_FakeStream(tty=False, encoding="utf-8"))

    assert style == TerminalStyle(color=False, unicode=True, width=None)


def test_detect_style_for_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    style = detect_terminal_style(_FakeStream(tty=True, encoding="cp437"))

    assert style.color is True
    assert style.unicode is False
    assert style.width is not None


def test_detect_style_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert detect_terminal_style(_FakeStream(tty=True, encoding="utf-8")).color is False


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_set_list_groups_real_sets(capsys):
    cli.main(["--SET-list"])
    output = capsys.readouterr().out

    assert "Language Model" in output
    assert "Single-turn" in output
    assert "Continual Learning" in output
    assert output.index("Single-turn") < output.index("lm_jailbreak")


def test_cli_set_list_category(capsys):
    cli.main(["--SET-list", "multi_turn"])
    output = capsys.readouterr().out

    assert "red_queen" in output
    assert "lm_jailbreak" not in output
    assert "cl_backdoor" not in output


def test_cli_set_list_unknown_category(capsys):
    cli.main(["--SET-list", "no-such-category"])
    output = capsys.readouterr().out

    assert "no category matches 'no-such-category'" in output
    assert "languagemodel (Language Model)" in output


def test_cli_connector_list_category(capsys):
    cli.main(["--connector-list", "continuallearning"])
    output = capsys.readouterr().out

    assert "generic-rest-cl" in output
    assert "ollama-lm" not in output
    assert "Available Report Formats" in output
