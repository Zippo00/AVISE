"""Plugin based registry system for component discovery and use.

Components are registered via decorators, enabling automatic
discovery and flexible use.
"""

import importlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any

# Label of the group holding classes registered outside the grouped root package.
OTHER_GROUP = "Other"


@dataclass
class RegistryGroup:
    """A category of registered classes, as shown by ``--SET-list`` and ``--connector-list``.

    Attributes:
        key: Package name the group comes from (e.g. "single_turn"), or the
            label given in a class's ``registry_category``.
        display_name: Human-readable group name.
        groups: Subgroups, keyed by their ``key``.
        items: Registered names directly in this group.
    """

    key: str
    display_name: str
    groups: Dict[str, "RegistryGroup"] = field(default_factory=dict)
    items: List[str] = field(default_factory=list)

    def count(self) -> int:
        """Count the registered names in this group and all its subgroups.

        Returns:
            Total number of registered names.
        """
        return len(self.items) + sum(g.count() for g in self.groups.values())

    def sorted_groups(self) -> List["RegistryGroup"]:
        """Return subgroups ordered by display name, with "Other" last.

        Returns:
            List of subgroups.
        """
        return sorted(
            self.groups.values(),
            key=lambda g: (g.key == OTHER_GROUP, g.display_name.casefold()),
        )

    def iter_groups(self):
        """Yield every subgroup below this group, depth first.

        Yields:
            RegistryGroup objects.
        """
        for group in self.sorted_groups():
            yield group
            yield from group.iter_groups()

    def filter(self, category: str) -> Optional["RegistryGroup"]:
        """Keep only the groups whose key or display name matches a category.

        Matching ignores case, spaces, hyphens and underscores, so
        "languagemodel", "Language Model" and "language_model" are equal. A
        matching group keeps all its contents; its parent groups are kept so
        the result still shows where it sits.

        Args:
            category: Category to keep.

        Returns:
            A copy of this group pruned to the matches, or None if nothing matched.
        """
        wanted = _normalize(category)
        if wanted in (_normalize(self.key), _normalize(self.display_name)):
            return self
        kept = {}
        for key, group in self.groups.items():
            match = group.filter(category)
            if match is not None:
                kept[key] = match
        if not kept:
            return None
        return RegistryGroup(self.key, self.display_name, groups=kept)


def _normalize(text: str) -> str:
    """Lower-case text and drop everything but letters and digits.

    Args:
        text: Text to normalize.

    Returns:
        The normalized text.
    """
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def _child_group(
    parent: RegistryGroup, key: str, display_name: str, from_package: bool
) -> RegistryGroup:
    """Find or create the subgroup of ``parent`` matching a key or display name.

    An existing subgroup matches if its key or display name equals ``key`` or
    ``display_name`` after normalization, so a ``registry_category`` label
    like "Language Model" joins the group of the ``languagemodel`` package.

    Args:
        parent: Group to look in.
        key: Package name or category label of the wanted subgroup.
        display_name: Display name used if the subgroup is created.
        from_package: Whether ``key`` is a package name; package names and
            display names replace those of a matching group created from a label.

    Returns:
        The matching or newly created subgroup.
    """
    wanted = {_normalize(key), _normalize(display_name)}
    for group in parent.groups.values():
        if wanted & {_normalize(group.key), _normalize(group.display_name)}:
            if from_package and group.key != key:
                del parent.groups[group.key]
                group.key, group.display_name = key, display_name
                parent.groups[key] = group
            return group
    group = RegistryGroup(key=key, display_name=display_name)
    parent.groups[key] = group
    return group


def _package_display_name(package: str) -> str:
    """Get a package's ``DISPLAY_NAME``, or derive one from its name.

    Args:
        package: Dotted package path, e.g. "avise.sets.languagemodel".

    Returns:
        The package's ``DISPLAY_NAME`` if it defines one, else its last name
        part with underscores as spaces and the first letter capitalized.
    """
    try:
        display_name = getattr(importlib.import_module(package), "DISPLAY_NAME", None)
    except ImportError:
        display_name = None
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    name = package.rsplit(".", 1)[-1].replace("_", " ").strip()
    return name[:1].upper() + name[1:]


class Registry:
    """Holds information about classes."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register a class.

        Example usage:
            @SET_registry.register("prompt_injection")
        """

        def decorator(cls):
            if name in self._items:
                raise ValueError(f"{self.name}: '{name}' is already registered")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """Get a registered class by name."""
        if name not in self._items:
            raise KeyError(f"{self.name}: '{name}' not found. Available: {self.list()}")
        return self._items[name]

    def create(self, name: str, *args, **kwargs) -> Any:
        """Create an instance of a registered class."""
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list(self) -> List[str]:
        """List all registered names."""
        return list(self._items.keys())

    def grouped(self, root_package: str, root_name: str = "") -> RegistryGroup:
        """Group registered names into categories.

        A class's category path is taken from its ``registry_category``
        attribute (a list of group labels) if set, otherwise from the packages
        between ``root_package`` and its module, e.g.
        ``avise.sets.languagemodel.single_turn.jailbreak`` gives
        languagemodel > single_turn. Group names come from each package's
        ``DISPLAY_NAME``. Classes defined outside ``root_package`` are grouped
        under "Other".

        Args:
            root_package: Package the categories are relative to, e.g. "avise.sets".
            root_name: Display name of the returned root group.

        Returns:
            The root group; its items are names registered directly in
            ``root_package``.
        """
        root = RegistryGroup(key=root_package, display_name=root_name or self.name)
        for name, cls in self._items.items():
            node = root
            override = getattr(cls, "registry_category", None)
            if override:
                labels = [override] if isinstance(override, str) else list(override)
                for label in labels:
                    node = _child_group(node, label, label, from_package=False)
            elif cls.__module__.startswith(root_package + "."):
                parts = cls.__module__[len(root_package) + 1 :].split(".")[:-1]
                for i, part in enumerate(parts):
                    package = ".".join([root_package, *parts[: i + 1]])
                    node = _child_group(
                        node, part, _package_display_name(package), from_package=True
                    )
            else:
                node = _child_group(node, OTHER_GROUP, OTHER_GROUP, from_package=False)
            node.items.append(name)
        return root


# Make registries global
evaluator_registry = Registry("evaluators")
connector_registry = Registry("connectors")
set_registry = Registry("sets")
