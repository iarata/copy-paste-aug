#!/usr/bin/env python
"""
Generate Python dataclasses from Hydra YAML config files and register them
as structured config schemas via Hydra's ConfigStore.

Discovery rules (subdirectory takes priority over flat file):
  - configs/{group}/default.yml   ← preferred (matches Hydra group/variant layout)
  - configs/{group}.yml           ← flat fallback

The generated cpa/utils/configs.py contains:
  - One @dataclass per YAML mapping (nested mappings become sub-dataclasses)
  - A root Config dataclass whose group fields are annotated with MISSING
    so Hydra knows they must be satisfied by a config-group selection
  - A register_configs() function that stores every schema in the ConfigStore

Usage:
    python scripts/generate_configs.py
    make generate_configs
"""

from __future__ import annotations

from pathlib import Path
import re
import textwrap
from typing import Any

from loguru import logger
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
OUTPUT_FILE = REPO_ROOT / "cpa" / "utils" / "configs.py"

# The primary config that wires everything together
PRIMARY_CONFIG = "default.yaml"

# Keys that are Hydra metadata – never become dataclass fields
HYDRA_RESERVED: frozenset[str] = frozenset(
    {"defaults", "_self_", "_target_", "_recursive_", "_convert_", "_global_"}
)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def to_class_name(name: str) -> str:
    """Convert a snake_case / kebab-case identifier to a CamelCaseConfig name.

    Examples
    --------
    >>> to_class_name("dataset")
    'DatasetConfig'
    >>> to_class_name("copy_paste_aug")
    'CopyPasteAugConfig'
    """
    return "".join(word.capitalize() for word in re.split(r"[_\-]", name)) + "Config"


def schema_name(group: str) -> str:
    """Return the ConfigStore name used for a group's base schema.

    Follows the Hydra convention of prefixing with ``base_``.

    Examples
    --------
    >>> schema_name("dataset")
    'base_dataset'
    """
    return f"base_{group}"


# ---------------------------------------------------------------------------
# Type / value helpers
# ---------------------------------------------------------------------------


def infer_type(value: Any) -> str:
    """Return a Python type-annotation string for a scalar YAML value.

    Note: bool must be checked before int because ``bool`` is a subclass of
    ``int`` in Python.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "Any"
    return "Any"


def format_default(value: Any) -> str:
    """Render a scalar YAML value as a Python literal string."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return repr(value)  # 'True' / 'False'
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(value)


# ---------------------------------------------------------------------------
# Dataclass collector
# ---------------------------------------------------------------------------


def collect_dataclasses(
    name: str,
    data: dict,
    ordered: dict[str, str],
) -> str:
    """Recursively emit ``@dataclass`` definitions into *ordered*.

    Parameters
    ----------
    name:
        The logical name for this mapping (becomes the class name).
    data:
        The YAML mapping dict to turn into a dataclass.
    ordered:
        Insertion-ordered dict of ``class_name -> class_source``.
        Classes are appended in *post-order* so every nested class is
        defined before the parent that references it.

    Returns
    -------
    str
        The class name that was generated for *name*.
    """
    class_name = to_class_name(name)
    field_lines: list[str] = []

    for key, value in data.items():
        if key in HYDRA_RESERVED:
            continue

        if isinstance(value, dict):
            nested_cls = collect_dataclasses(key, value, ordered)
            field_lines.append(f"    {key}: {nested_cls} = field(default_factory={nested_cls})")
        else:
            field_lines.append(f"    {key}: {infer_type(value)} = {format_default(value)}")

    body = "\n".join(field_lines) if field_lines else "    pass"
    class_src = f"@dataclass\nclass {class_name}:\n{body}"

    # First definition wins – don't overwrite if already collected
    if class_name not in ordered:
        ordered[class_name] = class_src

    return class_name


# ---------------------------------------------------------------------------
# Config-group discovery
# ---------------------------------------------------------------------------


def discover_groups(configs_dir: Path, skip_file: str) -> dict[str, Path]:
    """Return an ordered mapping of ``group_name -> yaml_path``.

    Search strategy (subdirectory structure has priority):

    1. ``configs/{group}/default.yml``   – canonical Hydra group/variant layout
    2. ``configs/{group}/*.yml``         – any other variant in a subdir
    3. ``configs/{group}.yml``           – legacy flat file (fallback)
    """
    groups: dict[str, Path] = {}

    # ── subdirectories (higher priority) ────────────────────────────────────
    for subdir in sorted(configs_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        default_variant = subdir / "default.yaml"
        if default_variant.exists():
            groups[subdir.name] = default_variant
        else:
            variants = sorted(subdir.glob("*.yaml"))
            if variants:
                groups[subdir.name] = variants[0]

    # ── flat files (lower priority / fallback) ───────────────────────────────
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        if yaml_path.name in (skip_file, ".gitkeep"):
            continue
        group_name = yaml_path.stem
        groups.setdefault(group_name, yaml_path)

    return groups


# ---------------------------------------------------------------------------
# Defaults-list parser
# ---------------------------------------------------------------------------


def parse_defaults_list(defaults: list) -> list[str]:
    """Extract config-group names from a Hydra ``defaults`` list.

    Each entry is one of:
    - ``{group: variant}``  → yield *group*
    - ``"_self_"``          → skip
    - ``"some_schema"``     → skip (bare schema references, not groups)
    """
    groups: list[str] = []
    for item in defaults:
        if isinstance(item, dict):
            for key in item:
                if key != "_self_":
                    groups.append(key)
        # bare strings that are not "_self_" are schema references – skip them
    return groups


# ---------------------------------------------------------------------------
# register_configs() emitter
# ---------------------------------------------------------------------------


def render_register_function(group_names: list[str]) -> str:
    """Return the source for the ``register_configs()`` helper.

    The function registers:
    - The root ``Config`` schema as ``"base_config"`` (no group).
    - Each sub-config class in its own group as ``"base_{group}"``.

    Callers must invoke ``register_configs()`` **before** the
    ``@hydra.main`` decorator is evaluated so that the schemas are
    available to Hydra at composition time.
    """
    lines = [
        "def register_configs() -> None:",
        '    """Register structured config schemas with Hydra\'s ConfigStore.',
        "",
        "    Call this once at module level (before @hydra.main) so that",
        "    the schemas are available during config composition:",
        "",
        "        from cpa.utils.configs import register_configs",
        "        register_configs()",
        "",
        "        @hydra.main(config_path='configs', config_name='default')",
        "        def main(cfg: Config) -> None:",
        "            ...",
        '    """',
        "    cs = ConfigStore.instance()",
        '    cs.store(name="base_config", node=Config)',
    ]
    for group in group_names:
        cls = to_class_name(group)
        lines.append(f'    cs.store(group="{group}", name="{schema_name(group)}", node={cls})')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_configs(configs_dir: Path, output_file: Path) -> None:
    """Parse YAML configs and write dataclasses + register_configs() to *output_file*."""

    # Insertion-ordered dict: class_name -> source lines
    ordered: dict[str, str] = {}

    # ── 1. Build sub-config dataclasses from group YAML files ────────────────
    groups = discover_groups(configs_dir, skip_file=PRIMARY_CONFIG)

    for group_name, yaml_path in groups.items():
        with yaml_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            # Empty or non-mapping file → emit a stub
            cls = to_class_name(group_name)
            ordered.setdefault(cls, f"@dataclass\nclass {cls}:\n    pass")
            continue

        collect_dataclasses(group_name, data, ordered)

    # ── 2. Build the root Config from default.yml ────────────────────────────
    primary_path = configs_dir / PRIMARY_CONFIG
    config_group_names: list[str] = []  # groups that contribute via config-group selection
    config_field_lines: list[str] = []

    if primary_path.exists():
        with primary_path.open(encoding="utf-8") as fh:
            primary_data = yaml.safe_load(fh) or {}

        # Config-group fields → annotated with MISSING so Hydra requires them
        for group_name in parse_defaults_list(primary_data.get("defaults", [])):
            cls = to_class_name(group_name)
            # Ensure a stub exists for groups that have no YAML file
            ordered.setdefault(cls, f"@dataclass\nclass {cls}:\n    pass")
            config_group_names.append(group_name)
            config_field_lines.append(f"    {group_name}: {cls} = MISSING")

        # Scalar / nested fields declared directly in default.yml
        for key, value in primary_data.items():
            if key in HYDRA_RESERVED or key == "defaults":
                continue
            if isinstance(value, dict):
                nested_cls = collect_dataclasses(key, value, ordered)
                config_field_lines.append(f"    {key}: {nested_cls} = field(default_factory={nested_cls})")
            else:
                config_field_lines.append(f"    {key}: {infer_type(value)} = {format_default(value)}")

    config_body = "\n".join(config_field_lines) if config_field_lines else "    pass"
    ordered["Config"] = f"@dataclass\nclass Config:\n{config_body}"

    # ── 3. Render output file ────────────────────────────────────────────────
    header = textwrap.dedent(
        '''\
        """
        Data classes for configuration management.
        Load configurations via Hydra.

        Auto-generated by scripts/generate_configs.py – do not edit by hand.
        Re-generate with:  make generate_configs
        """

        from dataclasses import dataclass, field
        from typing import Any

        from hydra.core.config_store import ConfigStore
        from omegaconf import MISSING
        '''
    )

    class_blocks = "\n\n\n".join(ordered.values())
    register_fn = render_register_function(config_group_names)

    content = header + "\n\n" + class_blocks + "\n\n\n" + register_fn + "\n"
    output_file.write_text(content, encoding="utf-8")

    logger.success(f"Generated  {output_file.relative_to(REPO_ROOT)}")
    logger.info(f"Classes    {', '.join(ordered)}")
    logger.info(f"CS groups  {', '.join(config_group_names) or '(none)'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_configs(CONFIGS_DIR, OUTPUT_FILE)
