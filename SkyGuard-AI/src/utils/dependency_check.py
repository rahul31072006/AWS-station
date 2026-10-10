"""Non-crashing dependency and runtime-environment checks."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    import_name: str
    installed: bool
    version: str | None
    install_command: str
    optional: bool = False


DEPENDENCIES = {
    "numpy": ("numpy", "python -m pip install numpy", False),
    "pandas": ("pandas", "python -m pip install pandas", False),
    "scikit-learn": ("sklearn", "python -m pip install scikit-learn", False),
    "scipy": ("scipy", "python -m pip install scipy", False),
    "plotly": ("plotly", "python -m pip install plotly", False),
    "shap": ("shap", "python -m pip install shap", True),
    "lightgbm": ("lightgbm", "python -m pip install lightgbm", True),
}


def check_dependencies() -> list[DependencyStatus]:
    """Return package availability without importing optional modules globally."""
    statuses = []
    for name, (import_name, command, optional) in DEPENDENCIES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "installed")
            statuses.append(DependencyStatus(name, import_name, True, str(version), command, optional))
        except ImportError:
            statuses.append(DependencyStatus(name, import_name, False, None, command, optional))
    return statuses


def environment_report() -> str:
    """Build a user-facing startup report with actionable missing-package commands."""
    lines = [
        "SkyGuard AI environment validation",
        f"Python executable: {sys.executable}",
        f"Python version: {platform.python_version()}",
        f"Virtual environment: {'yes' if sys.prefix != sys.base_prefix else 'no'}",
        "",
    ]
    for status in check_dependencies():
        marker = "OK" if status.installed else ("OPTIONAL" if status.optional else "MISSING")
        detail = status.version or f"install with: {status.install_command}"
        lines.append(f"[{marker}] {status.name}: {detail}")
    return "\n".join(lines)


def required_dependencies_missing() -> list[DependencyStatus]:
    return [status for status in check_dependencies() if not status.installed and not status.optional]