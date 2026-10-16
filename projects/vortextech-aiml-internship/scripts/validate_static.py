#!/usr/bin/env python3
"""Static validation for the repository's Python and Jupyter artifacts.

This check deliberately does not require the ML dependencies or a dataset. It
catches malformed Python/Notebook files and committed notebook output before a
user attempts a data-dependent run.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate_python(path: Path) -> list[str]:
    """Return syntax errors for a Python source file."""
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"{path}: {exc}"]
    return []


def validate_notebook(path: Path) -> list[str]:
    """Return structural, syntax, and stale-output errors for a notebook."""
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: invalid JSON: {exc}"]

    if notebook.get("nbformat", 0) < 4 or not isinstance(notebook.get("cells"), list):
        errors.append(f"{path}: expected nbformat >= 4 and a cells list")
        return errors

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            errors.append(f"{path}:cell-{index}: {exc}")
        if cell.get("outputs"):
            errors.append(f"{path}:cell-{index}: committed notebook outputs must be cleared")
        if cell.get("execution_count") is not None:
            errors.append(f"{path}:cell-{index}: execution_count must be null")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if ".git" not in path.parts and ".venv" not in path.parts:
            errors.extend(validate_python(path))
    for path in sorted(ROOT.rglob("*.ipynb")):
        if ".git" not in path.parts and ".venv" not in path.parts:
            errors.extend(validate_notebook(path))

    if errors:
        print("Static validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("Static validation passed: Python syntax, notebook JSON/code, and outputs checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
