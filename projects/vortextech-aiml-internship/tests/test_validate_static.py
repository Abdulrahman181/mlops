from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_static.py"


def test_static_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_all_notebooks_are_output_free_and_valid_json() -> None:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat", 0) >= 4
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None
