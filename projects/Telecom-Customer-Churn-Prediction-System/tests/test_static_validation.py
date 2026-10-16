"""Dependency-free static checks for the source notebook and inference app."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = next(ROOT.glob("*.ipynb"))


def main() -> None:
    ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    ast.parse((ROOT / "churn_features.py").read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    notebook_code = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        notebook_code.append(source)
        assert not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()), (
            "Notebook contains a shell/IPython magic command"
        )
        ast.parse(source)
        assert not cell.get("outputs"), "Notebook must be committed without stale outputs"
        assert cell.get("execution_count") is None, "Notebook must have null execution counts"
    combined = "\n".join(notebook_code)
    assert "validate_input_frame" in combined
    assert "fillna(df['TotalCharges'].median())" not in combined
    assert "positive_class_probability" in combined
    print("static validation passed")


if __name__ == "__main__":
    main()
