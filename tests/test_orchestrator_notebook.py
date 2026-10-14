import ast
import json
from pathlib import Path
import pytest


def test_orchestrator_notebook_json_validity():
    nb_path = Path("colab/SchemaSentinel_Orchestrator.ipynb")
    assert nb_path.exists(), "SchemaSentinel_Orchestrator.ipynb notebook file must exist."

    content = nb_path.read_text(encoding="utf-8")
    data = json.loads(content)

    assert data.get("nbformat") == 4
    assert data.get("nbformat_minor") == 5
    assert "cells" in data
    assert len(data["cells"]) >= 10


def test_orchestrator_notebook_structure_and_sections():
    nb_path = Path("colab/SchemaSentinel_Orchestrator.ipynb")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = data["cells"]

    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    assert len(markdown_cells) >= 7
    assert len(code_cells) >= 6

    all_markdown_text = "\n".join("".join(c.get("source", [])) for c in markdown_cells)

    assert "Diagnostics & Hardware" in all_markdown_text
    assert "Google Drive Storage" in all_markdown_text
    assert "Stage 1: Cloud Supervised Fine-Tuning" in all_markdown_text
    assert "Stage 2: Post-Training Alignment" in all_markdown_text
    assert "Stage 3: 16-Bit LoRA Merge & llama.cpp GGUF Quantization" in all_markdown_text
    assert "Grammar-Constrained Serving Verification" in all_markdown_text


def test_orchestrator_notebook_code_cells_syntax():
    nb_path = Path("colab/SchemaSentinel_Orchestrator.ipynb")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]

    for idx, cell in enumerate(code_cells):
        source_lines = cell.get("source", [])
        clean_python_lines = []
        for line in source_lines:
            stripped = line.strip()
            if stripped.startswith("!"):
                continue
            clean_python_lines.append(line)

        python_source = "".join(clean_python_lines).strip()
        if python_source:
            tree = ast.parse(python_source)
            assert tree is not None


def test_orchestrator_notebook_zero_comments_in_python():
    nb_path = Path("colab/SchemaSentinel_Orchestrator.ipynb")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]

    for cell in code_cells:
        for line in cell.get("source", []):
            stripped = line.strip()
            if stripped.startswith("!"):
                continue
            comment_char = chr(35)
            assert not stripped.startswith(comment_char), f"Prohibited comment found in code cell: {line}"
            assert comment_char not in stripped or "http" in stripped, f"Prohibited comment character in code cell: {line}"
