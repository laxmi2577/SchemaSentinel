from pathlib import Path
import pytest
import yaml


def test_ci_workflow_file_exists():
    workflow_path = Path(".github/workflows/ci_quality_gates.yml")
    assert workflow_path.exists(), "GitHub Actions CI workflow file must exist."


def test_ci_workflow_yaml_structure():
    workflow_path = Path(".github/workflows/ci_quality_gates.yml")
    content = workflow_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    assert "name" in data
    assert "on" in data or True in data
    assert "jobs" in data

    triggers = data.get("on") or data.get(True)
    assert "push" in triggers
    assert "pull_request" in triggers

    jobs = data["jobs"]
    assert "quality_gates" in jobs
    job = jobs["quality_gates"]
    assert job.get("runs-on") == "ubuntu-latest"
    assert "steps" in job


def test_ci_workflow_quality_gates_steps():
    workflow_path = Path(".github/workflows/ci_quality_gates.yml")
    content = workflow_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    steps = data["jobs"]["quality_gates"]["steps"]
    step_names = [s.get("name", "") for s in steps]

    assert any("Zero-Comment" in name for name in step_names)
    assert any("AST Compilation" in name for name in step_names)
    assert any("Data Engine" in name for name in step_names)
    assert any("Execution Sandbox" in name for name in step_names)
    assert any("LLMOps Telemetry" in name for name in step_names)
    assert any("Hardware Envelope" in name for name in step_names)


def test_ci_workflow_contains_zero_comment_enforcement():
    workflow_path = Path(".github/workflows/ci_quality_gates.yml")
    content = workflow_path.read_text(encoding="utf-8")

    assert "Zero-comment policy violation detected" in content
    assert "Gate 1 Passed" in content
    assert "Gate 6 Passed" in content
