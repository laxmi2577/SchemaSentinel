import json
from pathlib import Path
import pytest
from src.llmops.model_card import (
    SafetySLABenchmark,
    ModelCardConfig,
    GovernanceReport,
    get_default_safety_slas,
    ModelCardGenerator,
)


def test_model_card_config_defaults():
    config = ModelCardConfig()
    assert config.model_name == "SchemaSentinel-7B"
    assert config.model_version == "1.0.0"
    assert config.license == "Apache-2.0"
    assert "SQLite" in config.supported_dialects
    assert "PostgreSQL" in config.supported_dialects
    assert "DROP" in config.prohibited_commands
    assert "DELETE" in config.prohibited_commands


def test_get_default_safety_slas():
    slas = get_default_safety_slas()
    assert len(slas) == 5

    names = [s.metric_name for s in slas]
    assert "AST Read-Only Enforcement" in names
    assert "CFG Syntax Validity" in names
    assert "Adversarial Injection Refusal" in names
    assert "Tenant Isolation Compliance" in names
    assert "Local CPU RAM Envelope" in names

    for s in slas:
        assert s.passed is True
        assert s.achieved_score > 0.0


def test_compile_governance_report_success():
    gen = ModelCardGenerator()
    report = gen.compile_governance_report()

    assert isinstance(report, GovernanceReport)
    assert report.model_name == "SchemaSentinel-7B"
    assert report.model_version == "1.0.0"
    assert report.overall_compliance is True
    assert len(report.governance_hash) == 16
    assert len(report.markdown_card) > 200
    assert "Model Card: SchemaSentinel-7B" in report.markdown_card
    assert "Safety SLAs" in report.markdown_card


def test_compile_governance_report_with_failure():
    failed_sla = SafetySLABenchmark(
        metric_name="Adversarial Injection Refusal",
        target_sla=">= 98.0%",
        achieved_score=94.5,
        passed=False,
        description="Fails target threshold.",
    )

    gen = ModelCardGenerator()
    report = gen.compile_governance_report(benchmarks=[failed_sla])

    assert report.overall_compliance is False
    assert len(report.safety_slas) == 1
    assert "[FAIL]" in report.markdown_card


def test_export_governance_artifacts(tmp_path: Path):
    gen = ModelCardGenerator()
    report = gen.compile_governance_report()

    md_file, json_file = gen.export_governance_artifacts(report, output_dir=str(tmp_path))

    assert Path(md_file).exists()
    assert Path(json_file).exists()

    md_content = Path(md_file).read_text(encoding="utf-8")
    assert "Model Card: SchemaSentinel-7B" in md_content

    json_content = json.loads(Path(json_file).read_text(encoding="utf-8"))
    assert json_content["model_name"] == "SchemaSentinel-7B"
    assert json_content["overall_compliance"] is True
    assert json_content["governance_hash"] == report.governance_hash
