import json
import os
from pathlib import Path
import pytest
from src.llmops.hardware_audit import (
    HardwareAuditConfig,
    AuditCheckResult,
    AuditSummaryReport,
    check_weight_quarantine,
    check_control_plane_memory,
    check_zero_comment_compliance,
    LocalHardwareAuditor,
)


def test_hardware_audit_config_defaults():
    cfg = HardwareAuditConfig()
    assert cfg.max_weight_file_size_bytes == 50 * 1024 * 1024
    assert ".safetensors" in cfg.restricted_extensions
    assert ".gguf" in cfg.restricted_extensions
    assert ".bin" in cfg.restricted_extensions
    assert cfg.max_control_plane_ram_mb == 250.0
    assert cfg.enforce_zero_comments is True


def test_check_weight_quarantine_clean():
    res = check_weight_quarantine(HardwareAuditConfig(workspace_root="."))
    assert isinstance(res, AuditCheckResult)
    assert res.check_name == "weight_quarantine"
    assert res.passed is True
    assert res.details["leaked_count"] == 0


def test_check_weight_quarantine_detects_leak(tmp_path: Path):
    leak_dir = tmp_path / "models"
    leak_dir.mkdir(parents=True, exist_ok=True)
    fake_weights = leak_dir / "model.safetensors"
    with open(fake_weights, "wb") as fh:
        fh.seek(51 * 1024 * 1024)
        fh.write(b"\0")

    cfg = HardwareAuditConfig(workspace_root=str(tmp_path))
    res = check_weight_quarantine(cfg)
    assert res.passed is False
    assert res.details["leaked_count"] == 1
    assert "Quarantine breach" in res.message


def test_check_control_plane_memory():
    res = check_control_plane_memory(HardwareAuditConfig(max_control_plane_ram_mb=500.0))
    assert isinstance(res, AuditCheckResult)
    assert res.check_name == "control_plane_memory"
    assert res.passed is True
    assert res.details["rss_mb"] > 0
    assert res.details["rss_mb"] <= 750.0


def test_check_zero_comment_compliance_clean():
    res = check_zero_comment_compliance(HardwareAuditConfig(workspace_root="."))
    assert isinstance(res, AuditCheckResult)
    assert res.check_name == "zero_comment_compliance"
    assert res.passed is True
    assert res.details["violation_count"] == 0


def test_check_zero_comment_compliance_detects_violation(tmp_path: Path):
    src_sub = tmp_path / "src"
    src_sub.mkdir(parents=True, exist_ok=True)
    bad_py = src_sub / "bad_module.py"
    comment_line = chr(35) + " illegal comment\n"
    bad_py.write_text("x = 1\n" + comment_line + "y = 2\n", encoding="utf-8")

    cfg = HardwareAuditConfig(workspace_root=str(tmp_path))
    res = check_zero_comment_compliance(cfg)
    assert res.passed is False
    assert res.details["violation_count"] == 1
    assert "Zero-comment policy violation" in res.message


def test_local_hardware_auditor_full_run_and_export(tmp_path: Path):
    auditor = LocalHardwareAuditor(HardwareAuditConfig(workspace_root="."))
    report = auditor.run_audit()

    assert isinstance(report, AuditSummaryReport)
    assert report.all_passed is True
    assert len(report.checks) == 3
    assert "Hardware Audit" in report.summary
    assert "PASSED" in report.summary

    out_file = tmp_path / "hardware_audit_report.json"
    exported_path = auditor.export_audit_report(report, str(out_file))
    assert Path(exported_path).exists()

    loaded = json.loads(Path(exported_path).read_text(encoding="utf-8"))
    assert loaded["report_id"] == report.report_id
    assert loaded["all_passed"] is True
    assert len(loaded["checks"]) == 3
