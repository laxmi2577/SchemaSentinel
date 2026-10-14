import json
from pathlib import Path
import pytest
from src.llmops.drift_monitor import (
    DriftMonitorConfig,
    DriftMetricResult,
    DriftReport,
    compute_kolmogorov_smirnov,
    compute_population_stability_index,
    extract_schema_complexity_metrics,
    extract_token_length_distribution,
    EvidentlyDriftMonitor,
)


def test_drift_monitor_config_defaults():
    config = DriftMonitorConfig()
    assert config.ks_alpha == 0.05
    assert config.psi_threshold_warning == 0.1
    assert config.psi_threshold_drift == 0.25
    assert config.min_samples == 5
    assert config.num_bins == 10
    assert config.dry_run is False


def test_compute_kolmogorov_smirnov():
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    d_stat, p_val = compute_kolmogorov_smirnov(s1, s1)
    assert d_stat == 0.0
    assert p_val == 1.0

    s2 = [10.0, 20.0, 30.0, 40.0, 50.0]
    d_stat2, p_val2 = compute_kolmogorov_smirnov(s1, s2)
    assert d_stat2 == 1.0
    assert p_val2 < 0.05


def test_compute_population_stability_index():
    ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    psi_stable = compute_population_stability_index(ref, ref)
    assert psi_stable < 0.05

    shifted = [8.0, 9.0, 10.0, 10.0, 10.0, 11.0, 12.0, 12.0, 13.0, 15.0]
    psi_drift = compute_population_stability_index(ref, shifted)
    assert psi_drift > 0.1


def test_extract_schema_complexity_metrics():
    simple_query = "SELECT id FROM users;"
    complex_query = (
        "WITH cte AS (SELECT id FROM users) "
        "SELECT u.id, o.total FROM cte AS u "
        "JOIN orders AS o ON u.id = o.user_id "
        "WHERE o.total > 100 AND o.status = 'PAID' "
        "GROUP BY u.id, o.total;"
    )

    metrics = extract_schema_complexity_metrics([simple_query, complex_query])
    assert len(metrics) == 2
    assert metrics[1] > metrics[0]


def test_extract_token_length_distribution():
    queries = ["SELECT 1;", "SELECT a, b, c FROM table_name;"]
    lens = extract_token_length_distribution(queries)
    assert lens[0] == 2.0
    assert lens[1] == 6.0


def test_analyze_schema_complexity_drift():
    monitor = EvidentlyDriftMonitor(config=DriftMonitorConfig(min_samples=3))
    ref_queries = [
        "SELECT id FROM users WHERE id = 1;",
        "SELECT name FROM products WHERE price > 10;",
        "SELECT order_id FROM orders WHERE status = 'PENDING';",
    ]

    result = monitor.analyze_schema_complexity_drift(ref_queries, ref_queries)
    assert result.drift_detected is False
    assert result.status == "stable"
    assert result.statistic == 0.0


def test_analyze_token_length_psi_drift():
    monitor = EvidentlyDriftMonitor(config=DriftMonitorConfig(min_samples=3))
    ref_queries = [
        "SELECT a FROM b;",
        "SELECT c FROM d;",
        "SELECT e FROM f;",
    ]

    result = monitor.analyze_token_length_psi_drift(ref_queries, ref_queries)
    assert result.drift_detected is False
    assert result.status == "stable"


def test_run_drift_audit_and_export(tmp_path: Path):
    monitor = EvidentlyDriftMonitor(config=DriftMonitorConfig(min_samples=3))
    ref_queries = [
        "SELECT id FROM users;",
        "SELECT name FROM accounts;",
        "SELECT total FROM invoices;",
    ]
    prod_queries = [
        "SELECT id FROM users;",
        "SELECT name FROM accounts;",
        "SELECT total FROM invoices;",
    ]

    report = monitor.run_drift_audit(ref_queries, prod_queries)
    assert isinstance(report, DriftReport)
    assert "schema_complexity" in report.metrics
    assert "token_distribution" in report.metrics
    assert report.has_drift is False

    out_file = tmp_path / "drift_report.json"
    exported = monitor.export_drift_report(report, str(out_file))
    assert Path(exported).exists()

    loaded = json.loads(Path(exported).read_text(encoding="utf-8"))
    assert loaded["report_id"] == report.report_id
    assert loaded["has_drift"] == report.has_drift


def test_insufficient_samples_handling():
    monitor = EvidentlyDriftMonitor(config=DriftMonitorConfig(min_samples=10))
    sparse_queries = ["SELECT 1;"]

    result = monitor.analyze_schema_complexity_drift(sparse_queries, sparse_queries)
    assert result.status == "insufficient_samples"
    assert result.drift_detected is False
