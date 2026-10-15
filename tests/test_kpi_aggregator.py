import datetime
import polars as pl
import pytest

from src.analytics_engine import (
    CohortTier,
    KPICohort,
    OLAPConfig,
    TrendDirection,
    TrendReport,
    DuckDBWarehouse,
    KPICohortAggregator,
)


def test_classify_tenant_tier():
    agg = KPICohortAggregator()

    assert agg.classify_tenant_tier(1500, 0.01, 0.01) == CohortTier.ENTERPRISE
    assert agg.classify_tenant_tier(500, 0.02, 0.02) == CohortTier.GROWTH
    assert agg.classify_tenant_tier(50, 0.01, 0.01) == CohortTier.STARTER
    assert agg.classify_tenant_tier(2000, 0.06, 0.01) == CohortTier.AT_RISK
    assert agg.classify_tenant_tier(2000, 0.01, 0.12) == CohortTier.AT_RISK


def test_compute_health_score():
    agg = KPICohortAggregator(sla_p99_latency_ms=100.0)

    score_perfect = agg.compute_health_score(0.0, 0.0, 20.0)
    assert score_perfect == 100.0

    score_errors = agg.compute_health_score(0.05, 0.0, 20.0)
    assert score_errors < 100.0

    score_sla_breach = agg.compute_health_score(0.0, 0.0, 200.0)
    assert score_sla_breach < 100.0


def test_compute_tenant_cohorts():
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))

    events = [
        {
            "event_id": f"e_{i}",
            "tenant_id": "tenant_1" if i < 120 else "tenant_2",
            "latency_ms": 25.0 if i < 120 else 150.0,
            "is_error": False,
            "is_anomaly": False if i < 120 else True,
            "tokens_used": 30,
        }
        for i in range(150)
    ]
    wh.insert_events(events)

    agg = KPICohortAggregator()
    cohorts = agg.compute_tenant_cohorts(wh)

    assert len(cohorts) == 2
    by_tenant = {c.tenant_id: c for c in cohorts}

    assert "tenant_1" in by_tenant
    assert by_tenant["tenant_1"].cohort_tier == CohortTier.GROWTH
    assert by_tenant["tenant_1"].query_count == 120

    assert "tenant_2" in by_tenant
    assert by_tenant["tenant_2"].cohort_tier == CohortTier.AT_RISK
    assert by_tenant["tenant_2"].query_count == 30

    wh.close()


def test_compare_time_periods_trend_detection():
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    now = datetime.datetime.now()

    baseline_events = [
        {
            "event_id": f"b_{i}",
            "tenant_id": "tenant_trend",
            "timestamp": now - datetime.timedelta(days=10, hours=i),
            "latency_ms": 15.0 + (i % 3),
            "is_error": False,
            "is_anomaly": False,
            "tokens_used": 20,
        }
        for i in range(25)
    ]
    wh.insert_events(baseline_events)

    current_events = [
        {
            "event_id": f"c_{i}",
            "tenant_id": "tenant_trend",
            "timestamp": now - datetime.timedelta(days=2, hours=i),
            "latency_ms": 85.0 + (i % 5),
            "is_error": i % 5 == 0,
            "is_anomaly": i % 4 == 0,
            "tokens_used": 100,
        }
        for i in range(25)
    ]
    wh.insert_events(current_events)

    agg = KPICohortAggregator()
    report = agg.compare_time_periods(wh, tenant_id="tenant_trend", current_days=7, baseline_days=7)

    assert isinstance(report, TrendReport)
    assert report.tenant_id == "tenant_trend"
    assert "latency" in report.metrics
    assert report.metrics["latency"].direction == TrendDirection.INCREASING
    assert report.metrics["latency"].is_significant is True
    assert report.anomalous_drift_detected is True
    assert report.overall_health_score < 100.0

    wh.close()


def test_empty_warehouse_handling():
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    agg = KPICohortAggregator()

    cohorts = agg.compute_tenant_cohorts(wh)
    assert cohorts == []

    report = agg.compare_time_periods(wh, "non_existent_tenant")
    assert report.metrics["query_volume"].current_value == 0.0
    assert report.overall_health_score == 100.0

    wh.close()
