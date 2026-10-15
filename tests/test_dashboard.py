import datetime
import plotly.graph_objects as go
import polars as pl
import pytest

from src.analytics_engine import (
    DashboardState,
    DuckDBWarehouse,
    NLQueryResult,
    OLAPConfig,
    StreamlitDashboard,
    compute_kpi_tiles,
    create_anomaly_breakdown_chart,
    create_latency_chart,
    create_tenant_activity_heatmap,
    execute_nl_sql_pipeline,
    run_statistical_comparison,
)
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.serving.local_runner import LocalRunnerConfig, SchemaSentinelCPURunner


def _seed_test_warehouse() -> DuckDBWarehouse:
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    now = datetime.datetime.now()
    events = [
        {
            "event_id": f"e_{i}",
            "tenant_id": "tenant_north" if i < 30 else "tenant_south",
            "schema_id": "schema_crm",
            "user_id": f"u_{i % 4}",
            "timestamp": now - datetime.timedelta(days=i % 5, hours=i % 24),
            "latency_ms": 15.0 + (i * 1.5),
            "is_error": i % 8 == 0,
            "is_anomaly": i % 10 == 0,
            "tokens_used": 40 + i,
            "tables_accessed": 2,
            "ast_complexity": 1.2,
        }
        for i in range(50)
    ]
    wh.insert_events(events)
    return wh


def test_dashboard_state_defaults():
    state = DashboardState()
    assert state.selected_tenant is None
    assert state.time_range_days == 7
    assert state.auto_refresh_seconds == 30
    assert state.dark_mode is True


def test_compute_kpi_tiles_populated():
    wh = _seed_test_warehouse()
    kpis = compute_kpi_tiles(wh)

    assert isinstance(kpis, dict)
    assert kpis["total_query_volume"] == 50
    assert kpis["active_tenant_count"] == 2
    assert kpis["mean_latency_ms"] > 0
    assert kpis["error_rate"] >= 0.0
    assert kpis["anomaly_rate"] >= 0.0
    assert kpis["sla_compliance_pct"] > 80.0
    wh.close()


def test_compute_kpi_tiles_empty():
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    kpis = compute_kpi_tiles(wh)

    assert kpis["total_query_volume"] == 0
    assert kpis["active_tenant_count"] == 0
    assert kpis["mean_latency_ms"] == 0.0
    assert kpis["sla_compliance_pct"] == 100.0
    wh.close()


def test_create_latency_chart():
    wh = _seed_test_warehouse()
    matrix = wh.get_tenant_performance_matrix()
    fig = create_latency_chart(matrix)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert fig.data[0].name == "p50 Latency (ms)"
    assert fig.data[1].name == "p90 Latency (ms)"
    assert fig.data[2].name == "p99 Latency (ms)"
    wh.close()


def test_create_anomaly_breakdown_chart():
    wh = _seed_test_warehouse()
    matrix = wh.get_tenant_performance_matrix()
    fig = create_anomaly_breakdown_chart(matrix)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Nominal Queries"
    assert fig.data[1].name == "Anomalous Queries"
    wh.close()


def test_create_tenant_activity_heatmap():
    wh = _seed_test_warehouse()
    fig = create_tenant_activity_heatmap(wh)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    wh.close()


def test_execute_nl_sql_pipeline_valid():
    wh = _seed_test_warehouse()
    runner = SchemaSentinelCPURunner(LocalRunnerConfig(dry_run=True))
    validator = SQLPolicyASTValidator()

    question = "Show query count by tenant"
    result, df = execute_nl_sql_pipeline(
        question=question,
        warehouse=wh,
        runner=runner,
        validator=validator,
    )

    assert isinstance(result, NLQueryResult)
    assert result.is_valid_syntax is True
    assert result.is_policy_compliant is True
    assert result.execution_success is True
    assert result.row_count > 0
    assert isinstance(df, pl.DataFrame)
    wh.close()


def test_execute_nl_sql_pipeline_mutation_blocked():
    wh = _seed_test_warehouse()

    class MockMutationRunner:
        def generate_query(self, text: str):
            class MockRes:
                extracted_sql = "DROP TABLE fact_query_events;"
            return MockRes()

    result, df = execute_nl_sql_pipeline(
        question="Drop the events table",
        warehouse=wh,
        runner=MockMutationRunner(),
    )

    assert result.execution_success is False
    assert result.is_policy_compliant is False or result.is_valid_syntax is False
    assert "violation" in result.error_message.lower()
    assert len(df) == 0
    wh.close()


def test_run_statistical_comparison():
    wh = _seed_test_warehouse()
    comp = run_statistical_comparison(
        warehouse=wh,
        tenant_a="tenant_north",
        tenant_b="tenant_south",
        metric_column="latency_ms",
    )

    assert "hypothesis_result" in comp
    assert "comparison_chart" in comp
    assert isinstance(comp["comparison_chart"], go.Figure)
    assert comp["sample_a_count"] == 30
    assert comp["sample_b_count"] == 20
    assert comp["hypothesis_result"].sample_size_a == 30
    assert comp["hypothesis_result"].sample_size_b == 20
    wh.close()


def test_streamlit_dashboard_render_layout():
    wh = _seed_test_warehouse()
    dashboard = StreamlitDashboard(warehouse=wh)
    layout = dashboard.render_layout()

    assert "kpi_tiles" in layout
    assert "performance_matrix" in layout
    assert "latency_chart" in layout
    assert "anomaly_chart" in layout
    assert "activity_heatmap" in layout
    assert "cohorts" in layout

    assert layout["kpi_tiles"]["total_query_volume"] == 50
    assert len(layout["cohorts"]) == 2
    wh.close()
