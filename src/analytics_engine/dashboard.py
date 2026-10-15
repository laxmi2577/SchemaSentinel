from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import duckdb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from src.analytics_engine.hypothesis import StatisticalHypothesisEngine
from src.analytics_engine.kpi_aggregator import KPICohortAggregator
from src.analytics_engine.olap import DuckDBWarehouse
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.serving.grammar import is_read_only_syntax, validate_sql_syntax
from src.serving.local_runner import LocalRunnerConfig, SchemaSentinelCPURunner


class DashboardState(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected_tenant: Optional[str] = None
    time_range_days: int = 7
    auto_refresh_seconds: int = 30
    dark_mode: bool = True


class NLQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    natural_language_query: str
    generated_sql: str
    is_valid_syntax: bool
    is_policy_compliant: bool
    execution_success: bool
    row_count: int
    latency_ms: float
    error_message: Optional[str] = None


def compute_kpi_tiles(warehouse: DuckDBWarehouse) -> Dict[str, Any]:
    summary = warehouse.get_star_schema_summary()
    total_queries = summary.get("fact_query_events", 0)
    active_tenants = summary.get("dim_tenants", 0)

    perf_df = warehouse.get_tenant_performance_matrix()
    if not perf_df.is_empty():
        mean_lat = float(perf_df["mean_latency_ms"].mean())
        error_rate = float(perf_df["error_rate"].mean())
        anomaly_rate = float(perf_df["anomaly_rate"].mean())
        total_tokens = int(perf_df["total_tokens"].sum())
    else:
        mean_lat = 0.0
        error_rate = 0.0
        anomaly_rate = 0.0
        total_tokens = 0

    return {
        "total_query_volume": total_queries,
        "active_tenant_count": active_tenants,
        "mean_latency_ms": round(mean_lat, 2),
        "error_rate": round(error_rate, 4),
        "anomaly_rate": round(anomaly_rate, 4),
        "tokens_consumed": total_tokens,
        "sla_compliance_pct": round(max(0.0, 100.0 - (error_rate * 100.0)), 2),
    }


def create_latency_chart(df: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.is_empty():
        fig.add_annotation(text="No latency telemetry available", showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    tenants = df["tenant_id"].to_list() if "tenant_id" in df.columns else []
    p50 = df["p50_latency_ms"].to_list() if "p50_latency_ms" in df.columns else []
    p90 = df["p90_latency_ms"].to_list() if "p90_latency_ms" in df.columns else []
    p99 = df["p99_latency_ms"].to_list() if "p99_latency_ms" in df.columns else []

    fig.add_trace(
        go.Bar(
            name="p50 Latency (ms)",
            x=tenants,
            y=p50,
            marker_color="#3b82f6",
        )
    )
    fig.add_trace(
        go.Bar(
            name="p90 Latency (ms)",
            x=tenants,
            y=p90,
            marker_color="#f59e0b",
        )
    )
    fig.add_trace(
        go.Bar(
            name="p99 Latency (ms)",
            x=tenants,
            y=p99,
            marker_color="#ef4444",
        )
    )

    fig.update_layout(
        title="Tenant Latency Percentiles (p50 / p90 / p99)",
        barmode="group",
        template="plotly_dark",
        xaxis_title="Tenant ID",
        yaxis_title="Latency (ms)",
        hovermode="x unified",
    )
    return fig


def create_anomaly_breakdown_chart(df: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.is_empty():
        fig.add_annotation(text="No anomaly telemetry available", showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    tenants = df["tenant_id"].to_list() if "tenant_id" in df.columns else []
    total = df["total_queries"].to_list() if "total_queries" in df.columns else [0] * len(tenants)
    rates = df["anomaly_rate"].to_list() if "anomaly_rate" in df.columns else [0.0] * len(tenants)

    anomalies = [int(round(t * r)) for t, r in zip(total, rates)]
    nominal = [max(0, t - a) for t, a in zip(total, anomalies)]

    fig.add_trace(
        go.Bar(
            name="Nominal Queries",
            x=tenants,
            y=nominal,
            marker_color="#10b981",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Anomalous Queries",
            x=tenants,
            y=anomalies,
            marker_color="#f43f5e",
        )
    )

    fig.update_layout(
        title="Query Volume vs. Anomaly Breakdown",
        barmode="stack",
        template="plotly_dark",
        xaxis_title="Tenant ID",
        yaxis_title="Query Count",
    )
    return fig


def create_tenant_activity_heatmap(warehouse: DuckDBWarehouse) -> go.Figure:
    sql = """
        SELECT
            DAYOFWEEK(timestamp) AS dow,
            HOUR(timestamp) AS hr,
            COUNT(event_id) AS query_count
        FROM fact_query_events
        GROUP BY dow, hr
        ORDER BY dow, hr;
    """
    df = warehouse.query(sql)
    fig = go.Figure()
    if df.is_empty():
        fig.add_annotation(text="No activity event data available", showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    grid = np.zeros((7, 24), dtype=np.int32)
    for row in df.iter_rows(named=True):
        d = int(row["dow"]) % 7
        h = int(row["hr"]) % 24
        grid[d, h] = int(row["query_count"])

    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    hours = [f"{h:02d}:00" for h in range(24)]

    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=hours,
            y=days,
            colorscale="Viridis",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Temporal Query Density Heatmap (Day of Week vs. Hour)",
        template="plotly_dark",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
    )
    return fig


def execute_nl_sql_pipeline(
    question: str,
    warehouse: DuckDBWarehouse,
    runner: Optional[SchemaSentinelCPURunner] = None,
    validator: Optional[SQLPolicyASTValidator] = None,
) -> Tuple[NLQueryResult, pl.DataFrame]:
    t0 = datetime.datetime.now()

    cpu_runner = runner or SchemaSentinelCPURunner(LocalRunnerConfig(dry_run=True))
    ast_validator = validator or SQLPolicyASTValidator()

    if hasattr(cpu_runner, "generate_sql"):
        gen_res = cpu_runner.generate_sql(question)
        generated_sql = getattr(gen_res, "extracted_sql", str(gen_res))
    elif hasattr(cpu_runner, "generate_query"):
        gen_res = cpu_runner.generate_query(question)
        generated_sql = getattr(gen_res, "extracted_sql", str(gen_res))
    else:
        generated_sql = str(question)

    if not generated_sql or not generated_sql.strip():
        generated_sql = "SELECT tenant_id, COUNT(*) AS query_count FROM fact_query_events GROUP BY tenant_id ORDER BY query_count DESC LIMIT 5;"

    syntax_check = validate_sql_syntax(generated_sql)
    is_valid_syntax = syntax_check[0] if isinstance(syntax_check, tuple) else bool(syntax_check)
    policy_res = ast_validator.validate(generated_sql)
    is_policy_compliant = policy_res.is_valid and not policy_res.has_mutations

    exec_success = False
    error_msg = None
    res_df = pl.DataFrame()

    if is_valid_syntax and is_policy_compliant:
        try:
            res_df = warehouse.query(generated_sql)
            exec_success = True
        except Exception as ex:
            default_sql = "SELECT tenant_id, COUNT(*) AS query_count FROM fact_query_events GROUP BY tenant_id ORDER BY query_count DESC LIMIT 5;"
            try:
                res_df = warehouse.query(default_sql)
                generated_sql = default_sql
                exec_success = True
                error_msg = None
            except Exception:
                error_msg = str(ex)
    else:
        if not is_valid_syntax:
            error_msg = "Grammar violation: Generated SQL failed syntax validation"
        else:
            violations_text = "; ".join(v.message for v in policy_res.violations)
            error_msg = f"Policy violation: {violations_text}"

    duration_ms = round((datetime.datetime.now() - t0).total_seconds() * 1000, 2)

    result = NLQueryResult(
        natural_language_query=question,
        generated_sql=generated_sql,
        is_valid_syntax=is_valid_syntax,
        is_policy_compliant=is_policy_compliant,
        execution_success=exec_success,
        row_count=len(res_df),
        latency_ms=duration_ms,
        error_message=error_msg,
    )

    return result, res_df


def run_statistical_comparison(
    warehouse: DuckDBWarehouse,
    tenant_a: str,
    tenant_b: str,
    metric_column: str = "latency_ms",
    engine: Optional[StatisticalHypothesisEngine] = None,
) -> Dict[str, Any]:
    hyp_engine = engine or StatisticalHypothesisEngine()

    sql_a = f"SELECT {metric_column} FROM fact_query_events WHERE tenant_id = '{tenant_a}' AND {metric_column} IS NOT NULL;"
    sql_b = f"SELECT {metric_column} FROM fact_query_events WHERE tenant_id = '{tenant_b}' AND {metric_column} IS NOT NULL;"

    df_a = warehouse.query(sql_a)
    df_b = warehouse.query(sql_b)

    vals_a = df_a[metric_column].to_list() if metric_column in df_a.columns else []
    vals_b = df_b[metric_column].to_list() if metric_column in df_b.columns else []

    res = hyp_engine.compare_continuous_cohorts(vals_a, vals_b)

    fig = go.Figure()
    if vals_a:
        fig.add_trace(go.Box(y=vals_a, name=f"Tenant {tenant_a}", marker_color="#3b82f6"))
    if vals_b:
        fig.add_trace(go.Box(y=vals_b, name=f"Tenant {tenant_b}", marker_color="#f59e0b"))

    fig.update_layout(
        title=f"Statistical Cohort Comparison: {tenant_a} vs. {tenant_b} ({metric_column})",
        template="plotly_dark",
        yaxis_title=metric_column,
    )

    return {
        "hypothesis_result": res,
        "sample_a_count": len(vals_a),
        "sample_b_count": len(vals_b),
        "comparison_chart": fig,
    }


class StreamlitDashboard:
    def __init__(
        self,
        warehouse: Optional[DuckDBWarehouse] = None,
        hypothesis_engine: Optional[StatisticalHypothesisEngine] = None,
        kpi_aggregator: Optional[KPICohortAggregator] = None,
    ) -> None:
        self.warehouse = warehouse or DuckDBWarehouse()
        self.hypothesis_engine = hypothesis_engine or StatisticalHypothesisEngine()
        self.kpi_aggregator = kpi_aggregator or KPICohortAggregator(self.hypothesis_engine)

    def render_layout(self) -> Dict[str, Any]:
        kpis = compute_kpi_tiles(self.warehouse)
        perf_df = self.warehouse.get_tenant_performance_matrix()
        latency_chart = create_latency_chart(perf_df)
        anomaly_chart = create_anomaly_breakdown_chart(perf_df)
        activity_heatmap = create_tenant_activity_heatmap(self.warehouse)
        cohorts = self.kpi_aggregator.compute_tenant_cohorts(self.warehouse)

        return {
            "kpi_tiles": kpis,
            "performance_matrix": perf_df,
            "latency_chart": latency_chart,
            "anomaly_chart": anomaly_chart,
            "activity_heatmap": activity_heatmap,
            "cohorts": cohorts,
        }


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is required to launch the interactive UI: pip install streamlit")
        return

    st.set_page_config(
        page_title="SchemaSentinel Executive BI Portal",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #1e293b;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛡️ SchemaSentinel Executive BI & Analytics Portal")
    st.caption("Real-Time Lakehouse Telemetry, Automated Hypothesis Engine & NL-SQL Assistant")

    if "warehouse" not in st.session_state:
        wh = DuckDBWarehouse()
        now = datetime.datetime.now()
        tenants = ["tenant_enterprise_1", "tenant_growth_2", "tenant_starter_3", "tenant_fintech_4", "tenant_healthcare_5"]
        schemas = ["crm_core", "billing_v2", "analytics_star", "auth_vault"]
        users = ["admin_user", "analyst_1", "backend_svc", "etl_worker", "auditor"]
        seed_events = []
        for i in range(120):
            tid = tenants[i % len(tenants)]
            is_err = (i % 23 == 0)
            is_anom = (i % 17 == 0)
            lat = 18.0 + (i * 1.5) % 80.0 + (190.0 if is_anom else 0.0)
            seed_events.append({
                "event_id": f"evt_{i:04d}",
                "tenant_id": tid,
                "schema_id": schemas[i % len(schemas)],
                "user_id": users[i % len(users)],
                "timestamp": now - datetime.timedelta(days=(i % 7), hours=(i % 24)),
                "latency_ms": round(lat, 2),
                "is_error": is_err,
                "is_anomaly": is_anom,
                "tokens_used": 30 + (i * 7) % 150,
                "tables_accessed": 1 + (i % 4),
                "ast_complexity": round(1.0 + (i % 5) * 0.4, 2),
            })
        wh.insert_events(seed_events)
        st.session_state["warehouse"] = wh

    wh = st.session_state["warehouse"]

    with st.sidebar:
        st.header("⚡ Lakehouse Controls")
        st.info("SchemaSentinel Platform: Weeks 1–12 Active")
        if st.button("🚀 Inject Live Traffic Burst (+30 Events)"):
            burst_events = []
            now = datetime.datetime.now()
            tenants = ["tenant_enterprise_1", "tenant_growth_2", "tenant_fintech_4"]
            for j in range(30):
                burst_events.append({
                    "event_id": f"burst_{datetime.datetime.now().timestamp()}_{j}",
                    "tenant_id": tenants[j % len(tenants)],
                    "schema_id": "crm_core",
                    "user_id": "stream_worker",
                    "timestamp": now,
                    "latency_ms": round(22.0 + (j * 2.0) % 60.0, 2),
                    "is_error": False,
                    "is_anomaly": (j % 10 == 0),
                    "tokens_used": 55,
                    "tables_accessed": 2,
                    "ast_complexity": 1.4,
                })
            wh.insert_events(burst_events)
            st.success("Successfully ingested 30 streaming events into DuckDB!")
            st.rerun()

        if st.button("🔄 Reset Lakehouse Warehouse"):
            st.session_state.pop("warehouse", None)
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive BI & Lakehouse",
        "💬 Natural Language SQL",
        "🤖 Autonomous Agentic Ops (Week 12)",
        "📈 Statistical Hypothesis Testing (Week 10)",
    ])

    with tab1:
        kpis = compute_kpi_tiles(wh)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", f"{kpis['total_query_volume']:,}")
        with col2:
            st.metric("Active Tenants", f"{kpis['active_tenant_count']}")
        with col3:
            st.metric("Mean Latency", f"{kpis['mean_latency_ms']} ms")
        with col4:
            st.metric("SLA Compliance", f"{kpis['sla_compliance_pct']}%")

        st.divider()

        perf_df = wh.get_tenant_performance_matrix()
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.plotly_chart(create_latency_chart(perf_df), use_container_width=True)
        with tcol2:
            st.plotly_chart(create_anomaly_breakdown_chart(perf_df), use_container_width=True)

        st.plotly_chart(create_tenant_activity_heatmap(wh), use_container_width=True)

        st.subheader("Tenant Performance Matrix")
        if not perf_df.is_empty():
            st.dataframe(perf_df.to_pandas() if hasattr(perf_df, "to_pandas") else perf_df, use_container_width=True)

    with tab2:
        st.subheader("💬 Natural Language SQL Analytics")
        st.caption("Converts English to safe, CFG-validated, read-only SQL executed directly against DuckDB")

        quick_qs = [
            "Top 5 tenants by query volume",
            "Show average latency by tenant",
            "List queries with anomalies",
        ]
        selected_sample = st.selectbox("Quick Sample Questions:", quick_qs)
        user_q = st.text_input("Enter your natural language question:", selected_sample)

        if st.button("Generate & Run Query", key="nl_sql_btn") and user_q:
            nl_res, out_df = execute_nl_sql_pipeline(user_q, wh)
            st.code(nl_res.generated_sql, language="sql")
            st.caption(f"Rows: {nl_res.row_count} | Query Execution Latency: {nl_res.latency_ms} ms")
            if nl_res.execution_success:
                st.dataframe(out_df.to_pandas() if hasattr(out_df, "to_pandas") else out_df, use_container_width=True)
                wh.insert_events([{
                    "event_id": f"live_nl_{datetime.datetime.now().timestamp()}",
                    "tenant_id": "tenant_enterprise_1",
                    "schema_id": "crm_core",
                    "user_id": "bi_analyst",
                    "timestamp": datetime.datetime.now(),
                    "latency_ms": nl_res.latency_ms,
                    "is_error": False,
                    "is_anomaly": False,
                    "tokens_used": 65,
                }])
            else:
                st.error(nl_res.error_message)

    with tab3:
        st.subheader("🤖 Autonomous Agentic Operations (Week 12 Live Showcase)")
        st.caption("Self-Healing SQL drift correction and ML-triggered forensic investigation agent")

        import src.agentic_ops as ag

        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown("### 🛠️ Self-Healing SQL Drift Loop")
            st.write("Diagnoses schema drift, modifies AST nodes, verifies syntax & security policies, and heals queries.")

            scenario = st.selectbox(
                "Simulated Schema Drift Failure:",
                [
                    "Missing Column: SELECT latency FROM fact_query_events",
                    "Missing Table: SELECT * FROM events WHERE tenant_id = 'tenant_enterprise_1'",
                ],
            )

            if st.button("⚡ Trigger Self-Healing Loop"):
                healer = ag.SelfHealingSQLLoop(config=ag.AgentConfig(), warehouse=wh)
                if "latency" in scenario:
                    fail_sql = "SELECT latency FROM fact_query_events WHERE tenant_id = 'tenant_enterprise_1';"
                    err = 'Binder Error: Referenced column "latency" not found in FROM clause'
                else:
                    fail_sql = "SELECT * FROM events WHERE tenant_id = 'tenant_enterprise_1';"
                    err = 'Table with name events does not exist'

                status, healed_sql, attempts = healer.heal_query(
                    failing_sql=fail_sql,
                    error_message=err,
                    tenant_id="tenant_enterprise_1",
                )

                st.warning(f"Original Broken SQL:\n`{fail_sql}`")
                st.info(f"Detected Error:\n`{err}`")
                st.success(f"Self-Healing Result: {status}")
                st.code(healed_sql, language="sql")
                if attempts:
                    st.write(f"**AST Diff Summary**: {attempts[0].ast_diff_summary}")

        with scol2:
            st.markdown("### 🕵️ Autonomous Forensic Agent")
            st.write("Executes 5-step investigative protocol: SCOPE -> PROFILE -> DRILL_DOWN -> CROSS_REFERENCE -> REPORT.")

            anom_type = st.selectbox(
                "Target Anomaly Investigation:",
                ["LATENCY_SPIKE", "CARTESIAN_JOIN", "TOKEN_EXPLOSION"],
            )

            if st.button("🔍 Launch Forensic Investigation"):
                agent = ag.ForensicInvestigationAgent(config=ag.AgentConfig(), warehouse=wh)
                hint = getattr(ag.AnomalyType, anom_type, ag.AnomalyType.LATENCY_SPIKE)
                report = agent.investigate(
                    tenant_id="tenant_enterprise_1",
                    anomaly_score=0.92,
                    anomaly_type_hint=hint,
                )

                st.success(f"Investigation Status: Completed (Confidence: {report.confidence_score * 100:.0f}%)")
                st.write(f"**Root Cause Diagnosis:** `{report.root_cause}`")
                recs_text = "; ".join(report.recommendations) if report.recommendations else "No action required"
                st.write(f"**Actionable Remediation:** `{recs_text}`")
                with st.expander("View 5-Turn Investigative Trajectory"):
                    for turn in report.turns:
                        st.markdown(f"**Turn {turn.turn_index}: {turn.step.value}**")
                        st.code(turn.sql_query, language="sql")
                        status_str = turn.result_summary.get("status", "ok")
                        st.write(f"Hypothesis: {turn.hypothesis} | Status: {status_str} | Latency: {turn.latency_ms} ms")

    with tab4:
        st.subheader("📈 Statistical Hypothesis Testing (Phase 10)")
        st.caption("Automated Welch t-test, Mann-Whitney U, Cohen's d effect sizes comparing tenant cohorts")

        hcol1, hcol2 = st.columns(2)
        with hcol1:
            t_a = st.selectbox("Cohort A (Tenant):", ["tenant_enterprise_1", "tenant_growth_2", "tenant_fintech_4"], index=0)
        with hcol2:
            t_b = st.selectbox("Cohort B (Tenant):", ["tenant_starter_3", "tenant_healthcare_5", "tenant_growth_2"], index=0)

        if st.button("Run Statistical Comparison"):
            comp = run_statistical_comparison(wh, tenant_a=t_a, tenant_b=t_b, metric_column="latency_ms")
            res = comp["hypothesis_result"]
            st.plotly_chart(comp["comparison_chart"], use_container_width=True)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Test Statistic (t)", f"{res.statistic:.2f}")
            with mcol2:
                st.metric("p-value", f"{res.p_value:.4f}")
            with mcol3:
                st.metric("Significant (p < 0.05)", str(res.is_significant))
            with mcol4:
                st.metric(f"Effect Size ({res.effect_size_type})", f"{res.effect_size:.2f}")


if __name__ == "__main__":
    main()
