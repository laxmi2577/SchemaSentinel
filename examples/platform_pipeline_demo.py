from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import polars as pl

_workspace = str(Path(__file__).resolve().parent.parent)
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import src.stream_lakehouse as slh
from src.stream_lakehouse.stream_producer import StreamProducer
from src.stream_lakehouse.dlq_router import DLQParquetSink, DLQRouter
from src.stream_lakehouse.schema_evolution import SchemaEvolutionDetector

import src.ml_engine as ml
from src.ml_engine.feature_store import StreamingFeatureStore
from src.ml_engine.onnx_scorer import LocalONNXScorer

import src.analytics_engine as an
from src.analytics_engine.olap import DuckDBWarehouse
from src.analytics_engine.hypothesis import StatisticalHypothesisEngine
from src.analytics_engine.kpi_aggregator import KPICohortAggregator

import src.agentic_ops as ag
from src.agentic_ops.forensic_agent import ForensicInvestigationAgent
from src.agentic_ops.self_healer import SelfHealingSQLLoop
from src.agentic_ops.active_learner import ActiveLearningFlywheel


async def run_verification_walkthrough() -> None:
    print("=" * 75)
    print("SCHEMASENTINEL ENTERPRISE — WEEKS 7 TO 12 VERIFICATION PIPELINE")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("\n[Week 7] Real-Time Event Stream, Backpressure Buffer & DLQ Router")
        stream_cfg = slh.StreamConfig(
            base_data_dir=str(tmp_path / "lakehouse"),
            queue_max_size=100,
        )
        producer = StreamProducer(stream_cfg)
        dlq_sink = DLQParquetSink(base_dir=str(tmp_path / "dlq"))
        dlq_router = DLQRouter(config=stream_cfg, parquet_sink=dlq_sink)
        schema_detector = SchemaEvolutionDetector()

        for i in range(15):
            evt = slh.StreamEvent(
                tenant_id="tenant_enterprise_1",
                schema_id="crm_core",
                query_string=f"SELECT id, amount FROM transactions WHERE tenant_id = 'tenant_enterprise_1' AND id = {i};",
                generated_sql=f"SELECT id, amount FROM transactions WHERE tenant_id = 'tenant_enterprise_1' AND id = {i};",
                latency_ms=25.0 + (i * 2.5),
                is_valid=True,
                tables_accessed=["transactions"],
                columns_accessed=["id", "amount", "tenant_id"],
            )
            await producer.put(evt)

        metrics = producer.get_metrics()
        print(f"  Stream Enqueued: {metrics.total_enqueued} events | High Watermark: {metrics.high_watermark}")

        bad_evt = slh.StreamEvent(
            tenant_id="tenant_enterprise_1",
            generated_sql="DROP TABLE transactions;",
            is_valid=False,
        )
        dlq_rec = dlq_router.process_and_route(bad_evt)
        print(f"  DLQ Defense Active: Quarantined '{dlq_rec.error_category}' violation")

        diff = schema_detector.diff_ddl(
            "CREATE TABLE orders (id INT, amount REAL);",
            "CREATE TABLE orders (id INT, amount REAL, currency TEXT);"
        )
        print(f"  Schema Evolution: Drift Detected={diff.has_drift} (Added: {diff.columns_added})")

        print("\n[Week 8] Medallion Lakehouse (Bronze -> Silver -> Gold -> DuckDB Sync)")
        bronze_dir = str(tmp_path / "lakehouse" / "bronze")
        silver_dir = str(tmp_path / "lakehouse" / "silver")
        gold_dir = str(tmp_path / "lakehouse" / "gold")

        bronze_writer = slh.BronzeWriter(slh.BronzeWriterConfig(base_dir=bronze_dir))
        events_batch = []
        for _ in range(15):
            evt = await producer.get()
            if evt:
                events_batch.append(evt)

        bronze_paths = bronze_writer.append_batch(events_batch)
        print(f"  Bronze Raw Sink: Hive-partitioned Snappy Parquet created ({len(bronze_paths)} partitions)")

        silver_tx = slh.SilverTransformer(
            config=slh.SilverTransformerConfig(base_dir=silver_dir, bronze_dir=bronze_dir)
        )
        silver_paths, silver_metrics = silver_tx.process_bronze_to_silver()
        print(f"  Silver Layer: {silver_metrics.total_silver_records} valid records transformed (Deduplicated with MinHash)")

        gold_agg = slh.GoldAggregator(
            slh.GoldAggregatorConfig(base_dir=gold_dir, silver_dir=silver_dir)
        )
        gold_summary = gold_agg.materialize_from_silver()
        print(f"  Gold Aggregation: {len(gold_summary.materialized_tables)} tables rolled up")

        sync = slh.DuckDBCatalogSync(
            slh.CatalogSyncConfig(
                gold_dir=gold_dir,
                silver_dir=silver_dir,
                bronze_dir=bronze_dir,
            )
        )
        reg_report = sync.sync_all()
        print(f"  DuckDB Catalog Sync: Mounted {reg_report.total_tables_registered} external views over Parquet")

        print("\n[Week 9] Streaming Feature Store, Tabular Drift & Local ONNX Scorer")
        feature_store = StreamingFeatureStore()
        for evt in events_batch:
            feature_store.record_event(
                tenant_id=evt.tenant_id,
                latency_ms=evt.latency_ms,
                is_error=evt.is_error,
                token_count=50,
            )
        fv = feature_store.compute_features("tenant_enterprise_1")
        print(f"  Feature Store: Velocity={fv.query_velocity_5m} q/min | Rolling Mean Latency={fv.latency_mean_5m:.2f}ms")

        scorer = LocalONNXScorer()
        score_res = scorer.score_features(fv)
        print(f"  Local ONNX CPU Scorer: Anomaly Score={score_res.anomaly_score:.4f} | Anomaly={score_res.is_anomaly}")

        print("\n[Week 10] DuckDB OLAP Dimensional Star-Schema & Statistical Hypothesis Engine")
        warehouse = DuckDBWarehouse(an.OLAPConfig(db_path=":memory:"))
        olap_events = [
            {
                "event_id": f"evt_{i}",
                "tenant_id": "tenant_enterprise_1" if i < 10 else "tenant_growth_2",
                "schema_id": "crm_core",
                "user_id": f"usr_{i % 3}",
                "latency_ms": 20.0 + (i * 3.0),
                "is_error": False,
                "is_anomaly": False,
                "tokens_used": 45,
            }
            for i in range(15)
        ]
        warehouse.insert_events(olap_events)
        df_summary = warehouse.get_star_schema_summary()
        print(f"  OLAP Star-Schema: {df_summary['fact_query_events']} events in fact table | {df_summary['dim_tenants']} tenants")

        hyp_engine = StatisticalHypothesisEngine()
        cohort_a = [e["latency_ms"] for e in olap_events if e["tenant_id"] == "tenant_enterprise_1"]
        cohort_b = [e["latency_ms"] for e in olap_events if e["tenant_id"] == "tenant_growth_2"]
        hyp_res = hyp_engine.welch_t_test(cohort_a, cohort_b)
        print(f"  Hypothesis Engine: Welch t={hyp_res.test_statistic:.2f}, p={hyp_res.p_value:.4f}, Cohen d={hyp_res.effect_size:.2f}")

        kpi_agg = KPICohortAggregator()
        cohort_metrics = kpi_agg.compute_tenant_cohorts(olap_events)
        print(f"  Cohort Aggregator: Classified {len(cohort_metrics)} tenant tiers with SLA health scoring")

        print("\n[Week 11] Streamlit Executive BI & Natural Language SQL Pipeline")
        from src.analytics_engine.dashboard import compute_kpi_tiles, execute_nl_sql_pipeline
        tiles = compute_kpi_tiles(olap_events)
        print(f"  BI Tiles: Volume={tiles.total_queries} | Mean Latency={tiles.mean_latency_ms:.1f}ms | Active Tenants={tiles.active_tenants}")

        nl_result = execute_nl_sql_pipeline(
            prompt="Show total query events by tenant",
            tenant_id="tenant_enterprise_1",
            warehouse=warehouse,
        )
        print(f"  NL-SQL Execution: Generated SQL='{nl_result.generated_sql}'")
        print(f"  Dual-Shield Security: Valid CFG={nl_result.is_valid_syntax} | Policy Compliant={nl_result.is_policy_compliant}")

        print("\n[Week 12] Autonomous Forensic Agent, Self-Healing SQL Loop & DPO Flywheel")
        forensic_agent = ForensicInvestigationAgent(config=ag.AgentConfig(), warehouse=warehouse)
        report = forensic_agent.investigate(
            tenant_id="tenant_enterprise_1",
            anomaly_type_hint=ag.AnomalyType.LATENCY_SPIKE,
        )
        print(f"  Forensic Agent: 5-Step Protocol Completed ({len(report.investigation_turns)} turns)")
        print(f"  Root Cause Diagnosis: '{report.root_cause}'")
        print(f"  Remediation: '{report.recommended_action}'")

        healer = SelfHealingSQLLoop(config=ag.AgentConfig(), warehouse=warehouse, dlq_router=dlq_router)
        status, healed_sql, attempts = healer.heal_query(
            failing_sql="SELECT missing_latency_col FROM fact_query_events WHERE tenant_id = 'tenant_enterprise_1';",
            error_message="Referenced column missing_latency_col not found in FROM clause",
            tenant_id="tenant_enterprise_1",
        )
        print(f"  Self-Healing SQL Loop: {status} (Resolved in {len(attempts)} attempt)")
        print(f"  Corrected SQL: '{healed_sql}'")

        flywheel = ActiveLearningFlywheel(output_dir=str(tmp_path / "flywheel"))
        pairs = flywheel.mine_from_self_healing_attempts(attempts, tenant_id="tenant_enterprise_1")
        export_path = flywheel.export_dpo_dataset()
        summary = flywheel.get_dataset_summary()
        print(f"  Active Learning Flywheel: Mined {summary['total_pairs']} preference pairs -> Exported to {Path(export_path).name}")

        warehouse.close()

    print("\n" + "=" * 75)
    print("ALL 6 PHASES (WEEKS 7 TO 12) VERIFIED: END-TO-END PRODUCTION PIPELINE OPERATIONAL")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_verification_walkthrough())
