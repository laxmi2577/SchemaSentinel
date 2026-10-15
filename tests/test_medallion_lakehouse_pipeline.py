import asyncio
import os
import psutil
from pathlib import Path
import polars as pl
import pytest
from src.stream_lakehouse import (
    StreamEvent,
    StreamConfig,
    StreamProducer,
    DLQRouter,
    DLQParquetSink,
    SchemaEvolutionDetector,
    BronzeWriter,
    BronzeWriterConfig,
    SilverTransformer,
    SilverTransformerConfig,
    GoldAggregator,
    GoldAggregatorConfig,
    GoldAggregationSummary,
    DuckDBCatalogSync,
    CatalogSyncConfig,
    classify_failure,
)
from src.data_engine.schema_parser import DatabaseCatalog, TableMetadata, ColumnMetadata


def test_medallion_lakehouse_full_lifecycle_e2e(tmp_path: Path):
    bronze_dir = str(tmp_path / "lakehouse" / "bronze")
    silver_dir = str(tmp_path / "lakehouse" / "silver")
    gold_dir = str(tmp_path / "lakehouse" / "gold")

    col_id = ColumnMetadata(name="id", data_type="INTEGER", is_primary_key=True)
    col_amount = ColumnMetadata(name="amount", data_type="REAL")
    table_orders = TableMetadata(
        name="orders",
        columns={"id": col_id, "amount": col_amount},
        primary_keys=["id"],
    )
    catalog = DatabaseCatalog(db_id="e2e_db", tables={"orders": table_orders})

    events = [
        StreamEvent(
            tenant_id="tenant_corp",
            user_id="user_admin",
            natural_query="SELECT id, amount FROM orders WHERE amount > 100;",
            generated_sql="SELECT id, amount FROM orders WHERE amount > 100;",
            tables_accessed=["orders"],
            columns_accessed=["id", "amount"],
            latency_ms=15.0,
            tokens_used=45,
            is_valid=True,
            anomaly_score=0.02,
        ),
        StreamEvent(
            tenant_id="tenant_corp",
            user_id="user_analyst",
            natural_query="SELECT id, amount FROM orders WHERE amount > 100;",
            generated_sql="SELECT id, amount FROM orders WHERE amount > 100;",
            tables_accessed=["orders"],
            columns_accessed=["id", "amount"],
            latency_ms=16.0,
            tokens_used=45,
            is_valid=True,
            anomaly_score=0.03,
        ),
        StreamEvent(
            tenant_id="tenant_retail",
            user_id="user_clerk",
            natural_query="SELECT COUNT(*) FROM orders;",
            generated_sql="SELECT COUNT(*) FROM orders;",
            tables_accessed=["orders"],
            columns_accessed=["id"],
            latency_ms=8.0,
            tokens_used=25,
            is_valid=True,
            anomaly_score=0.01,
        ),
        StreamEvent(
            tenant_id="tenant_retail",
            user_id="user_intruder",
            natural_query="SELECT secret FROM shadow_table;",
            generated_sql="SELECT secret FROM shadow_table;",
            tables_accessed=["shadow_table"],
            columns_accessed=["secret"],
            latency_ms=40.0,
            tokens_used=60,
            is_valid=False,
            anomaly_score=0.95,
        ),
    ]

    bronze_writer = BronzeWriter(BronzeWriterConfig(base_dir=bronze_dir))
    bronze_paths = bronze_writer.append_batch(events)
    assert len(bronze_paths) >= 2
    assert bronze_writer.get_total_events_count() == 4

    silver_transformer = SilverTransformer(
        config=SilverTransformerConfig(base_dir=silver_dir, bronze_dir=bronze_dir),
        catalog=catalog,
    )
    silver_paths, silver_metrics = silver_transformer.process_bronze_to_silver()
    assert len(silver_paths) >= 2
    assert silver_metrics.total_input_records == 4
    assert silver_metrics.duplicates_dropped == 1
    assert silver_metrics.catalog_violations_dropped == 1
    assert silver_metrics.total_silver_records == 2

    gold_aggregator = GoldAggregator(
        GoldAggregatorConfig(base_dir=gold_dir, silver_dir=silver_dir)
    )
    gold_summary = gold_aggregator.materialize_from_silver()
    assert isinstance(gold_summary, GoldAggregationSummary)
    assert gold_summary.total_silver_records_processed == 2
    assert "kpi_tenant_daily" in gold_summary.materialized_tables
    assert "hourly_rollups" in gold_summary.materialized_tables
    assert "tenant_cohorts" in gold_summary.materialized_tables

    catalog_syncer = DuckDBCatalogSync(
        CatalogSyncConfig(
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            bronze_dir=bronze_dir,
        )
    )
    sync_report = catalog_syncer.sync_all()
    assert sync_report.total_tables_registered >= 3
    assert "kpi_tenant_daily" in sync_report.registrations
    assert "silver_events" in sync_report.registrations
    assert "bronze_events" in sync_report.registrations

    query_res = catalog_syncer.query(
        "SELECT tenant_id, total_queries, error_rate FROM kpi_tenant_daily ORDER BY tenant_id"
    )
    assert isinstance(query_res, pl.DataFrame)
    assert query_res.shape[0] == 2
    tenants = query_res["tenant_id"].to_list()
    assert "tenant_corp" in tenants
    assert "tenant_retail" in tenants

    silver_query_res = catalog_syncer.query(
        "SELECT count(*) as cnt FROM silver_events"
    )
    assert silver_query_res["cnt"][0] == 2

    catalog_syncer.close()


def test_medallion_schema_evolution_and_silver_sync(tmp_path: Path):
    silver_dir = str(tmp_path / "lakehouse" / "silver")
    detector = SchemaEvolutionDetector()

    old_ddl = "CREATE TABLE customers (id INT, name TEXT);"
    new_ddl = "CREATE TABLE customers (id INT, name TEXT, tier VARCHAR(50));"

    diff = detector.diff_ddl(old_ddl=old_ddl, new_ddl=new_ddl)
    assert diff.is_breaking is False
    assert "tier" in diff.added_columns

    evt = detector.create_change_event(
        tenant_id="tenant_growth",
        old_ddl=old_ddl,
        new_ddl=new_ddl,
    )
    assert evt.is_breaking is False
    assert evt.change_type == "ADD_COLUMN"

    breaking_ddl = "CREATE TABLE customers (id INT);"
    breaking_diff = detector.diff_ddl(old_ddl=old_ddl, new_ddl=breaking_ddl)
    assert breaking_diff.is_breaking is True
    assert "name" in breaking_diff.dropped_columns

    breaking_evt = detector.create_change_event(
        tenant_id="tenant_growth",
        old_ddl=old_ddl,
        new_ddl=breaking_ddl,
    )
    assert breaking_evt.is_breaking is True


def test_medallion_dlq_quarantine_isolation(tmp_path: Path):
    dlq_dir = str(tmp_path / "lakehouse" / "dlq")
    sink = DLQParquetSink(base_dir=dlq_dir)
    cfg = StreamConfig(dlq_threshold_retries=2)
    router = DLQRouter(config=cfg, parquet_sink=sink)

    malformed_event = StreamEvent(
        tenant_id="tenant_malicious",
        natural_query="DROP TABLE critical_ledger;",
        generated_sql="DROP TABLE critical_ledger;",
        is_valid=False,
    )

    failed, reason, cat = classify_failure(malformed_event)
    assert failed is True
    assert cat == "AST_VALIDATION_ERROR"

    dlq_record = router.process_and_route(malformed_event)
    assert dlq_record is not None
    assert dlq_record.error_category == "AST_VALIDATION_ERROR"
    assert dlq_record.retry_count == 0

    replayed1 = router.replay(dlq_record.dlq_id)
    assert replayed1 is not None
    assert replayed1.tenant_id == "tenant_malicious"

    replayed2 = router.replay(dlq_record.dlq_id)
    assert replayed2 is not None

    replayed3 = router.replay(dlq_record.dlq_id)
    assert replayed3 is None

    records = sink.read_records(tenant_id="tenant_malicious")
    assert len(records) >= 1
    assert records[0]["tenant_id"] == "tenant_malicious"
    assert records[0]["error_category"] == "AST_VALIDATION_ERROR"


def test_medallion_control_plane_ram_budget(tmp_path: Path):
    bronze_dir = str(tmp_path / "lakehouse" / "bronze")
    silver_dir = str(tmp_path / "lakehouse" / "silver")
    gold_dir = str(tmp_path / "lakehouse" / "gold")

    bw = BronzeWriter(BronzeWriterConfig(base_dir=bronze_dir))
    batch = [
        StreamEvent(
            tenant_id=f"tenant_{i % 5}",
            natural_query=f"SELECT col FROM table_{i % 3};",
            generated_sql=f"SELECT col FROM table_{i % 3};",
            latency_ms=10.0 + (i % 20),
            tokens_used=20 + i,
            is_valid=(i % 10 != 0),
            anomaly_score=0.01 * (i % 50),
        )
        for i in range(200)
    ]
    bw.append_batch(batch)

    st = SilverTransformer(SilverTransformerConfig(base_dir=silver_dir, bronze_dir=bronze_dir))
    st.process_bronze_to_silver()

    ga = GoldAggregator(GoldAggregatorConfig(base_dir=gold_dir, silver_dir=silver_dir))
    ga.materialize_from_silver()

    syncer = DuckDBCatalogSync(
        CatalogSyncConfig(gold_dir=gold_dir, silver_dir=silver_dir, bronze_dir=bronze_dir)
    )
    report = syncer.sync_all()
    assert report.total_tables_registered >= 3

    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    assert rss_mb < 750.0
    syncer.close()
