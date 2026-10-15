from pathlib import Path
import polars as pl
import pytest
from src.stream_lakehouse import (
    StreamEvent,
    BronzeWriter,
    BronzeWriterConfig,
    SilverTransformer,
    SilverTransformerConfig,
    GoldAggregator,
    GoldAggregatorConfig,
    GoldAggregationSummary,
)


def test_gold_aggregator_config_defaults():
    cfg = GoldAggregatorConfig()
    assert cfg.base_dir == "data/lakehouse/gold"
    assert cfg.silver_dir == "data/lakehouse/silver"
    assert cfg.compression == "snappy"
    assert cfg.anomaly_threshold == 0.5


def test_compute_tenant_kpis():
    agg = GoldAggregator()
    sample_df = pl.DataFrame(
        {
            "tenant_id": ["t_1", "t_1", "t_1", "t_2"],
            "event_date": ["2026-10-15", "2026-10-15", "2026-10-15", "2026-10-15"],
            "latency_ms": [10.0, 20.0, 30.0, 50.0],
            "tokens_used": [10, 20, 30, 100],
            "is_valid": [True, True, False, True],
            "anomaly_score": [0.1, 0.2, 0.8, 0.05],
        }
    )

    kpi_df = agg.compute_tenant_kpis(sample_df)
    assert kpi_df.shape[0] == 2

    t1_row = kpi_df.filter(pl.col("tenant_id") == "t_1")
    assert t1_row["total_queries"][0] == 3
    assert t1_row["valid_queries"][0] == 2
    assert t1_row["invalid_queries"][0] == 1
    assert round(t1_row["error_rate"][0], 2) == 0.33
    assert t1_row["latency_p50"][0] == 20.0
    assert t1_row["total_tokens"][0] == 60
    assert t1_row["anomaly_count"][0] == 1

    t2_row = kpi_df.filter(pl.col("tenant_id") == "t_2")
    assert t2_row["total_queries"][0] == 1
    assert t2_row["valid_queries"][0] == 1
    assert t2_row["invalid_queries"][0] == 0
    assert t2_row["error_rate"][0] == 0.0


def test_compute_hourly_rollups():
    agg = GoldAggregator()
    sample_df = pl.DataFrame(
        {
            "tenant_id": ["t_1", "t_1"],
            "event_date": ["2026-10-15", "2026-10-15"],
            "timestamp": [1700000000.0, 1700000100.0],
            "latency_ms": [15.0, 25.0],
            "tokens_used": [10, 20],
            "is_valid": [True, False],
        }
    )

    rollups = agg.compute_hourly_rollups(sample_df)
    assert rollups.shape[0] == 1
    assert rollups["query_count"][0] == 2
    assert rollups["mean_latency_ms"][0] == 20.0
    assert rollups["error_count"][0] == 1
    assert rollups["total_tokens"][0] == 30


def test_compute_tenant_cohorts():
    agg = GoldAggregator()
    sample_df = pl.DataFrame(
        {
            "tenant_id": ["t_small", "t_error"],
            "event_date": ["2026-10-15", "2026-10-15"],
            "latency_ms": [12.0, 80.0],
            "tokens_used": [50, 150],
            "is_valid": [True, False],
            "anomaly_score": [0.05, 0.9],
        }
    )

    cohorts = agg.compute_tenant_cohorts(sample_df)
    assert cohorts.shape[0] == 2

    small_row = cohorts.filter(pl.col("tenant_id") == "t_small")
    assert small_row["activity_cohort"][0] == "low_volume"
    assert small_row["health_status"][0] == "nominal"

    error_row = cohorts.filter(pl.col("tenant_id") == "t_error")
    assert error_row["health_status"][0] == "degraded"


def test_write_gold_and_read_gold(tmp_path: Path):
    gold_dir = str(tmp_path / "gold")
    agg = GoldAggregator(GoldAggregatorConfig(base_dir=gold_dir))

    sample_kpi = pl.DataFrame(
        {
            "tenant_id": ["t_1", "t_2"],
            "event_date": ["2026-10-15", "2026-10-15"],
            "total_queries": [10, 20],
            "valid_queries": [9, 20],
            "invalid_queries": [1, 0],
            "error_rate": [0.1, 0.0],
            "latency_p50": [15.0, 25.0],
            "latency_p90": [20.0, 30.0],
            "latency_p99": [25.0, 35.0],
            "latency_mean": [16.0, 26.0],
            "total_tokens": [100, 200],
            "anomaly_count": [0, 1],
            "mean_anomaly_score": [0.02, 0.55],
        }
    )

    paths = agg.write_gold_table(sample_kpi, "kpi_tenant_daily")
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()
        assert p.endswith(".parquet")

    read_df = agg.read_gold("kpi_tenant_daily")
    assert read_df.shape[0] == 2

    scan_lf = agg.scan_gold("kpi_tenant_daily")
    assert scan_lf.collect().shape[0] == 2

    partitions = agg.get_partitions("kpi_tenant_daily")
    assert len(partitions) == 2


def test_materialize_from_silver_e2e(tmp_path: Path):
    bronze_dir = str(tmp_path / "bronze")
    silver_dir = str(tmp_path / "silver")
    gold_dir = str(tmp_path / "gold")

    bw = BronzeWriter(BronzeWriterConfig(base_dir=bronze_dir))
    evts = [
        StreamEvent(
            tenant_id="tenant_gold_test",
            natural_query="SELECT COUNT(*) FROM sales;",
            generated_sql="SELECT COUNT(*) FROM sales;",
            tables_accessed=["sales"],
            columns_accessed=["id"],
            latency_ms=14.0,
            tokens_used=35,
            is_valid=True,
            anomaly_score=0.02,
        ),
        StreamEvent(
            tenant_id="tenant_gold_test",
            natural_query="SELECT SUM(total) FROM sales;",
            generated_sql="SELECT SUM(total) FROM sales;",
            tables_accessed=["sales"],
            columns_accessed=["total"],
            latency_ms=28.0,
            tokens_used=40,
            is_valid=True,
            anomaly_score=0.04,
        ),
    ]
    bw.append_batch(evts)

    st = SilverTransformer(SilverTransformerConfig(base_dir=silver_dir, bronze_dir=bronze_dir))
    st.process_bronze_to_silver()

    agg = GoldAggregator(GoldAggregatorConfig(base_dir=gold_dir, silver_dir=silver_dir))
    summary = agg.materialize_from_silver()

    assert isinstance(summary, GoldAggregationSummary)
    assert summary.total_silver_records_processed == 2
    assert "kpi_tenant_daily" in summary.materialized_tables
    assert "hourly_rollups" in summary.materialized_tables
    assert "tenant_cohorts" in summary.materialized_tables

    gold_kpi = agg.read_gold("kpi_tenant_daily")
    assert gold_kpi.shape[0] == 1
    assert gold_kpi["tenant_id"][0] == "tenant_gold_test"
    assert gold_kpi["total_queries"][0] == 2


def test_empty_dataframe_handling(tmp_path: Path):
    agg = GoldAggregator(GoldAggregatorConfig(base_dir=str(tmp_path / "gold")))
    empty = pl.DataFrame()
    assert agg.compute_tenant_kpis(empty).is_empty()
    assert agg.compute_hourly_rollups(empty).is_empty()
    assert agg.compute_tenant_cohorts(empty).is_empty()
    assert agg.write_gold_table(empty, "test_table") == []
    assert agg.read_gold("non_existent").is_empty()
    assert agg.scan_gold("non_existent").collect().is_empty()
    assert agg.get_partitions("non_existent") == []
