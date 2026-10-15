from pathlib import Path
import polars as pl
import pytest
from src.stream_lakehouse import (
    DuckDBCatalogSync,
    CatalogSyncConfig,
    TableSyncRegistration,
    CatalogSyncReport,
)


def test_catalog_sync_config_defaults():
    cfg = CatalogSyncConfig()
    assert cfg.gold_dir == "data/lakehouse/gold"
    assert cfg.silver_dir == "data/lakehouse/silver"
    assert cfg.bronze_dir == "data/lakehouse/bronze"
    assert cfg.db_path == ":memory:"
    assert cfg.enable_hive_partitioning is True


def test_register_parquet_table(tmp_path: Path):
    table_dir = tmp_path / "lakehouse" / "gold" / "sample_kpis"
    partition_dir = table_dir / "tenant_id=alpha" / "event_date=2026-10-15"
    partition_dir.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(
        {
            "query_count": [10, 20],
            "latency_mean": [15.5, 22.1],
        }
    )
    df.write_parquet(partition_dir / "part_0.parquet")

    syncer = DuckDBCatalogSync()
    glob_pattern = f"{table_dir}/**/*.parquet"
    reg = syncer.register_parquet_table("sample_kpis", glob_pattern)

    assert isinstance(reg, TableSyncRegistration)
    assert reg.table_name == "sample_kpis"
    assert reg.row_count == 2
    assert "query_count" in reg.column_names
    assert "tenant_id" in reg.column_names
    assert "event_date" in reg.column_names
    assert "sample_kpis" in syncer.get_registered_tables()
    syncer.close()


def test_sync_gold_and_all(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    silver_dir = tmp_path / "silver"
    bronze_dir = tmp_path / "bronze"

    g_table = gold_dir / "kpi_tenant_daily" / "tenant_id=t1" / "event_date=2026-10-15"
    g_table.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"total_queries": [5], "error_rate": [0.0]}).write_parquet(g_table / "kpi.parquet")

    s_table = silver_dir / "tenant_id=t1" / "event_date=2026-10-15"
    s_table.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"event_id": ["e1"], "is_valid": [True]}).write_parquet(s_table / "silver.parquet")

    b_table = bronze_dir / "tenant_id=t1" / "event_date=2026-10-15"
    b_table.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"event_id": ["e1"], "raw": ["{}"]}).write_parquet(b_table / "bronze.parquet")

    cfg = CatalogSyncConfig(
        gold_dir=str(gold_dir),
        silver_dir=str(silver_dir),
        bronze_dir=str(bronze_dir),
    )
    syncer = DuckDBCatalogSync(cfg)
    report = syncer.sync_all()

    assert isinstance(report, CatalogSyncReport)
    assert report.total_tables_registered == 3
    assert "kpi_tenant_daily" in report.registrations
    assert "silver_events" in report.registrations
    assert "bronze_events" in report.registrations
    assert report.catalog_status == "healthy"

    polars_res = syncer.query("SELECT total_queries, tenant_id FROM kpi_tenant_daily")
    assert isinstance(polars_res, pl.DataFrame)
    assert polars_res.shape[0] == 1
    assert polars_res["total_queries"][0] == 5
    assert polars_res["tenant_id"][0] == "t1"

    syncer.close()


def test_missing_directory_handling(tmp_path: Path):
    cfg = CatalogSyncConfig(
        gold_dir=str(tmp_path / "non_existent_gold"),
        silver_dir=str(tmp_path / "non_existent_silver"),
        bronze_dir=str(tmp_path / "non_existent_bronze"),
    )
    syncer = DuckDBCatalogSync(cfg)
    assert syncer.sync_gold_catalog() == {}
    assert syncer.sync_silver_catalog() is None
    assert syncer.sync_bronze_catalog() is None

    report = syncer.sync_all()
    assert report.total_tables_registered == 0
    assert report.registrations == {}
    syncer.close()
