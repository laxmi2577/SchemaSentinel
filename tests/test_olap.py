import datetime
from pathlib import Path
import polars as pl
import pytest

from src.analytics_engine import OLAPConfig, DuckDBWarehouse


def test_olap_config_defaults():
    cfg = OLAPConfig()
    assert cfg.db_path == ":memory:"
    assert cfg.gold_parquet_dir == "data/lakehouse/gold"
    assert cfg.memory_limit == "250MB"
    assert cfg.threads == 2
    assert cfg.enable_hive_partitioning is True
    assert cfg.read_only is False


def test_initialize_star_schema():
    warehouse = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    summary = warehouse.get_star_schema_summary()

    assert "dim_tenants" in summary
    assert "dim_schemas" in summary
    assert "dim_time_periods" in summary
    assert "dim_users" in summary
    assert "fact_query_events" in summary
    assert all(count == 0 for count in summary.values())
    warehouse.close()


def test_insert_events_and_dimension_population():
    warehouse = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))

    events_data = [
        {
            "event_id": f"evt_{i}",
            "tenant_id": "tenant_alpha" if i < 15 else "tenant_beta",
            "schema_id": "schema_finance" if i < 15 else "schema_ops",
            "user_id": f"user_{i % 3}",
            "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=i),
            "latency_ms": 10.0 + (i * 2.0),
            "is_error": i % 10 == 0,
            "is_anomaly": i == 14 or i == 19,
            "tokens_used": 50 + i,
            "tables_accessed": 2,
            "ast_complexity": 1.5,
        }
        for i in range(20)
    ]
    df = pl.DataFrame(events_data)
    inserted = warehouse.insert_events(df)
    assert inserted == 20

    summary = warehouse.get_star_schema_summary()
    assert summary["fact_query_events"] == 20
    assert summary["dim_tenants"] == 2
    assert summary["dim_schemas"] == 2
    assert summary["dim_users"] == 3

    matrix_df = warehouse.get_tenant_performance_matrix()
    assert len(matrix_df) == 2
    tenants = matrix_df["tenant_id"].to_list()
    assert "tenant_alpha" in tenants
    assert "tenant_beta" in tenants

    alpha_row = matrix_df.filter(pl.col("tenant_id") == "tenant_alpha").to_dicts()[0]
    assert alpha_row["total_queries"] == 15
    assert alpha_row["p50_latency_ms"] > 0
    assert alpha_row["p99_latency_ms"] >= alpha_row["p50_latency_ms"]

    warehouse.close()


def test_insert_empty_events():
    warehouse = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    assert warehouse.insert_events([]) == 0
    assert warehouse.insert_events(pl.DataFrame()) == 0
    warehouse.close()


def test_register_gold_parquet_views(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    kpis_dir = gold_dir / "tenant_kpis"
    kpis_dir.mkdir(parents=True, exist_ok=True)

    df_kpi = pl.DataFrame(
        {
            "tenant_id": ["tenant_x", "tenant_y"],
            "query_count": [100, 200],
            "error_rate": [0.01, 0.02],
        }
    )
    df_kpi.write_parquet(kpis_dir / "kpis.parquet")

    warehouse = DuckDBWarehouse(OLAPConfig(db_path=":memory:", gold_parquet_dir=str(gold_dir)))
    registered = warehouse.register_gold_parquet_views(str(gold_dir))

    assert "gold_tenant_kpis" in registered
    res = warehouse.query("SELECT COUNT(*) as count FROM gold_tenant_kpis;")
    assert res["count"][0] == 2

    warehouse.close()


def test_persistent_db_path(tmp_path: Path):
    db_file = tmp_path / "test_warehouse.duckdb"
    cfg = OLAPConfig(db_path=str(db_file))
    wh1 = DuckDBWarehouse(cfg)
    wh1.execute("INSERT INTO dim_tenants (tenant_id, tier) VALUES ('tenant_perm', 'enterprise');")
    wh1.close()

    assert db_file.exists()
    wh2 = DuckDBWarehouse(cfg)
    res = wh2.query("SELECT COUNT(*) AS count FROM dim_tenants WHERE tenant_id = 'tenant_perm';")
    assert res["count"][0] == 1
    wh2.close()
