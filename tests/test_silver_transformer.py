from pathlib import Path
import polars as pl
import pytest
from src.stream_lakehouse import (
    StreamEvent,
    BronzeWriter,
    BronzeWriterConfig,
    SilverTransformer,
    SilverTransformerConfig,
    TransformationMetrics,
)
from src.data_engine.schema_parser import DatabaseCatalog, TableMetadata, ColumnMetadata


def test_silver_transformer_config_defaults():
    cfg = SilverTransformerConfig()
    assert cfg.base_dir == "data/lakehouse/silver"
    assert cfg.bronze_dir == "data/lakehouse/bronze"
    assert cfg.compression == "snappy"
    assert cfg.dedup_threshold == 0.85
    assert cfg.minhash_num_perm == 128


def test_silver_transformer_null_imputation_and_types():
    transformer = SilverTransformer()
    raw_df = pl.DataFrame(
        {
            "event_id": ["ev_1", "ev_2"],
            "tenant_id": ["tenant_x", "tenant_x"],
            "user_id": [None, "user_42"],
            "timestamp": [1700000000.0, 1700000001.0],
            "event_date": ["2026-10-15", "2026-10-15"],
            "event_type": [None, "query_execution"],
            "natural_query": [None, "SELECT * FROM users;"],
            "generated_sql": [None, "SELECT * FROM users;"],
            "tables_accessed": [["users"], ["users"]],
            "columns_accessed": [["id"], ["name"]],
            "latency_ms": [None, 45.2],
            "tokens_used": [None, 30],
            "is_valid": [None, False],
            "anomaly_score": [None, 0.75],
            "raw_payload_json": [None, "{}"],
        }
    )

    silver_df, metrics = transformer.process_bronze_batch(raw_df)
    assert silver_df.shape[0] == 2
    assert metrics.nulls_imputed > 0
    assert silver_df["user_id"][0] == "unknown_user"
    assert silver_df["latency_ms"][0] == 0.0
    assert silver_df["anomaly_score"][0] == 0.0
    assert silver_df["tokens_used"][0] == 0
    assert silver_df["natural_query"][0] == ""
    assert silver_df["generated_sql"][0] == ""
    assert silver_df["is_valid"][0] is True
    assert silver_df["raw_payload_json"][0] == "{}"


def test_silver_transformer_minhash_deduplication():
    transformer = SilverTransformer(SilverTransformerConfig(dedup_threshold=0.80))
    raw_df = pl.DataFrame(
        {
            "event_id": ["ev_1", "ev_2", "ev_3"],
            "tenant_id": ["tenant_x", "tenant_x", "tenant_x"],
            "user_id": ["user_1", "user_2", "user_3"],
            "timestamp": [1700000000.0, 1700000001.0, 1700000002.0],
            "event_date": ["2026-10-15", "2026-10-15", "2026-10-15"],
            "event_type": ["query_execution", "query_execution", "query_execution"],
            "natural_query": [
                "find all active customer orders from last month",
                "find all active customer orders from last month",
                "calculate total revenue grouped by regional territory",
            ],
            "generated_sql": [
                "SELECT * FROM orders WHERE status = 'active';",
                "SELECT * FROM orders WHERE status = 'active';",
                "SELECT territory, SUM(amount) FROM revenue GROUP BY territory;",
            ],
            "tables_accessed": [["orders"], ["orders"], ["revenue"]],
            "columns_accessed": [["status"], ["status"], ["territory", "amount"]],
            "latency_ms": [10.0, 12.0, 50.0],
            "tokens_used": [20, 22, 60],
            "is_valid": [True, True, True],
            "anomaly_score": [0.01, 0.02, 0.05],
            "raw_payload_json": ["{}", "{}", "{}"],
        }
    )

    silver_df, metrics = transformer.process_bronze_batch(raw_df)
    assert metrics.total_input_records == 3
    assert metrics.duplicates_dropped == 1
    assert metrics.total_silver_records == 2
    assert silver_df.shape[0] == 2


def test_silver_transformer_catalog_validation():
    col_id = ColumnMetadata(name="id", data_type="INTEGER", is_primary_key=True)
    table_users = TableMetadata(name="users", columns={"id": col_id}, primary_keys=["id"])
    catalog = DatabaseCatalog(db_id="prod_db", tables={"users": table_users})

    transformer = SilverTransformer(catalog=catalog)
    raw_df = pl.DataFrame(
        {
            "event_id": ["ev_valid", "ev_invalid"],
            "tenant_id": ["tenant_x", "tenant_x"],
            "user_id": ["user_1", "user_2"],
            "timestamp": [1700000000.0, 1700000001.0],
            "event_date": ["2026-10-15", "2026-10-15"],
            "event_type": ["query_execution", "query_execution"],
            "natural_query": ["select user ids", "select shadow secrets"],
            "generated_sql": ["SELECT id FROM users;", "SELECT secret FROM shadow_table;"],
            "tables_accessed": [["users"], ["shadow_table"]],
            "columns_accessed": [["id"], ["secret"]],
            "latency_ms": [10.0, 20.0],
            "tokens_used": [15, 25],
            "is_valid": [True, True],
            "anomaly_score": [0.01, 0.99],
            "raw_payload_json": ["{}", "{}"],
        }
    )

    silver_df, metrics = transformer.process_bronze_batch(raw_df)
    assert metrics.catalog_violations_dropped == 1
    assert metrics.total_silver_records == 1
    assert silver_df.shape[0] == 1
    assert silver_df["event_id"][0] == "ev_valid"


def test_silver_transformer_write_read_and_e2e_bronze_to_silver(tmp_path: Path):
    bronze_dir = str(tmp_path / "bronze")
    silver_dir = str(tmp_path / "silver")

    bronze_writer = BronzeWriter(BronzeWriterConfig(base_dir=bronze_dir))
    evts = [
        StreamEvent(
            tenant_id="tenant_omega",
            natural_query="SELECT COUNT(*) FROM telemetry;",
            generated_sql="SELECT COUNT(*) FROM telemetry;",
            tables_accessed=["telemetry"],
            columns_accessed=["id"],
            latency_ms=18.0,
            tokens_used=40,
        ),
        StreamEvent(
            tenant_id="tenant_omega",
            natural_query="SELECT AVG(latency) FROM metrics;",
            generated_sql="SELECT AVG(latency) FROM metrics;",
            tables_accessed=["metrics"],
            columns_accessed=["latency"],
            latency_ms=22.0,
            tokens_used=45,
        ),
    ]
    bronze_writer.append_batch(evts)

    transformer = SilverTransformer(
        SilverTransformerConfig(base_dir=silver_dir, bronze_dir=bronze_dir)
    )

    written_paths, metrics = transformer.process_bronze_to_silver(tenant_id="tenant_omega")
    assert len(written_paths) == 1
    assert Path(written_paths[0]).exists()
    assert "tenant_id=tenant_omega" in written_paths[0]
    assert metrics.total_silver_records == 2

    silver_df = transformer.read_silver()
    assert silver_df.shape[0] == 2
    assert transformer.get_total_events_count() == 2

    silver_lf = transformer.scan_silver()
    assert silver_lf.collect().shape[0] == 2

    partitions = transformer.get_partitions()
    assert len(partitions) == 1
    assert partitions[0]["tenant_id"] == "tenant_omega"


def test_silver_transformer_empty_batch(tmp_path: Path):
    transformer = SilverTransformer(
        SilverTransformerConfig(
            base_dir=str(tmp_path / "silver"),
            bronze_dir=str(tmp_path / "bronze"),
        )
    )
    df, metrics = transformer.process_bronze_batch(pl.DataFrame())
    assert df.is_empty()
    assert metrics.total_input_records == 0
    assert metrics.total_silver_records == 0
    assert transformer.write_silver(df) == []
    assert transformer.read_silver().is_empty()
    assert transformer.scan_silver().collect().is_empty()
    assert transformer.get_total_events_count() == 0
    assert transformer.get_partitions() == []
