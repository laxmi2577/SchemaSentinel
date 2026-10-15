from pathlib import Path
import polars as pl
import pytest
from src.stream_lakehouse import StreamEvent, BronzeWriter, BronzeWriterConfig


def test_bronze_writer_config_defaults():
    cfg = BronzeWriterConfig()
    assert cfg.base_dir == "data/lakehouse/bronze"
    assert cfg.compression == "snappy"


def test_bronze_writer_append_single_event(tmp_path: Path):
    writer = BronzeWriter(BronzeWriterConfig(base_dir=str(tmp_path / "bronze")))
    evt = StreamEvent(
        tenant_id="tenant_alpha",
        user_id="user_1",
        natural_query="SELECT COUNT(*) FROM orders;",
        generated_sql="SELECT COUNT(*) FROM orders;",
        tables_accessed=["orders"],
        columns_accessed=["id"],
        latency_ms=12.5,
        tokens_used=45,
    )
    written_path = writer.append_event(evt)
    assert written_path != ""
    assert Path(written_path).exists()
    assert "tenant_id=tenant_alpha" in written_path
    assert written_path.endswith(".parquet")

    df = pl.read_parquet(written_path)
    assert df.shape[0] == 1
    assert df["tenant_id"][0] == "tenant_alpha"
    assert df["user_id"][0] == "user_1"
    assert df["latency_ms"][0] == 12.5
    assert df["tokens_used"][0] == 45
    assert df["tables_accessed"].to_list()[0] == ["orders"]


def test_bronze_writer_append_batch_multi_tenant(tmp_path: Path):
    writer = BronzeWriter(BronzeWriterConfig(base_dir=str(tmp_path / "bronze")))
    evts = [
        StreamEvent(tenant_id="tenant_1", natural_query="query 1"),
        StreamEvent(tenant_id="tenant_1", natural_query="query 2"),
        StreamEvent(tenant_id="tenant_2", natural_query="query 3"),
    ]
    paths = writer.append_batch(evts)
    assert len(paths) == 2
    assert writer.get_total_events_count() == 3

    partitions = writer.get_partitions()
    tenant_ids = {p["tenant_id"] for p in partitions}
    assert "tenant_1" in tenant_ids
    assert "tenant_2" in tenant_ids


def test_bronze_writer_read_and_scan_raw(tmp_path: Path):
    writer = BronzeWriter(BronzeWriterConfig(base_dir=str(tmp_path / "bronze")))
    evts = [
        StreamEvent(tenant_id="tenant_abc", natural_query="query A"),
        StreamEvent(tenant_id="tenant_xyz", natural_query="query B"),
    ]
    writer.append_batch(evts)

    df = writer.read_raw()
    assert df.shape[0] == 2
    assert "raw_payload_json" in df.columns

    lf = writer.scan_raw()
    collected = lf.collect()
    assert collected.shape[0] == 2

    filtered_df = writer.read_raw(tenant_id="tenant_abc")
    assert filtered_df.shape[0] == 1
    assert filtered_df["tenant_id"][0] == "tenant_abc"


def test_bronze_writer_empty_batch(tmp_path: Path):
    writer = BronzeWriter(BronzeWriterConfig(base_dir=str(tmp_path / "bronze")))
    assert writer.append_batch([]) == []
    assert writer.read_raw().is_empty()
    assert writer.scan_raw().collect().is_empty()
    assert writer.get_total_events_count() == 0
    assert writer.get_partitions() == []
