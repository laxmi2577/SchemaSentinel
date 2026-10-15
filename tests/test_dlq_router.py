from __future__ import annotations

from pathlib import Path
import pytest

import src.stream_lakehouse as slh


class DummyCatalog:
    def __init__(self, tables: list[str]) -> None:
        self.tables = set(tables)

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables


def test_classify_failure_valid_event():
    event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="SELECT id, amount FROM orders WHERE tenant_id = 'tenant_01';",
        tables_accessed=["orders"],
        is_valid=True,
        anomaly_score=0.12,
    )
    failed, reason, category = slh.classify_failure(event)
    assert failed is False
    assert reason is None
    assert category is None


def test_classify_failure_missing_tenant():
    event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="SELECT id, amount FROM orders WHERE amount > 100;",
        is_valid=True,
    )
    failed, reason, category = slh.classify_failure(event)
    assert failed is True
    assert category == "TENANT_ISOLATION_VIOLATION"
    assert "isolation predicate missing" in reason


def test_classify_failure_destructive_ast():
    event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="DROP TABLE orders;",
        is_valid=True,
    )
    failed, reason, category = slh.classify_failure(event)
    assert failed is True
    assert category == "AST_VALIDATION_ERROR"
    assert "Destructive" in reason


def test_classify_failure_syntax_error():
    event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="SELECT SELECT FROM WHERE WHERE",
        is_valid=True,
    )
    failed, reason, category = slh.classify_failure(event)
    assert failed is True
    assert category == "AST_VALIDATION_ERROR"
    assert "parsing failure" in reason


def test_classify_failure_anomaly_score_breach():
    event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="SELECT id FROM orders WHERE tenant_id = 'tenant_01';",
        anomaly_score=0.98,
        is_valid=True,
    )
    failed, reason, category = slh.classify_failure(event)
    assert failed is True
    assert category == "ANOMALY_THRESHOLD_BREACH"
    assert "exceeded threshold" in reason


def test_classify_failure_catalog_mismatch():
    catalog = DummyCatalog(tables=["orders", "users"])
    event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="SELECT * FROM secret_credentials WHERE tenant_id = 'tenant_01';",
        tables_accessed=["secret_credentials"],
        is_valid=True,
    )
    failed, reason, category = slh.classify_failure(event, catalog=catalog)
    assert failed is True
    assert category == "SCHEMA_MISMATCH"
    assert "absent from catalog" in reason


def test_dlq_router_route_and_retrieve():
    router = slh.DLQRouter()
    event = slh.StreamEvent(tenant_id="tenant_42", is_valid=False)

    record = router.route(
        event=event,
        failure_reason="Malformed syntax",
        error_category="AST_VALIDATION_ERROR",
        diagnostics={"parser": "sqlglot"},
    )
    assert record.original_event.tenant_id == "tenant_42"
    assert record.error_category == "AST_VALIDATION_ERROR"
    assert router.get_quarantined_count() == 1

    by_cat = router.get_records(category="AST_VALIDATION_ERROR")
    assert len(by_cat) == 1

    by_tid = router.get_records(tenant_id="tenant_42")
    assert len(by_tid) == 1

    by_other = router.get_records(tenant_id="tenant_99")
    assert len(by_other) == 0


def test_dlq_router_process_and_route():
    router = slh.DLQRouter()
    bad_event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="DELETE FROM users WHERE tenant_id = 'tenant_01';",
    )
    record = router.process_and_route(bad_event)
    assert record is not None
    assert record.error_category == "AST_VALIDATION_ERROR"
    assert router.get_quarantined_count() == 1

    good_event = slh.StreamEvent(
        tenant_id="tenant_01",
        generated_sql="SELECT id FROM users WHERE tenant_id = 'tenant_01';",
    )
    assert router.process_and_route(good_event) is None
    assert router.get_quarantined_count() == 1


def test_dlq_router_replay_within_threshold():
    cfg = slh.StreamConfig(dlq_threshold_retries=2)
    router = slh.DLQRouter(config=cfg)
    event = slh.StreamEvent(tenant_id="t1", is_valid=False)
    rec = router.route(event, "Timeout", "TIMEOUT")

    replayed1 = router.replay(rec.dlq_id)
    assert replayed1 is not None
    assert replayed1.tenant_id == "t1"

    replayed2 = router.replay(rec.dlq_id)
    assert replayed2 is not None

    replayed3 = router.replay(rec.dlq_id)
    assert replayed3 is None


def test_dlq_parquet_sink_write_and_read(tmp_path: Path):
    sink = slh.DLQParquetSink(base_dir=str(tmp_path / "quarantine"))
    event = slh.StreamEvent(
        tenant_id="tenant_prod",
        natural_query="Select orders",
        generated_sql="SELECT * FROM orders WHERE tenant_id = 'tenant_prod';",
        anomaly_score=0.92,
    )
    record = slh.DLQRecord(
        original_event=event,
        failure_reason="Anomaly breach",
        error_category="ANOMALY_THRESHOLD_BREACH",
        diagnostics={"detector": "IsolationForest"},
    )

    written_path = sink.write_record(record)
    assert Path(written_path).exists()
    assert written_path.endswith(".parquet")

    records = sink.read_records(tenant_id="tenant_prod", category="ANOMALY_THRESHOLD_BREACH")
    assert len(records) == 1
    assert records[0]["tenant_id"] == "tenant_prod"
    assert records[0]["error_category"] == "ANOMALY_THRESHOLD_BREACH"
    assert records[0]["anomaly_score"] == 0.92
