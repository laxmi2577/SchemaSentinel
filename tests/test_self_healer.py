import datetime
from pathlib import Path
import pytest

from src.agentic_ops import AgentConfig, SelfHealAttempt, SelfHealStatus, SelfHealingSQLLoop
from src.analytics_engine.olap import DuckDBWarehouse, OLAPConfig
from src.stream_lakehouse.dlq_router import DLQParquetSink, DLQRouter


def _seed_warehouse() -> DuckDBWarehouse:
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    events = [
        {
            "event_id": f"e_{i}",
            "tenant_id": "tenant_heal",
            "schema_id": "schema_crm",
            "user_id": "user_1",
            "latency_ms": 25.0,
            "is_error": False,
            "is_anomaly": False,
            "tokens_used": 50,
        }
        for i in range(10)
    ]
    wh.insert_events(events)
    return wh


def test_parse_error_details():
    healer = SelfHealingSQLLoop()

    dtype, entity = healer.parse_error_details('Binder Error: Referenced column "latency" not found in FROM clause')
    assert dtype == "COLUMN_NOT_FOUND"
    assert entity == "latency"

    dtype, entity = healer.parse_error_details("Catalog Error: Table with name events does not exist!")
    assert dtype == "TABLE_NOT_FOUND"
    assert entity == "events"

    dtype, entity = healer.parse_error_details("Parser Error: syntax error at or near 'FORM'")
    assert dtype == "SYNTAX_ERROR"
    assert entity is None


def test_self_heal_column_rename_success():
    wh = _seed_warehouse()
    healer = SelfHealingSQLLoop(warehouse=wh)

    failing_sql = "SELECT latency, tenant_id FROM fact_query_events WHERE tenant_id = 'tenant_heal';"
    error_msg = 'Binder Error: Referenced column "latency" not found'

    status, healed_sql, attempts = healer.heal_query(
        failing_sql=failing_sql,
        error_message=error_msg,
        tenant_id="tenant_heal",
    )

    assert status == SelfHealStatus.SUCCESS
    assert healed_sql is not None
    assert "latency_ms" in healed_sql
    assert len(attempts) >= 1
    assert attempts[0].execution_success is True

    wh.close()


def test_self_heal_table_rename_success():
    wh = _seed_warehouse()
    healer = SelfHealingSQLLoop(warehouse=wh)

    failing_sql = "SELECT tenant_id FROM events WHERE tenant_id = 'tenant_heal';"
    error_msg = "Table with name events does not exist"

    status, healed_sql, attempts = healer.heal_query(
        failing_sql=failing_sql,
        error_message=error_msg,
        tenant_id="tenant_heal",
    )

    assert status == SelfHealStatus.SUCCESS
    assert healed_sql is not None
    assert "fact_query_events" in healed_sql
    assert len(attempts) >= 1
    assert attempts[0].execution_success is True

    wh.close()


def test_self_heal_exhaustion_escalates_to_dlq(tmp_path: Path):
    wh = _seed_warehouse()
    dlq_dir = tmp_path / "dlq"
    sink = DLQParquetSink(base_dir=str(dlq_dir))
    router = DLQRouter(parquet_sink=sink)
    cfg = AgentConfig(max_self_heal_retries=2)
    healer = SelfHealingSQLLoop(config=cfg, warehouse=wh, dlq_router=router)

    failing_sql = "SELECT unresolvable_bogus_col_xyz FROM completely_unknown_table_abc;"
    error_msg = 'Table with name completely_unknown_table_abc does not exist'

    status, healed_sql, attempts = healer.heal_query(
        failing_sql=failing_sql,
        error_message=error_msg,
        tenant_id="tenant_heal",
    )

    assert status in (SelfHealStatus.SUCCESS, SelfHealStatus.FAILED_ESCALATED)
    assert len(attempts) <= 2
    wh.close()
