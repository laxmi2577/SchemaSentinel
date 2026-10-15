from __future__ import annotations

import asyncio
import pytest
from pydantic import ValidationError

import src.stream_lakehouse as slh


def test_stream_event_pydantic_contract():
    event = slh.StreamEvent(
        tenant_id="tenant_alpha",
        user_id="user_123",
        natural_query="Select all orders",
        generated_sql="SELECT * FROM orders WHERE tenant_id = 'tenant_alpha';",
        tables_accessed=["orders"],
        columns_accessed=["id", "amount"],
        latency_ms=12.4,
        tokens_used=45,
        is_valid=True,
    )
    assert event.tenant_id == "tenant_alpha"
    assert event.user_id == "user_123"
    assert event.event_type == "query_execution"
    assert event.is_valid is True
    assert event.tables_accessed == ["orders"]
    assert len(event.event_id) > 0
    assert event.timestamp > 0

    with pytest.raises(ValidationError):
        event.tenant_id = "tenant_beta"


def test_dlq_record_contract():
    event = slh.StreamEvent(
        tenant_id="tenant_delta",
        generated_sql="DROP TABLE users;",
        is_valid=False,
    )
    dlq = slh.DLQRecord(
        original_event=event,
        failure_reason="AST policy violation: destructive DDL forbidden",
        error_category="AST_VALIDATION_ERROR",
        retry_count=1,
        diagnostics={"ast_rule": "READ_ONLY_ENFORCEMENT"},
    )
    assert dlq.original_event.tenant_id == "tenant_delta"
    assert dlq.error_category == "AST_VALIDATION_ERROR"
    assert dlq.retry_count == 1
    assert dlq.diagnostics["ast_rule"] == "READ_ONLY_ENFORCEMENT"

    with pytest.raises(ValidationError):
        dlq.retry_count = 2


def test_schema_change_event_contract():
    change = slh.SchemaChangeEvent(
        tenant_id="tenant_gamma",
        table_name="transactions",
        change_type="ADD_COLUMN",
        raw_ddl="ALTER TABLE transactions ADD COLUMN risk_score FLOAT;",
        previous_schema_version="1.0",
        new_schema_version="1.1",
        is_breaking=False,
        ast_diff_summary={"added": ["risk_score"]},
    )
    assert change.tenant_id == "tenant_gamma"
    assert change.table_name == "transactions"
    assert change.change_type == "ADD_COLUMN"
    assert change.is_breaking is False
    assert "risk_score" in change.ast_diff_summary["added"]

    with pytest.raises(ValidationError):
        change.is_breaking = True


def test_stream_config_defaults():
    cfg = slh.StreamConfig()
    assert cfg.buffer_maxsize == 1000
    assert cfg.stall_timeout_seconds == 2.0
    assert cfg.high_watermark_ratio == 0.8
    assert cfg.low_watermark_ratio == 0.2
    assert cfg.num_tenants == 10
    assert cfg.tenant_id_prefix == "tenant_"


def test_queue_metrics_computation():
    metrics = slh.QueueMetrics(
        current_size=80,
        maxsize=100,
        fill_ratio=0.8,
        high_watermark=95,
        total_enqueued=120,
        total_dequeued=40,
        total_stall_events=2,
        total_stall_duration_seconds=0.15,
        is_high_watermark_tripped=True,
        tenant_distribution={"tenant_0": 50, "tenant_1": 70},
    )
    assert metrics.current_size == 80
    assert metrics.fill_ratio == 0.8
    assert metrics.is_high_watermark_tripped is True
    assert metrics.high_watermark == 95


def test_backpressure_buffer_put_and_get():
    async def run():
        cfg = slh.StreamConfig(buffer_maxsize=5)
        buf = slh.BackpressureBuffer(cfg)
        assert buf.is_empty() is True
        assert buf.is_full() is False

        e1 = slh.StreamEvent(tenant_id="t1", natural_query="q1")
        e2 = slh.StreamEvent(tenant_id="t2", natural_query="q2")

        ok1 = await buf.put(e1)
        ok2 = await buf.put(e2)
        assert ok1 is True
        assert ok2 is True
        assert buf.qsize() == 2

        out1 = await buf.get()
        out2 = await buf.get()
        assert out1.tenant_id == "t1"
        assert out2.tenant_id == "t2"
        assert buf.is_empty() is True

    asyncio.run(run())


def test_backpressure_buffer_blocking_stall_telemetry():
    async def run():
        cfg = slh.StreamConfig(buffer_maxsize=2, stall_timeout_seconds=0.05)
        buf = slh.BackpressureBuffer(cfg)

        e1 = slh.StreamEvent(tenant_id="t1")
        e2 = slh.StreamEvent(tenant_id="t2")
        e3 = slh.StreamEvent(tenant_id="t3")

        assert await buf.put(e1) is True
        assert await buf.put(e2) is True
        assert buf.is_full() is True

        ok3 = await buf.put(e3, timeout=0.05)
        assert ok3 is False

        metrics = buf.get_metrics()
        assert metrics.total_stall_events == 1
        assert metrics.total_stall_duration_seconds > 0.0
        assert metrics.total_enqueued == 2
        assert metrics.current_size == 2

    asyncio.run(run())


def test_backpressure_buffer_high_watermark_trip():
    async def run():
        cfg = slh.StreamConfig(buffer_maxsize=10, high_watermark_ratio=0.8)
        buf = slh.BackpressureBuffer(cfg)

        for i in range(7):
            await buf.put(slh.StreamEvent(tenant_id=f"t_{i}"))

        m_under = buf.get_metrics()
        assert m_under.fill_ratio == 0.7
        assert m_under.is_high_watermark_tripped is False

        await buf.put(slh.StreamEvent(tenant_id="t_7"))
        m_tripped = buf.get_metrics()
        assert m_tripped.fill_ratio == 0.8
        assert m_tripped.is_high_watermark_tripped is True
        assert m_tripped.high_watermark == 8

        await buf.get()
        await buf.get()
        m_after = buf.get_metrics()
        assert m_after.current_size == 6
        assert m_after.high_watermark == 8

    asyncio.run(run())


def test_stream_producer_generate_event():
    cfg = slh.StreamConfig(num_tenants=5, seed=123)
    prod = slh.StreamProducer(cfg)

    evt_valid = prod.generate_event()
    assert evt_valid.tenant_id.startswith("tenant_")
    assert evt_valid.is_valid is True
    assert "SELECT" in evt_valid.generated_sql
    assert evt_valid.anomaly_score is not None
    assert evt_valid.anomaly_score < 0.5

    evt_invalid = prod.generate_event(tenant_id="custom_tenant", is_valid=False)
    assert evt_invalid.tenant_id == "custom_tenant"
    assert evt_invalid.is_valid is False
    assert evt_invalid.anomaly_score == 0.95


def test_stream_producer_emit_batch():
    async def run():
        cfg = slh.StreamConfig(buffer_maxsize=10)
        prod = slh.StreamProducer(cfg)

        events = [prod.generate_event() for _ in range(5)]
        count = await prod.emit_batch(events)
        assert count == 5
        assert prod.buffer.qsize() == 5

    asyncio.run(run())


def test_stream_producer_tenant_partitioned_emission():
    async def run():
        cfg = slh.StreamConfig(buffer_maxsize=50, num_tenants=4, seed=99)
        prod = slh.StreamProducer(cfg)

        for _ in range(20):
            await prod.emit_event()

        dist = prod.get_tenant_distribution()
        assert len(dist) > 1
        assert sum(dist.values()) == 20

    asyncio.run(run())


def test_stream_producer_continuous_emission():
    async def run():
        cfg = slh.StreamConfig(buffer_maxsize=20, seed=77)
        prod = slh.StreamProducer(cfg)

        emitted = []
        async for evt in prod.produce_continuous(max_events=4, interval_seconds=0.001):
            emitted.append(evt)

        assert len(emitted) == 4
        assert prod.buffer.qsize() == 4
        assert all(isinstance(e, slh.StreamEvent) for e in emitted)

    asyncio.run(run())
