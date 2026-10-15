import datetime
import polars as pl
import pytest

from src.agentic_ops import (
    AgentConfig,
    AnomalyType,
    ForensicInvestigationAgent,
    ForensicReport,
    InvestigationStep,
    InvestigationTurn,
)
from src.analytics_engine.olap import DuckDBWarehouse, OLAPConfig


def _seed_warehouse() -> DuckDBWarehouse:
    wh = DuckDBWarehouse(OLAPConfig(db_path=":memory:"))
    now = datetime.datetime.now()
    events = [
        {
            "event_id": f"evt_{i}",
            "tenant_id": "tenant_incident",
            "schema_id": "schema_billing",
            "user_id": f"user_{i % 3}",
            "timestamp": now - datetime.timedelta(minutes=i * 2),
            "latency_ms": 180.0 if i % 2 == 0 else 45.0,
            "is_error": i % 6 == 0,
            "is_anomaly": i % 3 == 0,
            "tokens_used": 150 if i % 2 == 0 else 30,
            "tables_accessed": 5 if i % 2 == 0 else 1,
            "ast_complexity": 8.5 if i % 2 == 0 else 1.2,
        }
        for i in range(40)
    ]
    wh.insert_events(events)
    return wh


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_investigation_turns == 5
    assert cfg.anomaly_trigger_threshold == 0.85
    assert cfg.max_self_heal_retries == 3
    assert cfg.enable_auto_escalation is True


def test_forensic_agent_full_investigation_lifecycle():
    wh = _seed_warehouse()
    agent = ForensicInvestigationAgent(warehouse=wh)

    report = agent.investigate(tenant_id="tenant_incident", anomaly_score=0.94)

    assert isinstance(report, ForensicReport)
    assert report.tenant_id == "tenant_incident"
    assert report.anomaly_score == 0.94
    assert len(report.turns) == 5

    step_sequence = [t.step for t in report.turns]
    assert step_sequence == [
        InvestigationStep.SCOPE,
        InvestigationStep.PROFILE,
        InvestigationStep.DRILL_DOWN,
        InvestigationStep.CROSS_REFERENCE,
        InvestigationStep.REPORT,
    ]

    for turn in report.turns:
        assert isinstance(turn, InvestigationTurn)
        assert turn.is_valid_cfg is True
        assert turn.is_policy_compliant is True
        assert turn.result_summary.get("status") == "success"

    assert len(report.recommendations) > 0
    assert report.confidence_score >= 0.80

    wh.close()


def test_forensic_agent_diagnoses_cartesian_join():
    wh = _seed_warehouse()
    agent = ForensicInvestigationAgent(warehouse=wh)

    report = agent.investigate(
        tenant_id="tenant_incident",
        anomaly_score=0.99,
        anomaly_type_hint=AnomalyType.CARTESIAN_JOIN,
    )

    assert report.anomaly_type == AnomalyType.CARTESIAN_JOIN
    assert "Cartesian" in report.root_cause or "complexity" in report.root_cause.lower()
    wh.close()


def test_forensic_agent_invalid_sql_guard():
    wh = _seed_warehouse()
    agent = ForensicInvestigationAgent(warehouse=wh)

    turn = agent.execute_turn(
        turn_index=99,
        step=InvestigationStep.SCOPE,
        tenant_id="tenant_incident",
        hypothesis="Test illegal query guard",
        sql_query="DROP TABLE fact_query_events;",
    )

    assert turn.is_policy_compliant is False or turn.is_valid_cfg is False
    assert turn.result_summary["status"] == "validation_failure"
    wh.close()
