import json
from pathlib import Path
import pytest

from src.agentic_ops import (
    ActiveLearningFlywheel,
    ActiveLearningPair,
    AgentConfig,
    SelfHealAttempt,
)


def test_active_learning_flywheel_init():
    flywheel = ActiveLearningFlywheel()
    assert flywheel.get_pairs() == []
    summary = flywheel.get_dataset_summary()
    assert summary["total_pairs"] == 0
    assert summary["average_estimated_reward_margin"] == 0.0


def test_mine_from_self_healing():
    flywheel = ActiveLearningFlywheel()

    attempts = [
        SelfHealAttempt(
            attempt_index=1,
            original_query="SELECT latency FROM fact_query_events;",
            error_message="column latency not found",
            detected_drift_type="COLUMN_NOT_FOUND",
            ast_diff_summary="Substituted latency with latency_ms",
            proposed_query="SELECT latency FROM fact_query_events;",
            is_valid_syntax=True,
            is_policy_compliant=True,
            execution_success=False,
        ),
        SelfHealAttempt(
            attempt_index=2,
            original_query="SELECT latency FROM fact_query_events;",
            error_message="",
            detected_drift_type="COLUMN_NOT_FOUND",
            ast_diff_summary="Substituted latency with latency_ms",
            proposed_query="SELECT latency_ms FROM fact_query_events;",
            is_valid_syntax=True,
            is_policy_compliant=True,
            execution_success=True,
        ),
    ]

    mined = flywheel.mine_from_self_healing(attempts, tenant_id="tenant_active")

    assert len(mined) == 1
    pair = mined[0]
    assert isinstance(pair, ActiveLearningPair)
    assert pair.tenant_id == "tenant_active"
    assert "latency_ms" in pair.chosen_query
    assert "latency" in pair.rejected_query
    assert pair.source == "self_healing_loop"
    assert pair.estimated_reward_margin > 1.0


def test_mine_from_dlq():
    flywheel = ActiveLearningFlywheel()

    class MockDLQRecord:
        tenant_id = "tenant_quarantine"
        failing_sql = "DELETE FROM fact_query_events WHERE tenant_id = 'tenant_quarantine';"
        error_class = "AST_VALIDATION_ERROR_MUTATION"

    records = [MockDLQRecord()]
    mined = flywheel.mine_from_dlq(records)

    assert len(mined) == 1
    pair = mined[0]
    assert pair.tenant_id == "tenant_quarantine"
    assert "DELETE" in pair.rejected_query
    assert "SELECT" in pair.chosen_query
    assert pair.rejection_reason == "AST_VALIDATION_ERROR_MUTATION"
    assert pair.source == "dlq_quarantine"


def test_export_dataset_jsonl(tmp_path: Path):
    target_jsonl = tmp_path / "active_learning" / "dpo_pairs.jsonl"
    flywheel = ActiveLearningFlywheel(export_path=str(target_jsonl))

    p1 = ActiveLearningPair(
        pair_id="pair_1",
        tenant_id="t1",
        prompt="Count queries",
        chosen_query="SELECT COUNT(*) FROM fact_query_events;",
        rejected_query="SELECT COUNT(*) FROM non_existent;",
        rejection_reason="TABLE_NOT_FOUND",
        estimated_reward_margin=1.4,
        source="unit_test",
    )
    flywheel.add_pair(p1)

    exported_path, count = flywheel.export_dataset(str(target_jsonl))

    assert count == 1
    assert Path(exported_path).exists()

    with open(exported_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["pair_id"] == "pair_1"
    assert record["chosen"] == "SELECT COUNT(*) FROM fact_query_events;"
    assert record["metadata"]["source"] == "unit_test"

    summary = flywheel.get_dataset_summary()
    assert summary["total_pairs"] == 1
    assert summary["average_estimated_reward_margin"] == 1.4
    assert summary["rejection_breakdown"]["TABLE_NOT_FOUND"] == 1
