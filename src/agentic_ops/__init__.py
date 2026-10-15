from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class InvestigationStep(str, Enum):
    SCOPE = "scope"
    PROFILE = "profile"
    DRILL_DOWN = "drill_down"
    CROSS_REFERENCE = "cross_reference"
    REPORT = "report"


class AnomalyType(str, Enum):
    LATENCY_SPIKE = "latency_spike"
    ERROR_BURST = "error_burst"
    CARTESIAN_JOIN = "cartesian_join"
    SCHEMA_DRIFT = "schema_drift"
    CREDENTIAL_STUFFING = "credential_stuffing"
    UNAUTHORIZED_PROBE = "unauthorized_probe"


class SelfHealStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED_ESCALATED = "failed_escalated"


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_investigation_turns: int = 5
    anomaly_trigger_threshold: float = 0.85
    max_self_heal_retries: int = 3
    active_learning_export_path: str = "data/active_learning/dpo_pairs.jsonl"
    dlq_quarantine_dir: str = "data/lakehouse/dlq"
    enable_auto_escalation: bool = True
    dry_run: bool = False


class InvestigationTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_index: int
    step: InvestigationStep
    hypothesis: str
    sql_query: str
    result_summary: Dict[str, Any]
    latency_ms: float
    is_valid_cfg: bool
    is_policy_compliant: bool


class ForensicReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    investigation_id: str
    tenant_id: str
    anomaly_score: float
    anomaly_type: AnomalyType
    root_cause: str
    confidence_score: float
    turns: List[InvestigationTurn]
    recommendations: List[str]
    requires_schema_migration: bool
    generated_at: str


class SelfHealAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_index: int
    original_query: str
    error_message: str
    detected_drift_type: str
    ast_diff_summary: str
    proposed_query: str
    is_valid_syntax: bool
    is_policy_compliant: bool
    execution_success: bool


class ActiveLearningPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair_id: str
    tenant_id: str
    prompt: str
    chosen_query: str
    rejected_query: str
    rejection_reason: str
    estimated_reward_margin: float
    source: str


from src.agentic_ops.forensic_agent import ForensicInvestigationAgent
from src.agentic_ops.self_healer import SelfHealingSQLLoop
from src.agentic_ops.active_learner import ActiveLearningFlywheel

__all__ = [
    "InvestigationStep",
    "AnomalyType",
    "SelfHealStatus",
    "AgentConfig",
    "InvestigationTurn",
    "ForensicReport",
    "SelfHealAttempt",
    "ActiveLearningPair",
    "ForensicInvestigationAgent",
    "SelfHealingSQLLoop",
    "ActiveLearningFlywheel",
]
