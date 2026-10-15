from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HypothesisTestType(str, Enum):
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARED = "chi_squared"


class EffectSizeType(str, Enum):
    COHENS_D = "cohens_d"
    RANK_BISERIAL_R = "rank_biserial_r"
    CRAMERS_V = "cramers_v"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class CohortTier(str, Enum):
    ENTERPRISE = "enterprise"
    GROWTH = "growth"
    STARTER = "starter"
    AT_RISK = "at_risk"


class OLAPConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    db_path: str = ":memory:"
    gold_parquet_dir: str = "data/lakehouse/gold"
    memory_limit: str = "250MB"
    threads: int = 2
    enable_hive_partitioning: bool = True
    read_only: bool = False


class HypothesisTestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_type: HypothesisTestType
    statistic: float
    p_value: float
    p_value_adjusted: float
    is_significant: bool
    alpha: float = 0.05
    effect_size_type: EffectSizeType
    effect_size: float
    effect_magnitude: str
    sample_size_a: int
    sample_size_b: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KPICohort(BaseModel):
    model_config = ConfigDict(frozen=True)

    cohort_id: str
    tenant_id: str
    cohort_tier: CohortTier
    query_count: int
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    error_rate: float
    anomaly_rate: float
    tokens_consumed: int
    period_start: str
    period_end: str


class TrendMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_name: str
    direction: TrendDirection
    percentage_change: float
    current_value: float
    baseline_value: float
    is_significant: bool
    p_value: float


class TrendReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    period_label: str
    start_date: str
    end_date: str
    metrics: Dict[str, TrendMetric]
    overall_health_score: float
    anomalous_drift_detected: bool


from src.analytics_engine.olap import DuckDBWarehouse
from src.analytics_engine.hypothesis import StatisticalHypothesisEngine
from src.analytics_engine.kpi_aggregator import KPICohortAggregator
from src.analytics_engine.dashboard import (
    DashboardState,
    NLQueryResult,
    StreamlitDashboard,
    compute_kpi_tiles,
    create_anomaly_breakdown_chart,
    create_latency_chart,
    create_tenant_activity_heatmap,
    execute_nl_sql_pipeline,
    run_statistical_comparison,
)

__all__ = [
    "HypothesisTestType",
    "EffectSizeType",
    "TrendDirection",
    "CohortTier",
    "OLAPConfig",
    "HypothesisTestResult",
    "KPICohort",
    "TrendMetric",
    "TrendReport",
    "DuckDBWarehouse",
    "StatisticalHypothesisEngine",
    "KPICohortAggregator",
    "DashboardState",
    "NLQueryResult",
    "StreamlitDashboard",
    "compute_kpi_tiles",
    "create_latency_chart",
    "create_anomaly_breakdown_chart",
    "create_tenant_activity_heatmap",
    "execute_nl_sql_pipeline",
    "run_statistical_comparison",
]
