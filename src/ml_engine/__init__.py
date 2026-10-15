from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FeatureVector(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    timestamp: float = Field(default_factory=time.time)
    window_size_seconds: int = 60
    query_velocity: float = 0.0
    mean_latency_ms: float = 0.0
    latency_z_score: float = 0.0
    error_rate: float = 0.0
    schema_complexity_delta: float = 0.0
    distinct_tables_count: int = 0
    token_consumption_rate: float = 0.0

    def to_array(self) -> List[float]:
        return [
            float(self.query_velocity),
            float(self.mean_latency_ms),
            float(self.latency_z_score),
            float(self.error_rate),
            float(self.schema_complexity_delta),
            float(self.distinct_tables_count),
            float(self.token_consumption_rate),
        ]

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "query_velocity": float(self.query_velocity),
            "mean_latency_ms": float(self.mean_latency_ms),
            "latency_z_score": float(self.latency_z_score),
            "error_rate": float(self.error_rate),
            "schema_complexity_delta": float(self.schema_complexity_delta),
            "distinct_tables_count": float(self.distinct_tables_count),
            "token_consumption_rate": float(self.token_consumption_rate),
        }


class AnomalyScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    score_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    timestamp: float = Field(default_factory=time.time)
    anomaly_score: float
    is_anomaly: bool
    threshold_used: float = 0.50
    inference_latency_ms: float = 0.0
    contributing_features: Dict[str, float] = Field(default_factory=dict)


class ScorerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: str = "models/anomaly_detector.onnx"
    anomaly_threshold: float = 0.50
    calibrated_threshold: float = 0.65
    batch_size: int = 64
    execution_provider: str = "CPUExecutionProvider"
    intra_op_num_threads: int = 1


class FeatureStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    windows_seconds: List[int] = Field(default_factory=lambda: [60, 300, 900, 3600, 86400])
    offline_storage_dir: str = "data/feature_store"
    max_online_cache_size: int = 10000
    baseline_history_size: int = 500


from src.ml_engine.feature_store import (
    StreamingFeatureStore,
)
from src.ml_engine.onnx_scorer import (
    LocalONNXScorer,
)

__all__ = [
    "FeatureVector",
    "AnomalyScore",
    "ScorerConfig",
    "FeatureStoreConfig",
    "StreamingFeatureStore",
    "LocalONNXScorer",
]
