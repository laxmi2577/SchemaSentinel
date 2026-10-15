from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    event_type: str = "query_execution"
    natural_query: Optional[str] = None
    generated_sql: Optional[str] = None
    tables_accessed: List[str] = Field(default_factory=list)
    columns_accessed: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: int = 0
    is_valid: bool = True
    anomaly_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DLQRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    dlq_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_event: StreamEvent
    failure_reason: str
    error_category: str
    timestamp: float = Field(default_factory=time.time)
    retry_count: int = 0
    stack_trace: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class SchemaChangeEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    table_name: str
    change_type: str
    timestamp: float = Field(default_factory=time.time)
    raw_ddl: str
    previous_schema_version: Optional[str] = None
    new_schema_version: str = "1.0"
    ast_diff_summary: Dict[str, Any] = Field(default_factory=dict)
    is_breaking: bool = False


class StreamConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    buffer_maxsize: int = 1000
    stall_timeout_seconds: float = 2.0
    high_watermark_ratio: float = 0.8
    low_watermark_ratio: float = 0.2
    emission_rate_per_sec: float = 100.0
    num_tenants: int = 10
    tenant_id_prefix: str = "tenant_"
    seed: int = 42
    dlq_threshold_retries: int = 3


class QueueMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_size: int
    maxsize: int
    fill_ratio: float
    high_watermark: int
    total_enqueued: int
    total_dequeued: int
    total_stall_events: int
    total_stall_duration_seconds: float
    is_high_watermark_tripped: bool
    tenant_distribution: Dict[str, int] = Field(default_factory=dict)


from src.stream_lakehouse.stream_producer import (
    BackpressureBuffer,
    StreamProducer,
)
from src.stream_lakehouse.dlq_router import (
    DLQParquetSink,
    DLQRouter,
    classify_failure,
)
from src.stream_lakehouse.schema_evolution import (
    DDLDiff,
    SchemaEvolutionDetector,
)
from src.stream_lakehouse.bronze_writer import (
    BronzeWriter,
    BronzeWriterConfig,
)
from src.stream_lakehouse.silver_transformer import (
    SilverTransformer,
    SilverTransformerConfig,
    TransformationMetrics,
)
from src.stream_lakehouse.gold_aggregator import (
    GoldAggregator,
    GoldAggregatorConfig,
    GoldAggregationSummary,
)
from src.stream_lakehouse.catalog_sync import (
    DuckDBCatalogSync,
    CatalogSyncConfig,
    TableSyncRegistration,
    CatalogSyncReport,
)

__all__ = [
    "StreamEvent",
    "DLQRecord",
    "SchemaChangeEvent",
    "StreamConfig",
    "QueueMetrics",
    "BackpressureBuffer",
    "StreamProducer",
    "DLQRouter",
    "DLQParquetSink",
    "classify_failure",
    "DDLDiff",
    "SchemaEvolutionDetector",
    "BronzeWriter",
    "BronzeWriterConfig",
    "SilverTransformer",
    "SilverTransformerConfig",
    "TransformationMetrics",
    "GoldAggregator",
    "GoldAggregatorConfig",
    "GoldAggregationSummary",
    "DuckDBCatalogSync",
    "CatalogSyncConfig",
    "TableSyncRegistration",
    "CatalogSyncReport",
]
