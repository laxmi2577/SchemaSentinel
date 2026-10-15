from __future__ import annotations

import collections
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import polars as pl

from src.ml_engine import FeatureStoreConfig, FeatureVector
from src.stream_lakehouse import StreamEvent


class StreamingFeatureStore:
    def __init__(self, config: Optional[FeatureStoreConfig] = None) -> None:
        self.config = config or FeatureStoreConfig()
        self._events_by_tenant: Dict[str, collections.deque[Dict[str, Any]]] = {}
        self._baseline_by_tenant: Dict[str, Dict[str, Any]] = {}
        self.offline_dir = Path(self.config.offline_storage_dir)
        self.offline_dir.mkdir(parents=True, exist_ok=True)

    def _event_to_dict(self, event: StreamEvent | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(event, StreamEvent):
            return {
                "event_id": event.event_id,
                "tenant_id": event.tenant_id,
                "timestamp": float(event.timestamp),
                "latency_ms": float(event.latency_ms),
                "tokens_used": int(event.tokens_used),
                "is_valid": bool(event.is_valid),
                "tables_accessed": list(event.tables_accessed),
                "columns_accessed": list(event.columns_accessed),
            }
        return {
            "event_id": str(event.get("event_id", uuid.uuid4())),
            "tenant_id": str(event.get("tenant_id", "default_tenant")),
            "timestamp": float(event.get("timestamp", time.time())),
            "latency_ms": float(event.get("latency_ms", 0.0)),
            "tokens_used": int(event.get("tokens_used", 0)),
            "is_valid": bool(event.get("is_valid", True)),
            "tables_accessed": list(event.get("tables_accessed") or []),
            "columns_accessed": list(event.get("columns_accessed") or []),
        }

    def ingest_event(self, event: StreamEvent | Dict[str, Any]) -> None:
        record = self._event_to_dict(event)
        tid = record["tenant_id"]

        if tid not in self._events_by_tenant:
            self._events_by_tenant[tid] = collections.deque(maxlen=self.config.max_online_cache_size)
            self._baseline_by_tenant[tid] = {
                "latency_count": 0,
                "latency_sum": 0.0,
                "latency_sq_sum": 0.0,
                "table_count_sum": 0,
            }

        self._events_by_tenant[tid].append(record)

        base = self._baseline_by_tenant[tid]
        lat = record["latency_ms"]
        base["latency_count"] += 1
        base["latency_sum"] += lat
        base["latency_sq_sum"] += lat * lat
        base["table_count_sum"] += len(record["tables_accessed"])

    def ingest_batch(self, events: List[StreamEvent | Dict[str, Any]]) -> int:
        for ev in events:
            self.ingest_event(ev)
        return len(events)

    def compute_features(
        self,
        tenant_id: str,
        window_seconds: int = 60,
        current_time: Optional[float] = None,
    ) -> FeatureVector:
        now = current_time if current_time is not None else time.time()
        window_start = now - float(window_seconds)

        tenant_events = self._events_by_tenant.get(tenant_id)
        if not tenant_events:
            return FeatureVector(
                tenant_id=tenant_id,
                timestamp=now,
                window_size_seconds=window_seconds,
            )

        window_records = [e for e in tenant_events if window_start <= e["timestamp"] <= now]
        if not window_records:
            return FeatureVector(
                tenant_id=tenant_id,
                timestamp=now,
                window_size_seconds=window_seconds,
            )

        count = len(window_records)
        effective_window = max(1.0, float(window_seconds))
        query_velocity = count / effective_window

        latencies = [e["latency_ms"] for e in window_records]
        mean_latency = sum(latencies) / count

        base = self._baseline_by_tenant.get(tenant_id, {})
        base_count = base.get("latency_count", 0)
        if base_count > 1:
            base_mean = base["latency_sum"] / base_count
            variance = max(0.0, (base["latency_sq_sum"] / base_count) - (base_mean * base_mean))
            base_std = math.sqrt(variance)
            z_score = (mean_latency - base_mean) / (base_std + 1e-5)
            base_avg_tables = base["table_count_sum"] / base_count
        else:
            z_score = 0.0
            base_avg_tables = 0.0

        errors = sum(1 for e in window_records if not e["is_valid"])
        error_rate = errors / count

        tables: Set[str] = set()
        for e in window_records:
            tables.update(e["tables_accessed"])
        distinct_tables = len(tables)

        complexity_delta = float(distinct_tables) - base_avg_tables

        total_tokens = sum(e["tokens_used"] for e in window_records)
        token_rate = total_tokens / effective_window

        return FeatureVector(
            tenant_id=tenant_id,
            timestamp=now,
            window_size_seconds=window_seconds,
            query_velocity=round(query_velocity, 4),
            mean_latency_ms=round(mean_latency, 2),
            latency_z_score=round(z_score, 4),
            error_rate=round(error_rate, 4),
            schema_complexity_delta=round(complexity_delta, 2),
            distinct_tables_count=distinct_tables,
            token_consumption_rate=round(token_rate, 2),
        )

    def compute_multi_window_features(
        self,
        tenant_id: str,
        current_time: Optional[float] = None,
    ) -> Dict[int, FeatureVector]:
        result: Dict[int, FeatureVector] = {}
        for window in self.config.windows_seconds:
            result[window] = self.compute_features(tenant_id, window, current_time)
        return result

    def to_numpy_matrix(self, feature_vectors: List[FeatureVector]) -> np.ndarray:
        if not feature_vectors:
            return np.empty((0, 7), dtype=np.float32)
        arrays = [v.to_array() for v in feature_vectors]
        return np.array(arrays, dtype=np.float32)

    def persist_offline_features(
        self,
        feature_vectors: List[FeatureVector],
        date_str: str,
    ) -> List[str]:
        if not feature_vectors:
            return []

        partition_groups: Dict[str, List[Dict[str, Any]]] = {}
        for fv in feature_vectors:
            tid = fv.tenant_id
            if tid not in partition_groups:
                partition_groups[tid] = []
            row = fv.to_feature_dict()
            row["tenant_id"] = tid
            row["timestamp"] = fv.timestamp
            row["window_size_seconds"] = fv.window_size_seconds
            partition_groups[tid].append(row)

        written_paths: List[str] = []
        for tid, rows in partition_groups.items():
            target_dir = self.offline_dir / f"tenant_id={tid}" / f"date={date_str}"
            target_dir.mkdir(parents=True, exist_ok=True)
            batch_id = str(uuid.uuid4())[:8]
            target_file = target_dir / f"features_{batch_id}.parquet"

            df = pl.DataFrame(rows)
            df.write_parquet(target_file, compression="snappy")
            written_paths.append(str(target_file))

        return written_paths

    def read_offline_features(
        self,
        tenant_id: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> pl.DataFrame:
        if not self.offline_dir.exists():
            return pl.DataFrame()

        search_root = self.offline_dir
        if tenant_id and date_str:
            search_root = self.offline_dir / f"tenant_id={tenant_id}" / f"date={date_str}"
        elif tenant_id:
            search_root = self.offline_dir / f"tenant_id={tenant_id}"

        if not search_root.exists():
            return pl.DataFrame()

        files = list(search_root.rglob("*.parquet"))
        if date_str and not tenant_id:
            files = [p for p in files if f"date={date_str}" in str(p)]

        if not files:
            return pl.DataFrame()

        dfs = [pl.read_parquet(p) for p in files]
        return pl.concat(dfs)
