from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

import src.stream_lakehouse as slh


class BronzeWriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_dir: str = "data/lakehouse/bronze"
    compression: str = "snappy"


class BronzeWriter:
    def __init__(self, config: Optional[BronzeWriterConfig] = None) -> None:
        self.config = config or BronzeWriterConfig()
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _event_to_dict(self, event: slh.StreamEvent) -> Dict[str, Any]:
        dt = datetime.fromtimestamp(event.timestamp, timezone.utc)
        event_date = dt.strftime("%Y-%m-%d")
        return {
            "event_id": event.event_id,
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "timestamp": float(event.timestamp),
            "event_date": event_date,
            "event_type": event.event_type,
            "natural_query": event.natural_query,
            "generated_sql": event.generated_sql,
            "tables_accessed": event.tables_accessed,
            "columns_accessed": event.columns_accessed,
            "latency_ms": float(event.latency_ms),
            "tokens_used": int(event.tokens_used),
            "is_valid": bool(event.is_valid),
            "anomaly_score": float(event.anomaly_score) if event.anomaly_score is not None else None,
            "raw_payload_json": json.dumps(event.metadata),
        }

    def append_event(self, event: slh.StreamEvent) -> str:
        paths = self.append_batch([event])
        return paths[0] if paths else ""

    def append_batch(self, events: List[slh.StreamEvent]) -> List[str]:
        if not events:
            return []

        partition_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for event in events:
            record = self._event_to_dict(event)
            key = (record["tenant_id"], record["event_date"])
            if key not in partition_groups:
                partition_groups[key] = []
            partition_groups[key].append(record)

        written_paths: List[str] = []
        for (tenant_id, event_date), records in partition_groups.items():
            target_dir = self.base_dir / f"tenant_id={tenant_id}" / f"event_date={event_date}"
            target_dir.mkdir(parents=True, exist_ok=True)

            batch_id = str(uuid.uuid4())[:8]
            file_name = f"bronze_{batch_id}.parquet"
            target_file = target_dir / file_name

            schema = {
                "event_id": pl.Utf8,
                "tenant_id": pl.Utf8,
                "user_id": pl.Utf8,
                "timestamp": pl.Float64,
                "event_date": pl.Utf8,
                "event_type": pl.Utf8,
                "natural_query": pl.Utf8,
                "generated_sql": pl.Utf8,
                "tables_accessed": pl.List(pl.Utf8),
                "columns_accessed": pl.List(pl.Utf8),
                "latency_ms": pl.Float64,
                "tokens_used": pl.Int64,
                "is_valid": pl.Boolean,
                "anomaly_score": pl.Float64,
                "raw_payload_json": pl.Utf8,
            }

            df = pl.DataFrame(records, schema=schema)
            df.write_parquet(target_file, compression=self.config.compression)
            written_paths.append(str(target_file))

        return written_paths

    def _resolve_files(
        self,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> List[Path]:
        if not self.base_dir.exists():
            return []

        search_root = self.base_dir
        if tenant_id and event_date:
            search_root = self.base_dir / f"tenant_id={tenant_id}" / f"event_date={event_date}"
        elif tenant_id:
            search_root = self.base_dir / f"tenant_id={tenant_id}"

        if not search_root.exists():
            return []

        files = list(search_root.rglob("*.parquet"))
        if event_date and not tenant_id:
            files = [p for p in files if f"event_date={event_date}" in str(p)]
        return files

    def read_raw(
        self,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> pl.DataFrame:
        files = self._resolve_files(tenant_id=tenant_id, event_date=event_date)
        if not files:
            return pl.DataFrame()
        dfs = [pl.read_parquet(p) for p in files]
        return pl.concat(dfs)

    def scan_raw(
        self,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> pl.LazyFrame:
        files = self._resolve_files(tenant_id=tenant_id, event_date=event_date)
        if not files:
            return pl.LazyFrame()
        file_paths = [str(p) for p in files]
        return pl.scan_parquet(file_paths)

    def get_total_events_count(self) -> int:
        files = self._resolve_files()
        if not files:
            return 0
        total = 0
        for p in files:
            total += pl.read_parquet(p).shape[0]
        return total

    def get_partitions(self) -> List[Dict[str, str]]:
        if not self.base_dir.exists():
            return []
        partitions: List[Dict[str, str]] = []
        for t_dir in self.base_dir.glob("tenant_id=*"):
            if not t_dir.is_dir():
                continue
            tid = t_dir.name.split("=")[-1]
            for d_dir in t_dir.glob("event_date=*"):
                if not d_dir.is_dir():
                    continue
                dt = d_dir.name.split("=")[-1]
                partitions.append({"tenant_id": tid, "event_date": dt})
        return partitions
