from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import polars as pl
import sqlglot
from sqlglot import exp

import src.stream_lakehouse as slh


def classify_failure(
    event: slh.StreamEvent,
    catalog: Optional[Any] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    if not event.tenant_id or event.tenant_id.strip() == "":
        return True, "Missing tenant identification", "TENANT_ISOLATION_VIOLATION"

    sql = event.generated_sql
    if sql:
        try:
            parsed = sqlglot.parse_one(sql)
            if isinstance(parsed, (exp.Delete, exp.Drop, exp.Update, exp.Insert, exp.Alter)):
                return True, "Destructive DDL/DML mutation statement rejected", "AST_VALIDATION_ERROR"
        except Exception as exc:
            return True, f"SQL syntax parsing failure: {str(exc)}", "AST_VALIDATION_ERROR"

        lower_sql = sql.lower()
        tid_token = event.tenant_id.lower()
        if "tenant_id" not in lower_sql or tid_token not in lower_sql:
            return True, f"Tenant isolation predicate missing for {event.tenant_id}", "TENANT_ISOLATION_VIOLATION"

        if catalog is not None and hasattr(catalog, "has_table"):
            for tbl in event.tables_accessed:
                if not catalog.has_table(tbl):
                    return True, f"Referenced table '{tbl}' absent from catalog", "SCHEMA_MISMATCH"

    if event.anomaly_score is not None and event.anomaly_score > 0.85:
        return True, f"Anomaly score {event.anomaly_score} exceeded threshold 0.85", "ANOMALY_THRESHOLD_BREACH"

    if not event.is_valid:
        return True, "Event marked invalid by generator or upstream validator", "VALIDATION_FAILURE"

    return False, None, None


class DLQParquetSink:
    def __init__(self, base_dir: str = "data/dlq_quarantine") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_record(self, record: slh.DLQRecord) -> str:
        tid = record.original_event.tenant_id
        category = record.error_category
        target_dir = self.base_dir / f"tenant_id={tid}" / f"category={category}"
        target_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"dlq_{record.dlq_id}.parquet"
        file_path = target_dir / file_name

        data = {
            "dlq_id": [record.dlq_id],
            "event_id": [record.original_event.event_id],
            "tenant_id": [tid],
            "error_category": [category],
            "failure_reason": [record.failure_reason],
            "timestamp": [record.timestamp],
            "retry_count": [record.retry_count],
            "generated_sql": [record.original_event.generated_sql or ""],
            "natural_query": [record.original_event.natural_query or ""],
            "anomaly_score": [record.original_event.anomaly_score or 0.0],
            "diagnostics_json": [json.dumps(record.diagnostics)],
        }

        df = pl.DataFrame(data)
        df.write_parquet(file_path, compression="snappy")
        return str(file_path)

    def write_batch(self, records: List[slh.DLQRecord]) -> int:
        written = 0
        for rec in records:
            self.write_record(rec)
            written += 1
        return written

    def read_records(
        self,
        tenant_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.base_dir.exists():
            return []

        search_path = self.base_dir
        if tenant_id and category:
            search_path = self.base_dir / f"tenant_id={tenant_id}" / f"category={category}"
        elif tenant_id:
            search_path = self.base_dir / f"tenant_id={tenant_id}"

        parquet_files = list(search_path.rglob("*.parquet"))
        if not parquet_files:
            return []

        dfs = [pl.read_parquet(p) for p in parquet_files]
        combined = pl.concat(dfs)
        return combined.to_dicts()


class DLQRouter:
    def __init__(
        self,
        config: Optional[slh.StreamConfig] = None,
        parquet_sink: Optional[DLQParquetSink] = None,
    ) -> None:
        self.config = config or slh.StreamConfig()
        self.parquet_sink = parquet_sink
        self._records: Dict[str, slh.DLQRecord] = {}

    def route(
        self,
        event: slh.StreamEvent,
        failure_reason: str,
        error_category: str,
        stack_trace: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> slh.DLQRecord:
        record = slh.DLQRecord(
            original_event=event,
            failure_reason=failure_reason,
            error_category=error_category,
            timestamp=time.time(),
            retry_count=0,
            stack_trace=stack_trace,
            diagnostics=diagnostics or {},
        )
        self._records[record.dlq_id] = record

        if self.parquet_sink is not None:
            self.parquet_sink.write_record(record)

        return record

    def process_and_route(
        self,
        event: slh.StreamEvent,
        catalog: Optional[Any] = None,
    ) -> Optional[slh.DLQRecord]:
        failed, reason, category = classify_failure(event, catalog=catalog)
        if failed and reason and category:
            return self.route(
                event=event,
                failure_reason=reason,
                error_category=category,
                diagnostics={"quarantined_by": "DLQRouter.process_and_route"},
            )
        return None

    def replay(self, dlq_id: str) -> Optional[slh.StreamEvent]:
        rec = self._records.get(dlq_id)
        if rec is None:
            return None

        if rec.retry_count >= self.config.dlq_threshold_retries:
            return None

        updated = slh.DLQRecord(
            dlq_id=rec.dlq_id,
            original_event=rec.original_event,
            failure_reason=rec.failure_reason,
            error_category=rec.error_category,
            timestamp=rec.timestamp,
            retry_count=rec.retry_count + 1,
            stack_trace=rec.stack_trace,
            diagnostics=rec.diagnostics,
        )
        self._records[dlq_id] = updated
        return updated.original_event

    def get_quarantined_count(self) -> int:
        return len(self._records)

    def get_records(
        self,
        category: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[slh.DLQRecord]:
        results = list(self._records.values())
        if category is not None:
            results = [r for r in results if r.error_category == category]
        if tenant_id is not None:
            results = [r for r in results if r.original_event.tenant_id == tenant_id]
        return results
