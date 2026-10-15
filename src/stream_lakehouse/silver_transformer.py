from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

import src.stream_lakehouse as slh
from src.stream_lakehouse.bronze_writer import BronzeWriter, BronzeWriterConfig
from src.data_engine.schema_parser import DatabaseCatalog
from src.data_engine.deduplication import MinHashConfig, MinHashSignature, LSHIndex


class SilverTransformerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_dir: str = "data/lakehouse/silver"
    bronze_dir: str = "data/lakehouse/bronze"
    compression: str = "snappy"
    dedup_threshold: float = 0.85
    minhash_num_perm: int = 128
    minhash_shingle_k: int = 3
    default_latency_ms: float = 0.0
    default_anomaly_score: float = 0.0
    default_tokens_used: int = 0


class TransformationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_input_records: int = 0
    total_silver_records: int = 0
    nulls_imputed: int = 0
    duplicates_dropped: int = 0
    catalog_violations_dropped: int = 0
    partitions_processed: List[str] = Field(default_factory=list)


class SilverTransformer:
    def __init__(
        self,
        config: Optional[SilverTransformerConfig] = None,
        catalog: Optional[DatabaseCatalog] = None,
    ) -> None:
        self.config = config or SilverTransformerConfig()
        self.catalog = catalog
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._minhash_cfg = MinHashConfig(
            num_perm=self.config.minhash_num_perm,
            threshold=self.config.dedup_threshold,
            shingle_k=self.config.minhash_shingle_k,
        )
        self._hasher = MinHashSignature(self._minhash_cfg)

    def validate_record_against_catalog(self, record: Dict[str, Any]) -> bool:
        if self.catalog is None:
            return True

        tables = record.get("tables_accessed") or []
        for tbl in tables:
            clean_tbl = str(tbl).strip()
            if not self.catalog.has_table(clean_tbl):
                return False

        columns = record.get("columns_accessed") or []
        for col in columns:
            clean_col = str(col).strip()
            if "." in clean_col:
                tbl_part, col_part = clean_col.split(".", 1)
                tbl_meta = self.catalog.get_table(tbl_part)
                if not tbl_meta or not tbl_meta.has_column(col_part):
                    return False
            else:
                if tables:
                    found = any(
                        self.catalog.get_table(t) and self.catalog.get_table(t).has_column(clean_col)
                        for t in tables
                        if self.catalog.has_table(t)
                    )
                    if not found:
                        return False

        return True

    def transform_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            [
                pl.col("event_id").cast(pl.Utf8),
                pl.col("tenant_id").cast(pl.Utf8),
                pl.col("user_id").cast(pl.Utf8).fill_null("unknown_user"),
                pl.col("timestamp").cast(pl.Float64),
                pl.col("event_date").cast(pl.Utf8),
                pl.col("event_type").cast(pl.Utf8).fill_null("query_execution"),
                pl.col("natural_query").cast(pl.Utf8).fill_null(""),
                pl.col("generated_sql").cast(pl.Utf8).fill_null(""),
                pl.col("latency_ms").cast(pl.Float64).fill_null(self.config.default_latency_ms),
                pl.col("tokens_used").cast(pl.Int64).fill_null(self.config.default_tokens_used),
                pl.col("is_valid").cast(pl.Boolean).fill_null(True),
                pl.col("anomaly_score").cast(pl.Float64).fill_null(self.config.default_anomaly_score),
                pl.col("raw_payload_json").cast(pl.Utf8).fill_null("{}"),
            ]
        )

    def process_bronze_batch(
        self,
        batch: pl.DataFrame | pl.LazyFrame,
    ) -> Tuple[pl.DataFrame, TransformationMetrics]:
        if isinstance(batch, pl.LazyFrame):
            raw_df = batch.collect()
        else:
            raw_df = batch

        if raw_df.is_empty():
            return pl.DataFrame(), TransformationMetrics()

        total_input = raw_df.shape[0]

        null_count_before = 0
        for col_name in raw_df.columns:
            if col_name in ("latency_ms", "tokens_used", "anomaly_score", "user_id", "natural_query", "generated_sql"):
                null_count_before += raw_df[col_name].null_count()

        transformed_df = self.transform_lazy(raw_df.lazy()).collect()

        records = transformed_df.to_dicts()

        catalog_valid_records: List[Dict[str, Any]] = []
        catalog_violations = 0
        for rec in records:
            if self.validate_record_against_catalog(rec):
                catalog_valid_records.append(rec)
            else:
                catalog_violations += 1

        lsh_index = LSHIndex(self._minhash_cfg)
        deduped_records: List[Dict[str, Any]] = []
        duplicates_dropped = 0

        for rec in catalog_valid_records:
            nq = str(rec.get("natural_query") or "").strip()
            sql = str(rec.get("generated_sql") or "").strip()
            text = f"{nq} {sql}".strip()

            if not text:
                deduped_records.append(rec)
                continue

            sig = self._hasher.compute_signature(text)
            candidates = lsh_index.query_candidates(sig)

            is_duplicate = False
            for cand_id in candidates:
                cand_sig = lsh_index.sample_signatures.get(cand_id)
                if cand_sig and self._hasher.estimate_jaccard(sig, cand_sig) >= self.config.dedup_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                duplicates_dropped += 1
            else:
                rec_id = rec.get("event_id") or str(uuid.uuid4())
                lsh_index.insert(rec_id, sig)
                deduped_records.append(rec)

        if not deduped_records:
            schema = transformed_df.schema
            res_df = pl.DataFrame([], schema=schema)
        else:
            res_df = pl.DataFrame(deduped_records, schema=transformed_df.schema)

        metrics = TransformationMetrics(
            total_input_records=total_input,
            total_silver_records=res_df.shape[0],
            nulls_imputed=null_count_before,
            duplicates_dropped=duplicates_dropped,
            catalog_violations_dropped=catalog_violations,
        )

        return res_df, metrics

    def write_silver(self, df: pl.DataFrame) -> List[str]:
        if df.is_empty():
            return []

        partition_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for record in df.to_dicts():
            key = (str(record["tenant_id"]), str(record["event_date"]))
            if key not in partition_groups:
                partition_groups[key] = []
            partition_groups[key].append(record)

        written_paths: List[str] = []
        for (tenant_id, event_date), records in partition_groups.items():
            target_dir = self.base_dir / f"tenant_id={tenant_id}" / f"event_date={event_date}"
            target_dir.mkdir(parents=True, exist_ok=True)

            batch_id = str(uuid.uuid4())[:8]
            file_name = f"silver_{batch_id}.parquet"
            target_file = target_dir / file_name

            part_df = pl.DataFrame(records, schema=df.schema)
            part_df.write_parquet(target_file, compression=self.config.compression)
            written_paths.append(str(target_file))

        return written_paths

    def process_bronze_to_silver(
        self,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Tuple[List[str], TransformationMetrics]:
        bronze_writer = BronzeWriter(BronzeWriterConfig(base_dir=self.config.bronze_dir))
        bronze_lf = bronze_writer.scan_raw(tenant_id=tenant_id, event_date=event_date)
        silver_df, metrics = self.process_bronze_batch(bronze_lf)
        written = self.write_silver(silver_df)

        partitions = list({f"tenant_id={r['tenant_id']}/event_date={r['event_date']}" for r in silver_df.to_dicts()})
        updated_metrics = TransformationMetrics(
            total_input_records=metrics.total_input_records,
            total_silver_records=metrics.total_silver_records,
            nulls_imputed=metrics.nulls_imputed,
            duplicates_dropped=metrics.duplicates_dropped,
            catalog_violations_dropped=metrics.catalog_violations_dropped,
            partitions_processed=partitions,
        )
        return written, updated_metrics

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

    def read_silver(
        self,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> pl.DataFrame:
        files = self._resolve_files(tenant_id=tenant_id, event_date=event_date)
        if not files:
            return pl.DataFrame()
        dfs = [pl.read_parquet(p) for p in files]
        return pl.concat(dfs)

    def scan_silver(
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
