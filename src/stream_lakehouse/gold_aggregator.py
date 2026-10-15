from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from src.stream_lakehouse.silver_transformer import SilverTransformer, SilverTransformerConfig


class GoldAggregatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_dir: str = "data/lakehouse/gold"
    silver_dir: str = "data/lakehouse/silver"
    compression: str = "snappy"
    anomaly_threshold: float = 0.5


class GoldAggregationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_silver_records_processed: int = 0
    total_gold_records_generated: int = 0
    materialized_tables: List[str] = Field(default_factory=list)
    partitions_materialized: List[str] = Field(default_factory=list)


class GoldAggregator:
    def __init__(self, config: Optional[GoldAggregatorConfig] = None) -> None:
        self.config = config or GoldAggregatorConfig()
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def compute_tenant_kpis(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        if isinstance(df, pl.LazyFrame):
            data = df.collect()
        else:
            data = df

        if data.is_empty():
            schema = {
                "tenant_id": pl.Utf8,
                "event_date": pl.Utf8,
                "total_queries": pl.Int64,
                "valid_queries": pl.Int64,
                "invalid_queries": pl.Int64,
                "error_rate": pl.Float64,
                "latency_p50": pl.Float64,
                "latency_p90": pl.Float64,
                "latency_p99": pl.Float64,
                "latency_mean": pl.Float64,
                "total_tokens": pl.Int64,
                "anomaly_count": pl.Int64,
                "mean_anomaly_score": pl.Float64,
            }
            return pl.DataFrame([], schema=schema)

        grouped = (
            data.lazy()
            .group_by(["tenant_id", "event_date"])
            .agg(
                [
                    pl.len().alias("total_queries"),
                    pl.col("is_valid").cast(pl.Int64).sum().alias("valid_queries"),
                    pl.col("latency_ms").quantile(0.50).alias("latency_p50"),
                    pl.col("latency_ms").quantile(0.90).alias("latency_p90"),
                    pl.col("latency_ms").quantile(0.99).alias("latency_p99"),
                    pl.col("latency_ms").mean().alias("latency_mean"),
                    pl.col("tokens_used").sum().alias("total_tokens"),
                    (pl.col("anomaly_score") > self.config.anomaly_threshold)
                    .cast(pl.Int64)
                    .sum()
                    .alias("anomaly_count"),
                    pl.col("anomaly_score").mean().alias("mean_anomaly_score"),
                ]
            )
            .with_columns(
                [
                    (pl.col("total_queries") - pl.col("valid_queries")).alias("invalid_queries"),
                    (
                        (pl.col("total_queries") - pl.col("valid_queries")).cast(pl.Float64)
                        / pl.col("total_queries").cast(pl.Float64)
                    ).alias("error_rate"),
                ]
            )
            .select(
                [
                    "tenant_id",
                    "event_date",
                    "total_queries",
                    "valid_queries",
                    "invalid_queries",
                    "error_rate",
                    "latency_p50",
                    "latency_p90",
                    "latency_p99",
                    "latency_mean",
                    "total_tokens",
                    "anomaly_count",
                    "mean_anomaly_score",
                ]
            )
            .collect()
        )

        return grouped

    def compute_hourly_rollups(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        if isinstance(df, pl.LazyFrame):
            data = df.collect()
        else:
            data = df

        if data.is_empty():
            schema = {
                "tenant_id": pl.Utf8,
                "event_date": pl.Utf8,
                "window_hour": pl.Utf8,
                "query_count": pl.Int64,
                "mean_latency_ms": pl.Float64,
                "error_count": pl.Int64,
                "total_tokens": pl.Int64,
            }
            return pl.DataFrame([], schema=schema)

        records = data.to_dicts()
        augmented_records: List[Dict[str, Any]] = []
        for rec in records:
            ts = float(rec.get("timestamp") or 0.0)
            dt = datetime.fromtimestamp(ts, timezone.utc)
            rec["window_hour"] = dt.strftime("%Y-%m-%d %H:00:00")
            augmented_records.append(rec)

        aug_df = pl.DataFrame(augmented_records)
        grouped = (
            aug_df.lazy()
            .group_by(["tenant_id", "event_date", "window_hour"])
            .agg(
                [
                    pl.len().alias("query_count"),
                    pl.col("latency_ms").mean().alias("mean_latency_ms"),
                    (pl.col("is_valid") == False).cast(pl.Int64).sum().alias("error_count"),
                    pl.col("tokens_used").sum().alias("total_tokens"),
                ]
            )
            .collect()
        )
        return grouped

    def compute_tenant_cohorts(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        kpi_df = self.compute_tenant_kpis(df)
        if kpi_df.is_empty():
            schema = {
                "tenant_id": pl.Utf8,
                "total_volume": pl.Int64,
                "mean_latency_ms": pl.Float64,
                "overall_error_rate": pl.Float64,
                "activity_cohort": pl.Utf8,
                "health_status": pl.Utf8,
            }
            return pl.DataFrame([], schema=schema)

        cohort_data = (
            kpi_df.lazy()
            .group_by("tenant_id")
            .agg(
                [
                    pl.col("total_queries").sum().alias("total_volume"),
                    pl.col("latency_mean").mean().alias("mean_latency_ms"),
                    pl.col("error_rate").mean().alias("overall_error_rate"),
                    pl.col("mean_anomaly_score").mean().alias("overall_anomaly_rate"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("total_volume") >= 50)
                    .then(pl.lit("high_volume"))
                    .when(pl.col("total_volume") >= 10)
                    .then(pl.lit("medium_volume"))
                    .otherwise(pl.lit("low_volume"))
                    .alias("activity_cohort"),
                    pl.when((pl.col("overall_error_rate") > 0.10) | (pl.col("overall_anomaly_rate") > 0.50))
                    .then(pl.lit("degraded"))
                    .otherwise(pl.lit("nominal"))
                    .alias("health_status"),
                ]
            )
            .select(
                [
                    "tenant_id",
                    "total_volume",
                    "mean_latency_ms",
                    "overall_error_rate",
                    "activity_cohort",
                    "health_status",
                ]
            )
            .collect()
        )
        return cohort_data

    def write_gold_table(self, df: pl.DataFrame, table_name: str) -> List[str]:
        if df.is_empty():
            return []

        table_dir = self.base_dir / table_name
        table_dir.mkdir(parents=True, exist_ok=True)

        written_paths: List[str] = []

        if "tenant_id" in df.columns and "event_date" in df.columns:
            partition_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
            for record in df.to_dicts():
                key = (str(record["tenant_id"]), str(record["event_date"]))
                if key not in partition_groups:
                    partition_groups[key] = []
                partition_groups[key].append(record)

            for (tenant_id, event_date), records in partition_groups.items():
                part_dir = table_dir / f"tenant_id={tenant_id}" / f"event_date={event_date}"
                part_dir.mkdir(parents=True, exist_ok=True)
                batch_id = str(uuid.uuid4())[:8]
                target_file = part_dir / f"gold_{table_name}_{batch_id}.parquet"
                part_df = pl.DataFrame(records, schema=df.schema)
                part_df.write_parquet(target_file, compression=self.config.compression)
                written_paths.append(str(target_file))

        elif "tenant_id" in df.columns:
            partition_groups_tenant: Dict[str, List[Dict[str, Any]]] = {}
            for record in df.to_dicts():
                tid = str(record["tenant_id"])
                if tid not in partition_groups_tenant:
                    partition_groups_tenant[tid] = []
                partition_groups_tenant[tid].append(record)

            for tid, records in partition_groups_tenant.items():
                part_dir = table_dir / f"tenant_id={tid}"
                part_dir.mkdir(parents=True, exist_ok=True)
                batch_id = str(uuid.uuid4())[:8]
                target_file = part_dir / f"gold_{table_name}_{batch_id}.parquet"
                part_df = pl.DataFrame(records, schema=df.schema)
                part_df.write_parquet(target_file, compression=self.config.compression)
                written_paths.append(str(target_file))

        else:
            batch_id = str(uuid.uuid4())[:8]
            target_file = table_dir / f"gold_{table_name}_{batch_id}.parquet"
            df.write_parquet(target_file, compression=self.config.compression)
            written_paths.append(str(target_file))

        return written_paths

    def materialize_from_silver(
        self,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> GoldAggregationSummary:
        transformer = SilverTransformer(SilverTransformerConfig(base_dir=self.config.silver_dir))
        silver_lf = transformer.scan_silver(tenant_id=tenant_id, event_date=event_date)
        silver_df = silver_lf.collect()

        if silver_df.is_empty():
            return GoldAggregationSummary()

        total_silver = silver_df.shape[0]

        tenant_kpis = self.compute_tenant_kpis(silver_df)
        hourly_rollups = self.compute_hourly_rollups(silver_df)
        cohorts = self.compute_tenant_cohorts(silver_df)

        paths_kpi = self.write_gold_table(tenant_kpis, "kpi_tenant_daily")
        paths_hourly = self.write_gold_table(hourly_rollups, "hourly_rollups")
        paths_cohorts = self.write_gold_table(cohorts, "tenant_cohorts")

        all_paths = paths_kpi + paths_hourly + paths_cohorts
        total_generated = tenant_kpis.shape[0] + hourly_rollups.shape[0] + cohorts.shape[0]

        partitions = list({Path(p).parent.name for p in all_paths})

        return GoldAggregationSummary(
            total_silver_records_processed=total_silver,
            total_gold_records_generated=total_generated,
            materialized_tables=["kpi_tenant_daily", "hourly_rollups", "tenant_cohorts"],
            partitions_materialized=partitions,
        )

    def _resolve_files(
        self,
        table_name: str,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> List[Path]:
        table_dir = self.base_dir / table_name
        if not table_dir.exists():
            return []

        search_root = table_dir
        if tenant_id and event_date:
            search_root = table_dir / f"tenant_id={tenant_id}" / f"event_date={event_date}"
        elif tenant_id:
            search_root = table_dir / f"tenant_id={tenant_id}"

        if not search_root.exists():
            return []

        files = list(search_root.rglob("*.parquet"))
        if event_date and not tenant_id:
            files = [p for p in files if f"event_date={event_date}" in str(p)]
        return files

    def read_gold(
        self,
        table_name: str,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> pl.DataFrame:
        files = self._resolve_files(table_name, tenant_id=tenant_id, event_date=event_date)
        if not files:
            return pl.DataFrame()
        dfs = [pl.read_parquet(p) for p in files]
        return pl.concat(dfs)

    def scan_gold(
        self,
        table_name: str,
        tenant_id: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> pl.LazyFrame:
        files = self._resolve_files(table_name, tenant_id=tenant_id, event_date=event_date)
        if not files:
            return pl.LazyFrame()
        file_paths = [str(p) for p in files]
        return pl.scan_parquet(file_paths)

    def get_partitions(self, table_name: str) -> List[Dict[str, str]]:
        table_dir = self.base_dir / table_name
        if not table_dir.exists():
            return []
        partitions: List[Dict[str, str]] = []
        for t_dir in table_dir.glob("tenant_id=*"):
            if not t_dir.is_dir():
                continue
            tid = t_dir.name.split("=")[-1]
            sub_dates = list(t_dir.glob("event_date=*"))
            if sub_dates:
                for d_dir in sub_dates:
                    if not d_dir.is_dir():
                        continue
                    dt = d_dir.name.split("=")[-1]
                    partitions.append({"tenant_id": tid, "event_date": dt})
            else:
                partitions.append({"tenant_id": tid})
        return partitions
