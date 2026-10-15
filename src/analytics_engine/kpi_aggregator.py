from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence
import polars as pl

from src.analytics_engine import (
    CohortTier,
    KPICohort,
    TrendDirection,
    TrendMetric,
    TrendReport,
)
from src.analytics_engine.hypothesis import StatisticalHypothesisEngine
from src.analytics_engine.olap import DuckDBWarehouse


class KPICohortAggregator:
    def __init__(
        self,
        hypothesis_engine: Optional[StatisticalHypothesisEngine] = None,
        sla_p99_latency_ms: float = 100.0,
    ) -> None:
        self.hypothesis_engine = hypothesis_engine or StatisticalHypothesisEngine()
        self.sla_p99_latency_ms = sla_p99_latency_ms

    def classify_tenant_tier(
        self,
        query_count: int,
        error_rate: float,
        anomaly_rate: float,
    ) -> CohortTier:
        if error_rate >= 0.05 or anomaly_rate >= 0.10:
            return CohortTier.AT_RISK
        if query_count >= 1000:
            return CohortTier.ENTERPRISE
        if query_count >= 100:
            return CohortTier.GROWTH
        return CohortTier.STARTER

    def compute_tenant_cohorts(
        self,
        warehouse: DuckDBWarehouse,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> List[KPICohort]:
        p_start = period_start or (
            datetime.datetime.now() - datetime.timedelta(days=30)
        ).strftime("%Y-%m-%d")
        p_end = period_end or datetime.datetime.now().strftime("%Y-%m-%d")

        matrix_df = warehouse.get_tenant_performance_matrix()
        if matrix_df.is_empty():
            return []

        cohorts = []
        for row in matrix_df.iter_rows(named=True):
            tenant_id = str(row["tenant_id"])
            query_count = int(row["total_queries"])
            p50 = float(row["p50_latency_ms"]) if row["p50_latency_ms"] is not None else 0.0
            p90 = float(row["p90_latency_ms"]) if row["p90_latency_ms"] is not None else 0.0
            p99 = float(row["p99_latency_ms"]) if row["p99_latency_ms"] is not None else 0.0
            mean_lat = float(row["mean_latency_ms"]) if row["mean_latency_ms"] is not None else 0.0
            error_rate = float(row["error_rate"]) if row["error_rate"] is not None else 0.0
            anomaly_rate = float(row["anomaly_rate"]) if row["anomaly_rate"] is not None else 0.0
            tokens = int(row["total_tokens"]) if row["total_tokens"] is not None else 0

            tier = self.classify_tenant_tier(
                query_count=query_count,
                error_rate=error_rate,
                anomaly_rate=anomaly_rate,
            )

            cohort_id = f"cohort_{tenant_id}_{p_start}_{p_end}"
            cohorts.append(
                KPICohort(
                    cohort_id=cohort_id,
                    tenant_id=tenant_id,
                    cohort_tier=tier,
                    query_count=query_count,
                    p50_latency_ms=round(p50, 3),
                    p90_latency_ms=round(p90, 3),
                    p99_latency_ms=round(p99, 3),
                    mean_latency_ms=round(mean_lat, 3),
                    error_rate=round(error_rate, 4),
                    anomaly_rate=round(anomaly_rate, 4),
                    tokens_consumed=tokens,
                    period_start=p_start,
                    period_end=p_end,
                )
            )

        return cohorts

    def compute_health_score(
        self,
        error_rate: float,
        anomaly_rate: float,
        p99_latency_ms: float,
    ) -> float:
        score = 100.0
        score -= min(40.0, error_rate * 400.0)
        score -= min(30.0, anomaly_rate * 300.0)

        if p99_latency_ms > self.sla_p99_latency_ms:
            excess = p99_latency_ms - self.sla_p99_latency_ms
            latency_penalty = min(30.0, (excess / self.sla_p99_latency_ms) * 20.0)
            score -= latency_penalty

        return round(max(0.0, min(100.0, score)), 2)

    def compare_time_periods(
        self,
        warehouse: DuckDBWarehouse,
        tenant_id: str,
        current_days: int = 7,
        baseline_days: int = 7,
    ) -> TrendReport:
        now = datetime.datetime.now()
        cur_start = (now - datetime.timedelta(days=current_days)).strftime("%Y-%m-%d %H:%M:%S")
        base_start = (
            now - datetime.timedelta(days=current_days + baseline_days)
        ).strftime("%Y-%m-%d %H:%M:%S")

        cur_sql = f"""
            SELECT latency_ms, is_error, is_anomaly, tokens_used
            FROM fact_query_events
            WHERE tenant_id = '{tenant_id}' AND timestamp >= '{cur_start}';
        """
        base_sql = f"""
            SELECT latency_ms, is_error, is_anomaly, tokens_used
            FROM fact_query_events
            WHERE tenant_id = '{tenant_id}' AND timestamp >= '{base_start}' AND timestamp < '{cur_start}';
        """

        cur_df = warehouse.query(cur_sql)
        base_df = warehouse.query(base_sql)

        n_cur = len(cur_df)
        n_base = len(base_df)

        metrics: Dict[str, TrendMetric] = {}

        vol_change = (
            ((n_cur - n_base) / n_base * 100.0)
            if n_base > 0
            else (100.0 if n_cur > 0 else 0.0)
        )
        vol_dir = (
            TrendDirection.INCREASING
            if vol_change > 5.0
            else (TrendDirection.DECREASING if vol_change < -5.0 else TrendDirection.STABLE)
        )
        metrics["query_volume"] = TrendMetric(
            metric_name="query_volume",
            direction=vol_dir,
            percentage_change=round(vol_change, 2),
            current_value=float(n_cur),
            baseline_value=float(n_base),
            is_significant=abs(vol_change) >= 20.0 and (n_cur + n_base >= 10),
            p_value=0.01 if abs(vol_change) >= 20.0 else 0.50,
        )

        lat_cur = cur_df["latency_ms"].to_list() if "latency_ms" in cur_df.columns else []
        lat_base = base_df["latency_ms"].to_list() if "latency_ms" in base_df.columns else []

        if lat_cur and lat_base:
            test_res = self.hypothesis_engine.compare_continuous_cohorts(
                lat_cur, lat_base, num_tests=3
            )
            mean_cur = float(sum(lat_cur) / len(lat_cur))
            mean_base = float(sum(lat_base) / len(lat_base))
            lat_change = ((mean_cur - mean_base) / mean_base * 100.0) if mean_base > 0 else 0.0
            lat_dir = (
                TrendDirection.INCREASING
                if lat_change > 5.0
                else (TrendDirection.DECREASING if lat_change < -5.0 else TrendDirection.STABLE)
            )
            metrics["latency"] = TrendMetric(
                metric_name="latency",
                direction=lat_dir,
                percentage_change=round(lat_change, 2),
                current_value=round(mean_cur, 3),
                baseline_value=round(mean_base, 3),
                is_significant=test_res.is_significant,
                p_value=test_res.p_value_adjusted,
            )
        else:
            metrics["latency"] = TrendMetric(
                metric_name="latency",
                direction=TrendDirection.STABLE,
                percentage_change=0.0,
                current_value=0.0,
                baseline_value=0.0,
                is_significant=False,
                p_value=1.0,
            )

        err_cur = (
            float(cur_df["is_error"].mean())
            if "is_error" in cur_df.columns and n_cur > 0
            else 0.0
        )
        err_base = (
            float(base_df["is_error"].mean())
            if "is_error" in base_df.columns and n_base > 0
            else 0.0
        )
        err_diff = round((err_cur - err_base) * 100.0, 2)
        err_dir = (
            TrendDirection.INCREASING
            if err_diff > 1.0
            else (TrendDirection.DECREASING if err_diff < -1.0 else TrendDirection.STABLE)
        )
        metrics["error_rate"] = TrendMetric(
            metric_name="error_rate",
            direction=err_dir,
            percentage_change=err_diff,
            current_value=round(err_cur, 4),
            baseline_value=round(err_base, 4),
            is_significant=abs(err_diff) >= 5.0,
            p_value=0.01 if abs(err_diff) >= 5.0 else 0.40,
        )

        anom_cur = (
            float(cur_df["is_anomaly"].mean())
            if "is_anomaly" in cur_df.columns and n_cur > 0
            else 0.0
        )
        anom_base = (
            float(base_df["is_anomaly"].mean())
            if "is_anomaly" in base_df.columns and n_base > 0
            else 0.0
        )
        anom_diff = round((anom_cur - anom_base) * 100.0, 2)
        anom_dir = (
            TrendDirection.INCREASING
            if anom_diff > 1.0
            else (TrendDirection.DECREASING if anom_diff < -1.0 else TrendDirection.STABLE)
        )
        metrics["anomaly_rate"] = TrendMetric(
            metric_name="anomaly_rate",
            direction=anom_dir,
            percentage_change=anom_diff,
            current_value=round(anom_cur, 4),
            baseline_value=round(anom_base, 4),
            is_significant=abs(anom_diff) >= 5.0,
            p_value=0.01 if abs(anom_diff) >= 5.0 else 0.40,
        )

        lat_p99_cur = (
            float(cur_df["latency_ms"].quantile(0.99))
            if "latency_ms" in cur_df.columns and n_cur > 0
            else 0.0
        )
        health_score = self.compute_health_score(
            error_rate=err_cur,
            anomaly_rate=anom_cur,
            p99_latency_ms=lat_p99_cur,
        )

        drift_detected = (
            metrics["error_rate"].is_significant
            or metrics["anomaly_rate"].is_significant
            or (metrics["latency"].is_significant and metrics["latency"].direction == TrendDirection.INCREASING)
        )

        return TrendReport(
            tenant_id=tenant_id,
            period_label=f"Last {current_days}d vs Prior {baseline_days}d",
            start_date=base_start,
            end_date=cur_start,
            metrics=metrics,
            overall_health_score=health_score,
            anomalous_drift_detected=drift_detected,
        )
