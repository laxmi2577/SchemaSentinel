from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


class DriftMonitorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    ks_alpha: float = 0.05
    psi_threshold_warning: float = 0.1
    psi_threshold_drift: float = 0.25
    min_samples: int = 5
    num_bins: int = 10
    output_dir: str = "artifacts/drift"
    dry_run: bool = False


class DriftMetricResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_name: str
    statistic: float
    p_value: Optional[float] = None
    drift_detected: bool
    threshold: float
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class DriftReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_id: str
    timestamp: str
    has_drift: bool
    metrics: Dict[str, DriftMetricResult]
    summary: str


def compute_kolmogorov_smirnov(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> Tuple[float, float]:
    n1 = len(sample_a)
    n2 = len(sample_b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    try:
        from scipy import stats

        res = stats.ks_2samp(sample_a, sample_b)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        pass

    vals_a = sorted(sample_a)
    vals_b = sorted(sample_b)
    all_vals = sorted(set(vals_a + vals_b))

    d_stat = 0.0
    i = 0
    j = 0
    for v in all_vals:
        while i < n1 and vals_a[i] <= v:
            i += 1
        while j < n2 and vals_b[j] <= v:
            j += 1
        cdf_a = i / n1
        cdf_b = j / n2
        diff = abs(cdf_a - cdf_b)
        if diff > d_stat:
            d_stat = diff

    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    p_val = max(0.0, min(1.0, 2.0 * math.exp(-2.0 * (lambda_val**2))))
    return float(d_stat), float(p_val)


def compute_population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    if not reference or not current:
        return 0.0

    all_vals = list(reference) + list(current)
    min_val = min(all_vals)
    max_val = max(all_vals)

    if min_val == max_val:
        return 0.0

    bin_edges = [
        min_val + (max_val - min_val) * (i / num_bins) for i in range(num_bins + 1)
    ]

    ref_counts = [0] * num_bins
    cur_counts = [0] * num_bins

    for val in reference:
        assigned = False
        for b in range(num_bins):
            if val <= bin_edges[b + 1] or b == num_bins - 1:
                ref_counts[b] += 1
                assigned = True
                break
        if not assigned:
            ref_counts[-1] += 1

    for val in current:
        assigned = False
        for b in range(num_bins):
            if val <= bin_edges[b + 1] or b == num_bins - 1:
                cur_counts[b] += 1
                assigned = True
                break
        if not assigned:
            cur_counts[-1] += 1

    total_ref = len(reference)
    total_cur = len(current)

    psi = 0.0
    for b in range(num_bins):
        actual_pct = (cur_counts[b] / total_cur) + epsilon
        expected_pct = (ref_counts[b] / total_ref) + epsilon
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)

    return float(round(psi, 5))


def extract_schema_complexity_metrics(
    sql_queries: Sequence[str],
) -> List[float]:
    metrics: List[float] = []
    for q in sql_queries:
        cleaned = q.strip()
        if not cleaned:
            continue
        joins = len(re.findall(r"\bJOIN\b", cleaned, re.IGNORECASE))
        wheres = len(re.findall(r"\bWHERE\b|\bAND\b|\bOR\b", cleaned, re.IGNORECASE))
        groups = len(re.findall(r"\bGROUP\s+BY\b", cleaned, re.IGNORECASE))
        subqueries = cleaned.count("(")
        words = len(cleaned.split())

        complexity_score = (joins * 3.0) + (wheres * 1.5) + (groups * 2.0) + (subqueries * 2.5) + (words * 0.1)
        metrics.append(round(complexity_score, 3))
    return metrics


def extract_token_length_distribution(
    sql_queries: Sequence[str],
) -> List[float]:
    return [float(len(q.split())) for q in sql_queries if q.strip()]


class EvidentlyDriftMonitor:
    def __init__(
        self,
        config: Optional[DriftMonitorConfig] = None,
    ) -> None:
        self.config = config or DriftMonitorConfig()

    def analyze_schema_complexity_drift(
        self,
        reference_queries: Sequence[str],
        production_queries: Sequence[str],
    ) -> DriftMetricResult:
        ref_metrics = extract_schema_complexity_metrics(reference_queries)
        prod_metrics = extract_schema_complexity_metrics(production_queries)

        if len(ref_metrics) < self.config.min_samples or len(prod_metrics) < self.config.min_samples:
            return DriftMetricResult(
                metric_name="schema_complexity_ks_test",
                statistic=0.0,
                p_value=1.0,
                drift_detected=False,
                threshold=self.config.ks_alpha,
                status="insufficient_samples",
                details={"ref_count": len(ref_metrics), "prod_count": len(prod_metrics)},
            )

        d_stat, p_val = compute_kolmogorov_smirnov(ref_metrics, prod_metrics)
        drift_detected = p_val < self.config.ks_alpha

        status = "drift_detected" if drift_detected else "stable"
        return DriftMetricResult(
            metric_name="schema_complexity_ks_test",
            statistic=d_stat,
            p_value=p_val,
            drift_detected=drift_detected,
            threshold=self.config.ks_alpha,
            status=status,
            details={
                "ref_mean": round(sum(ref_metrics) / len(ref_metrics), 3),
                "prod_mean": round(sum(prod_metrics) / len(prod_metrics), 3),
            },
        )

    def analyze_token_length_psi_drift(
        self,
        reference_queries: Sequence[str],
        production_queries: Sequence[str],
    ) -> DriftMetricResult:
        ref_tokens = extract_token_length_distribution(reference_queries)
        prod_tokens = extract_token_length_distribution(production_queries)

        if len(ref_tokens) < self.config.min_samples or len(prod_tokens) < self.config.min_samples:
            return DriftMetricResult(
                metric_name="token_distribution_psi",
                statistic=0.0,
                p_value=None,
                drift_detected=False,
                threshold=self.config.psi_threshold_drift,
                status="insufficient_samples",
                details={"ref_count": len(ref_tokens), "prod_count": len(prod_tokens)},
            )

        psi = compute_population_stability_index(
            reference=ref_tokens,
            current=prod_tokens,
            num_bins=self.config.num_bins,
        )

        drift_detected = psi >= self.config.psi_threshold_drift
        status = "significant_drift" if drift_detected else ("warning" if psi >= self.config.psi_threshold_warning else "stable")

        return DriftMetricResult(
            metric_name="token_distribution_psi",
            statistic=psi,
            p_value=None,
            drift_detected=drift_detected,
            threshold=self.config.psi_threshold_drift,
            status=status,
            details={
                "warning_threshold": self.config.psi_threshold_warning,
                "drift_threshold": self.config.psi_threshold_drift,
            },
        )

    def run_drift_audit(
        self,
        reference_queries: Sequence[str],
        production_queries: Sequence[str],
    ) -> DriftReport:
        ks_res = self.analyze_schema_complexity_drift(reference_queries, production_queries)
        psi_res = self.analyze_token_length_psi_drift(reference_queries, production_queries)

        metrics = {
            "schema_complexity": ks_res,
            "token_distribution": psi_res,
        }

        has_drift = ks_res.drift_detected or psi_res.drift_detected
        summary = (
            f"Drift Audit: has_drift={has_drift} | "
            f"KS-stat={ks_res.statistic:.4f} (p={ks_res.p_value}) | "
            f"PSI={psi_res.statistic:.4f} ({psi_res.status})"
        )

        return DriftReport(
            report_id=str(uuid.uuid4())[:8],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            has_drift=has_drift,
            metrics=metrics,
            summary=summary,
        )

    def export_drift_report(
        self,
        report: DriftReport,
        output_path: str,
    ) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        return str(out)
