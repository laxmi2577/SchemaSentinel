from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.stats as stats

from src.analytics_engine import (
    EffectSizeType,
    HypothesisTestResult,
    HypothesisTestType,
)


class StatisticalHypothesisEngine:
    def __init__(self, default_alpha: float = 0.05) -> None:
        self.default_alpha = default_alpha

    def welch_t_test(
        self,
        group_a: Sequence[float],
        group_b: Sequence[float],
        alpha: Optional[float] = None,
        num_tests: int = 1,
    ) -> HypothesisTestResult:
        sig_level = alpha or self.default_alpha
        arr_a = np.asarray(group_a, dtype=np.float64)
        arr_b = np.asarray(group_b, dtype=np.float64)

        n_a = len(arr_a)
        n_b = len(arr_b)

        if n_a < 2 or n_b < 2:
            return HypothesisTestResult(
                test_type=HypothesisTestType.WELCH_T_TEST,
                statistic=0.0,
                p_value=1.0,
                p_value_adjusted=1.0,
                is_significant=False,
                alpha=sig_level,
                effect_size_type=EffectSizeType.COHENS_D,
                effect_size=0.0,
                effect_magnitude="negligible",
                sample_size_a=n_a,
                sample_size_b=n_b,
                metadata={"error": "Insufficient sample size (minimum 2 per group)"},
            )

        res = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        statistic = float(res.statistic) if not math.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

        p_adj = min(1.0, p_val * max(1, num_tests))
        is_sig = p_adj < sig_level

        mean_a = float(np.mean(arr_a))
        mean_b = float(np.mean(arr_b))
        var_a = float(np.var(arr_a, ddof=1))
        var_b = float(np.var(arr_b, ddof=1))

        pooled_denom = n_a + n_b - 2
        if pooled_denom > 0:
            pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / pooled_denom
            s_pooled = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
            cohens_d = (mean_a - mean_b) / s_pooled if s_pooled > 0 else 0.0
        else:
            cohens_d = 0.0

        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        return HypothesisTestResult(
            test_type=HypothesisTestType.WELCH_T_TEST,
            statistic=round(statistic, 4),
            p_value=round(p_val, 6),
            p_value_adjusted=round(p_adj, 6),
            is_significant=is_sig,
            alpha=sig_level,
            effect_size_type=EffectSizeType.COHENS_D,
            effect_size=round(cohens_d, 4),
            effect_magnitude=magnitude,
            sample_size_a=n_a,
            sample_size_b=n_b,
            metadata={
                "mean_a": round(mean_a, 4),
                "mean_b": round(mean_b, 4),
                "std_a": round(math.sqrt(var_a), 4) if var_a > 0 else 0.0,
                "std_b": round(math.sqrt(var_b), 4) if var_b > 0 else 0.0,
            },
        )

    def mann_whitney_u_test(
        self,
        group_a: Sequence[float],
        group_b: Sequence[float],
        alpha: Optional[float] = None,
        num_tests: int = 1,
    ) -> HypothesisTestResult:
        sig_level = alpha or self.default_alpha
        arr_a = np.asarray(group_a, dtype=np.float64)
        arr_b = np.asarray(group_b, dtype=np.float64)

        n_a = len(arr_a)
        n_b = len(arr_b)

        if n_a < 2 or n_b < 2:
            return HypothesisTestResult(
                test_type=HypothesisTestType.MANN_WHITNEY_U,
                statistic=0.0,
                p_value=1.0,
                p_value_adjusted=1.0,
                is_significant=False,
                alpha=sig_level,
                effect_size_type=EffectSizeType.RANK_BISERIAL_R,
                effect_size=0.0,
                effect_magnitude="negligible",
                sample_size_a=n_a,
                sample_size_b=n_b,
                metadata={"error": "Insufficient sample size (minimum 2 per group)"},
            )

        res = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

        p_adj = min(1.0, p_val * max(1, num_tests))
        is_sig = p_adj < sig_level

        denom = n_a * n_b
        rank_biserial = 1.0 - (2.0 * u_stat / denom) if denom > 0 else 0.0
        abs_r = abs(rank_biserial)

        if abs_r < 0.1:
            magnitude = "negligible"
        elif abs_r < 0.3:
            magnitude = "small"
        elif abs_r < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"

        return HypothesisTestResult(
            test_type=HypothesisTestType.MANN_WHITNEY_U,
            statistic=round(u_stat, 4),
            p_value=round(p_val, 6),
            p_value_adjusted=round(p_adj, 6),
            is_significant=is_sig,
            alpha=sig_level,
            effect_size_type=EffectSizeType.RANK_BISERIAL_R,
            effect_size=round(rank_biserial, 4),
            effect_magnitude=magnitude,
            sample_size_a=n_a,
            sample_size_b=n_b,
            metadata={
                "median_a": round(float(np.median(arr_a)), 4),
                "median_b": round(float(np.median(arr_b)), 4),
            },
        )

    def chi_squared_test(
        self,
        contingency_table: Sequence[Sequence[int]],
        alpha: Optional[float] = None,
        num_tests: int = 1,
    ) -> HypothesisTestResult:
        sig_level = alpha or self.default_alpha
        table = np.asarray(contingency_table, dtype=np.int64)

        if table.shape[0] < 2 or table.shape[1] < 2 or np.sum(table) == 0:
            return HypothesisTestResult(
                test_type=HypothesisTestType.CHI_SQUARED,
                statistic=0.0,
                p_value=1.0,
                p_value_adjusted=1.0,
                is_significant=False,
                alpha=sig_level,
                effect_size_type=EffectSizeType.CRAMERS_V,
                effect_size=0.0,
                effect_magnitude="negligible",
                sample_size_a=0,
                sample_size_b=0,
                metadata={"error": "Contingency table must be at least 2x2 with non-zero total"},
            )

        res = stats.chi2_contingency(table)
        chi2 = float(res.statistic)
        p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

        p_adj = min(1.0, p_val * max(1, num_tests))
        is_sig = p_adj < sig_level

        n_total = float(np.sum(table))
        min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
        if n_total > 0 and min_dim > 0:
            cramers_v = math.sqrt(chi2 / (n_total * min_dim))
        else:
            cramers_v = 0.0

        if cramers_v < 0.1:
            magnitude = "negligible"
        elif cramers_v < 0.3:
            magnitude = "small"
        elif cramers_v < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"

        n_a = int(np.sum(table[0])) if table.shape[0] > 0 else 0
        n_b = int(np.sum(table[1])) if table.shape[0] > 1 else 0

        return HypothesisTestResult(
            test_type=HypothesisTestType.CHI_SQUARED,
            statistic=round(chi2, 4),
            p_value=round(p_val, 6),
            p_value_adjusted=round(p_adj, 6),
            is_significant=is_sig,
            alpha=sig_level,
            effect_size_type=EffectSizeType.CRAMERS_V,
            effect_size=round(cramers_v, 4),
            effect_magnitude=magnitude,
            sample_size_a=n_a,
            sample_size_b=n_b,
            metadata={"dof": int(res.dof), "total_observations": int(n_total)},
        )

    def compare_continuous_cohorts(
        self,
        group_a: Sequence[float],
        group_b: Sequence[float],
        alpha: Optional[float] = None,
        num_tests: int = 1,
    ) -> HypothesisTestResult:
        arr_a = np.asarray(group_a, dtype=np.float64)
        arr_b = np.asarray(group_b, dtype=np.float64)

        if len(arr_a) < 3 or len(arr_b) < 3:
            return self.welch_t_test(group_a, group_b, alpha=alpha, num_tests=num_tests)

        try:
            _, p_norm_a = stats.shapiro(arr_a[:5000])
            _, p_norm_b = stats.shapiro(arr_b[:5000])
            both_normal = p_norm_a > 0.05 and p_norm_b > 0.05
        except Exception:
            both_normal = False

        if both_normal:
            return self.welch_t_test(group_a, group_b, alpha=alpha, num_tests=num_tests)
        return self.mann_whitney_u_test(group_a, group_b, alpha=alpha, num_tests=num_tests)
