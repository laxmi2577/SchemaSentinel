import numpy as np
import pytest

from src.analytics_engine import (
    EffectSizeType,
    HypothesisTestResult,
    HypothesisTestType,
    StatisticalHypothesisEngine,
)


def test_welch_t_test_significant():
    engine = StatisticalHypothesisEngine(default_alpha=0.05)
    rng = np.random.default_rng(42)

    group_a = rng.normal(loc=15.0, scale=2.0, size=50)
    group_b = rng.normal(loc=35.0, scale=3.0, size=50)

    res = engine.welch_t_test(group_a, group_b)

    assert isinstance(res, HypothesisTestResult)
    assert res.test_type == HypothesisTestType.WELCH_T_TEST
    assert res.is_significant is True
    assert res.p_value < 0.001
    assert res.p_value_adjusted < 0.001
    assert res.effect_size_type == EffectSizeType.COHENS_D
    assert res.effect_magnitude == "large"
    assert res.sample_size_a == 50
    assert res.sample_size_b == 50


def test_welch_t_test_no_difference():
    engine = StatisticalHypothesisEngine(default_alpha=0.05)
    rng = np.random.default_rng(123)

    group_a = rng.normal(loc=20.0, scale=2.0, size=40)
    group_b = rng.normal(loc=20.1, scale=2.0, size=40)

    res = engine.welch_t_test(group_a, group_b)

    assert res.is_significant is False
    assert res.p_value > 0.05
    assert res.effect_magnitude in ("negligible", "small")


def test_welch_t_test_small_sample_guard():
    engine = StatisticalHypothesisEngine()
    res = engine.welch_t_test([10.0], [20.0])
    assert res.is_significant is False
    assert res.p_value == 1.0
    assert "Insufficient sample size" in res.metadata.get("error", "")


def test_bonferroni_correction():
    engine = StatisticalHypothesisEngine(default_alpha=0.05)
    rng = np.random.default_rng(99)

    group_a = rng.normal(loc=20.0, scale=2.0, size=30)
    group_b = rng.normal(loc=21.5, scale=2.0, size=30)

    res_single = engine.welch_t_test(group_a, group_b, num_tests=1)
    res_multiple = engine.welch_t_test(group_a, group_b, num_tests=10)

    assert res_single.p_value == res_multiple.p_value
    assert res_multiple.p_value_adjusted >= res_single.p_value_adjusted
    assert res_multiple.p_value_adjusted <= 1.0


def test_mann_whitney_u_test():
    engine = StatisticalHypothesisEngine(default_alpha=0.05)
    rng = np.random.default_rng(77)

    group_a = rng.exponential(scale=5.0, size=60)
    group_b = rng.exponential(scale=25.0, size=60)

    res = engine.mann_whitney_u_test(group_a, group_b)

    assert isinstance(res, HypothesisTestResult)
    assert res.test_type == HypothesisTestType.MANN_WHITNEY_U
    assert res.is_significant is True
    assert res.effect_size_type == EffectSizeType.RANK_BISERIAL_R
    assert res.effect_size != 0.0
    assert res.sample_size_a == 60
    assert res.sample_size_b == 60


def test_chi_squared_test():
    engine = StatisticalHypothesisEngine(default_alpha=0.05)

    contingency_table = [
        [5, 95],
        [30, 70],
    ]

    res = engine.chi_squared_test(contingency_table)

    assert isinstance(res, HypothesisTestResult)
    assert res.test_type == HypothesisTestType.CHI_SQUARED
    assert res.is_significant is True
    assert res.p_value < 0.001
    assert res.effect_size_type == EffectSizeType.CRAMERS_V
    assert res.effect_size > 0.1
    assert res.sample_size_a == 100
    assert res.sample_size_b == 100


def test_compare_continuous_cohorts():
    engine = StatisticalHypothesisEngine()
    rng = np.random.default_rng(42)

    norm_a = rng.normal(loc=10.0, scale=1.0, size=50)
    norm_b = rng.normal(loc=20.0, scale=1.0, size=50)
    res_normal = engine.compare_continuous_cohorts(norm_a, norm_b)
    assert res_normal.test_type in (HypothesisTestType.WELCH_T_TEST, HypothesisTestType.MANN_WHITNEY_U)
    assert res_normal.is_significant is True

    skew_a = rng.pareto(a=1.5, size=50)
    skew_b = rng.pareto(a=3.0, size=50) + 10.0
    res_skew = engine.compare_continuous_cohorts(skew_a, skew_b)
    assert res_skew.test_type == HypothesisTestType.MANN_WHITNEY_U
    assert res_skew.is_significant is True
