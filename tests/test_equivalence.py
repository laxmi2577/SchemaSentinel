import pytest
from src.evaluation.runner import ExecutionResult, ExecutionStatus
from src.evaluation.equivalence import (
    EquivalenceConfig,
    EquivalenceReport,
    ResultSetComparator,
    normalize_result_cell,
    compare_result_sets,
)


def test_normalize_result_cell_types():
    assert normalize_result_cell(None) is None
    assert normalize_result_cell(True) == 1
    assert normalize_result_cell(False) == 0
    assert normalize_result_cell(42) == 42
    assert normalize_result_cell(42.0) == 42
    assert normalize_result_cell(12.34567) == 12.3457
    assert normalize_result_cell("  hello  ") == "hello"
    assert normalize_result_cell("123") == 123
    assert normalize_result_cell("12.50") == 12.5


def test_order_agnostic_multiset_matching():
    comparator = ResultSetComparator(EquivalenceConfig(order_agnostic=True))

    predicted = [
        ["Alice", 100],
        ["Bob", 200],
        ["Charlie", 300],
    ]
    gold = [
        ["Charlie", 300],
        ["Alice", 100],
        ["Bob", 200],
    ]

    report = comparator.compare(predicted, gold)
    assert report.is_equivalent is True
    assert report.predicted_row_count == 3
    assert report.gold_row_count == 3
    assert report.reason is None


def test_order_sensitive_mismatch():
    comparator = ResultSetComparator(EquivalenceConfig(order_agnostic=False))

    predicted = [
        ["Alice", 100],
        ["Bob", 200],
    ]
    gold = [
        ["Bob", 200],
        ["Alice", 100],
    ]

    report = comparator.compare(predicted, gold)
    assert report.is_equivalent is False
    assert "Row mismatch at index 0" in (report.reason or "")
    assert len(report.mismatched_rows) == 1


def test_float_tolerance_matching():
    comparator = ResultSetComparator(
        EquivalenceConfig(float_tolerance=1e-3, decimal_places=4)
    )

    predicted = [["Metric", 45.1234]]
    gold = [["Metric", 45.1238]]

    report = comparator.compare(predicted, gold)
    assert report.is_equivalent is True

    predicted_far = [["Metric", 45.12]]
    gold_far = [["Metric", 45.99]]
    report_far = comparator.compare(predicted_far, gold_far)
    assert report_far.is_equivalent is False


def test_row_count_and_column_count_mismatch():
    comparator = ResultSetComparator()

    pred_fewer = [["A", 1]]
    gold_more = [["A", 1], ["B", 2]]
    report_r = comparator.compare(pred_fewer, gold_more)
    assert report_r.is_equivalent is False
    assert "Row count mismatch" in (report_r.reason or "")

    pred_wide = [["A", 1, "extra"]]
    gold_narrow = [["A", 1]]
    report_c = comparator.compare(pred_wide, gold_narrow)
    assert report_c.is_equivalent is False
    assert "Column count mismatch" in (report_c.reason or "")


def test_compare_execution_results():
    comparator = ResultSetComparator()

    res_success_1 = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        columns=["user", "points"],
        rows=[["u1", 50], ["u2", 80]],
        row_count=2,
    )
    res_success_2 = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        columns=["user", "points"],
        rows=[["u2", 80], ["u1", 50]],
        row_count=2,
    )
    res_error = ExecutionResult(
        status=ExecutionStatus.SYNTAX_ERROR,
        columns=[],
        rows=[],
        row_count=0,
        error_message="no such table",
    )

    report_ok = comparator.compare_execution_results(res_success_1, res_success_2)
    assert report_ok.is_equivalent is True

    report_err = comparator.compare_execution_results(res_error, res_success_1)
    assert report_err.is_equivalent is False
    assert "SYNTAX_ERROR" in (report_err.reason or "")
