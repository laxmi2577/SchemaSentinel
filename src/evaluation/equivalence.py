from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field
from src.evaluation.runner import ExecutionResult, ExecutionStatus


class EquivalenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_agnostic: bool = True
    float_tolerance: float = 1e-4
    decimal_places: int = 4
    case_sensitive_strings: bool = True
    strip_strings: bool = True
    null_equivalent_to_empty: bool = False


class EquivalenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_equivalent: bool
    reason: Optional[str] = None
    predicted_row_count: int = 0
    gold_row_count: int = 0
    predicted_col_count: int = 0
    gold_col_count: int = 0
    mismatched_rows: List[Dict[str, Any]] = Field(default_factory=list)


def normalize_result_cell(
    val: Any,
    config: Optional[EquivalenceConfig] = None,
) -> Any:
    cfg = config or EquivalenceConfig()

    if val is None:
        return None

    if isinstance(val, bool):
        return 1 if val else 0

    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return str(val).lower()
        if isinstance(val, float) and val.is_integer():
            return int(val)
        if isinstance(val, float):
            return round(val, cfg.decimal_places)
        return val

    if isinstance(val, (bytes, bytearray)):
        return val.hex()

    if isinstance(val, str):
        cleaned = val.strip() if cfg.strip_strings else val
        if not cfg.case_sensitive_strings:
            cleaned = cleaned.lower()
        if cfg.null_equivalent_to_empty and cleaned == "":
            return None
        try:
            float_candidate = float(cleaned)
            if not (math.isnan(float_candidate) or math.isinf(float_candidate)):
                if float_candidate.is_integer():
                    return int(float_candidate)
                return round(float_candidate, cfg.decimal_places)
        except (ValueError, OverflowError):
            pass
        return cleaned

    return str(val)


def _cells_are_equal(c1: Any, c2: Any, float_tol: float) -> bool:
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        return False

    if isinstance(c1, (int, float)) and isinstance(c2, (int, float)):
        return math.isclose(float(c1), float(c2), abs_tol=float_tol, rel_tol=float_tol)

    return c1 == c2


def _rows_are_equal(r1: Sequence[Any], r2: Sequence[Any], float_tol: float) -> bool:
    if len(r1) != len(r2):
        return False
    return all(_cells_are_equal(a, b, float_tol) for a, b in zip(r1, r2))


class ResultSetComparator:
    def __init__(self, config: Optional[EquivalenceConfig] = None) -> None:
        self.config = config or EquivalenceConfig()

    def normalize_row(self, row: Sequence[Any]) -> Tuple[Any, ...]:
        return tuple(normalize_result_cell(cell, self.config) for cell in row)

    def normalize_table(self, table: Sequence[Sequence[Any]]) -> List[Tuple[Any, ...]]:
        return [self.normalize_row(r) for r in table]

    def compare(
        self,
        predicted_rows: Sequence[Sequence[Any]],
        gold_rows: Sequence[Sequence[Any]],
        predicted_columns: Optional[Sequence[str]] = None,
        gold_columns: Optional[Sequence[str]] = None,
    ) -> EquivalenceReport:
        norm_pred = self.normalize_table(predicted_rows)
        norm_gold = self.normalize_table(gold_rows)

        pred_r_count = len(norm_pred)
        gold_r_count = len(norm_gold)
        pred_c_count = len(norm_pred[0]) if norm_pred else (len(predicted_columns) if predicted_columns else 0)
        gold_c_count = len(norm_gold[0]) if norm_gold else (len(gold_columns) if gold_columns else 0)

        if pred_r_count != gold_r_count:
            return EquivalenceReport(
                is_equivalent=False,
                reason=f"Row count mismatch: predicted has {pred_r_count} rows, gold has {gold_r_count} rows",
                predicted_row_count=pred_r_count,
                gold_row_count=gold_r_count,
                predicted_col_count=pred_c_count,
                gold_col_count=gold_c_count,
            )

        if pred_r_count > 0 and pred_c_count != gold_c_count:
            return EquivalenceReport(
                is_equivalent=False,
                reason=f"Column count mismatch: predicted has {pred_c_count} columns, gold has {gold_c_count} columns",
                predicted_row_count=pred_r_count,
                gold_row_count=gold_r_count,
                predicted_col_count=pred_c_count,
                gold_col_count=gold_c_count,
            )

        if pred_r_count == 0 and gold_r_count == 0:
            return EquivalenceReport(
                is_equivalent=True,
                reason=None,
                predicted_row_count=0,
                gold_row_count=0,
                predicted_col_count=pred_c_count,
                gold_col_count=gold_c_count,
            )

        if not self.config.order_agnostic:
            for idx, (p_row, g_row) in enumerate(zip(norm_pred, norm_gold)):
                if not _rows_are_equal(p_row, g_row, self.config.float_tolerance):
                    return EquivalenceReport(
                        is_equivalent=False,
                        reason=f"Row mismatch at index {idx}: predicted {p_row} != gold {g_row}",
                        predicted_row_count=pred_r_count,
                        gold_row_count=gold_r_count,
                        predicted_col_count=pred_c_count,
                        gold_col_count=gold_c_count,
                        mismatched_rows=[{"index": idx, "predicted": list(p_row), "gold": list(g_row)}],
                    )
            return EquivalenceReport(
                is_equivalent=True,
                predicted_row_count=pred_r_count,
                gold_row_count=gold_r_count,
                predicted_col_count=pred_c_count,
                gold_col_count=gold_c_count,
            )

        unmatched_gold = list(norm_gold)
        missing_in_gold: List[Tuple[Any, ...]] = []

        for p_row in norm_pred:
            matched_idx = -1
            for g_idx, g_row in enumerate(unmatched_gold):
                if _rows_are_equal(p_row, g_row, self.config.float_tolerance):
                    matched_idx = g_idx
                    break

            if matched_idx != -1:
                unmatched_gold.pop(matched_idx)
            else:
                missing_in_gold.append(p_row)

        if missing_in_gold or unmatched_gold:
            mismatched = []
            for r in missing_in_gold[:5]:
                mismatched.append({"unexpected_in_predicted": list(r)})
            for r in unmatched_gold[:5]:
                mismatched.append({"missing_from_gold": list(r)})

            return EquivalenceReport(
                is_equivalent=False,
                reason="Multiset row difference detected between predicted and gold result sets",
                predicted_row_count=pred_r_count,
                gold_row_count=gold_r_count,
                predicted_col_count=pred_c_count,
                gold_col_count=gold_c_count,
                mismatched_rows=mismatched,
            )

        return EquivalenceReport(
            is_equivalent=True,
            predicted_row_count=pred_r_count,
            gold_row_count=gold_r_count,
            predicted_col_count=pred_c_count,
            gold_col_count=gold_c_count,
        )

    def compare_execution_results(
        self,
        predicted_result: ExecutionResult,
        gold_result: ExecutionResult,
    ) -> EquivalenceReport:
        if predicted_result.status != ExecutionStatus.SUCCESS:
            return EquivalenceReport(
                is_equivalent=False,
                reason=f"Predicted query failed with status {predicted_result.status.value}: {predicted_result.error_message}",
                predicted_row_count=0,
                gold_row_count=gold_result.row_count,
                predicted_col_count=0,
                gold_col_count=len(gold_result.columns),
            )

        if gold_result.status != ExecutionStatus.SUCCESS:
            return EquivalenceReport(
                is_equivalent=False,
                reason=f"Gold reference query failed with status {gold_result.status.value}: {gold_result.error_message}",
                predicted_row_count=predicted_result.row_count,
                gold_row_count=0,
                predicted_col_count=len(predicted_result.columns),
                gold_col_count=0,
            )

        return self.compare(
            predicted_rows=predicted_result.rows,
            gold_rows=gold_result.rows,
            predicted_columns=predicted_result.columns,
            gold_columns=gold_result.columns,
        )


def compare_result_sets(
    predicted_rows: Sequence[Sequence[Any]],
    gold_rows: Sequence[Sequence[Any]],
    config: Optional[EquivalenceConfig] = None,
) -> EquivalenceReport:
    comparator = ResultSetComparator(config=config)
    return comparator.compare(predicted_rows, gold_rows)
