from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import sqlglot
from sqlglot import exp

import src.stream_lakehouse as slh


class DDLDiff(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_name: str
    added_columns: List[str] = Field(default_factory=list)
    dropped_columns: List[str] = Field(default_factory=list)
    altered_columns: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    is_breaking: bool = False
    diff_summary: str = ""


class SchemaEvolutionDetector:
    def __init__(self) -> None:
        pass

    def extract_table_and_columns(self, ddl: str) -> Tuple[str, Dict[str, str]]:
        parsed = sqlglot.parse_one(ddl)
        table_name = ""
        columns: Dict[str, str] = {}

        if isinstance(parsed, exp.Create):
            this_table = parsed.this
            if isinstance(this_table, exp.Schema):
                table_name = this_table.this.name
                for col_def in this_table.expressions:
                    if isinstance(col_def, exp.ColumnDef):
                        col_name = col_def.name
                        col_type = col_def.kind.sql().upper() if col_def.kind else "TEXT"
                        columns[col_name] = col_type
            elif isinstance(this_table, exp.Table):
                table_name = this_table.name

        return table_name, columns

    def diff_ddl(self, old_ddl: str, new_ddl: str) -> DDLDiff:
        old_table, old_cols = self.extract_table_and_columns(old_ddl)
        new_table, new_cols = self.extract_table_and_columns(new_ddl)
        table_name = new_table or old_table

        added = [col for col in new_cols if col not in old_cols]
        dropped = [col for col in old_cols if col not in new_cols]
        altered: Dict[str, Dict[str, str]] = {}

        for col in old_cols:
            if col in new_cols and old_cols[col] != new_cols[col]:
                altered[col] = {"from": old_cols[col], "to": new_cols[col]}

        is_breaking = len(dropped) > 0 or len(altered) > 0
        summary_parts = []
        if added:
            summary_parts.append(f"Added columns: {', '.join(added)}")
        if dropped:
            summary_parts.append(f"Dropped columns: {', '.join(dropped)}")
        if altered:
            altered_desc = [f"{c} ({v['from']} -> {v['to']})" for c, v in altered.items()]
            summary_parts.append(f"Altered column types: {', '.join(altered_desc)}")

        summary = "; ".join(summary_parts) if summary_parts else "No structural column changes"

        return DDLDiff(
            table_name=table_name,
            added_columns=added,
            dropped_columns=dropped,
            altered_columns=altered,
            is_breaking=is_breaking,
            diff_summary=summary,
        )

    def create_change_event(
        self,
        tenant_id: str,
        old_ddl: str,
        new_ddl: str,
        previous_version: str = "1.0",
        new_version: str = "1.1",
    ) -> slh.SchemaChangeEvent:
        diff = self.diff_ddl(old_ddl, new_ddl)
        change_type = "ALTER_SCHEMA"
        if diff.dropped_columns and not diff.added_columns:
            change_type = "DROP_COLUMN"
        elif diff.added_columns and not diff.dropped_columns:
            change_type = "ADD_COLUMN"
        elif diff.altered_columns:
            change_type = "ALTER_TYPE"

        return slh.SchemaChangeEvent(
            tenant_id=tenant_id,
            table_name=diff.table_name,
            change_type=change_type,
            timestamp=time.time(),
            raw_ddl=new_ddl,
            previous_schema_version=previous_version,
            new_schema_version=new_version,
            ast_diff_summary=diff.model_dump(),
            is_breaking=diff.is_breaking,
        )

    def detect_breaking_changes(self, change_event: slh.SchemaChangeEvent) -> bool:
        return change_event.is_breaking

    def evaluate_silver_recomputation_trigger(
        self,
        change_event: slh.SchemaChangeEvent,
    ) -> Dict[str, Any]:
        should_recompute = change_event.is_breaking
        reason = (
            f"Breaking schema drift on table '{change_event.table_name}' requires full Silver layer recomputation."
            if should_recompute
            else f"Additive backwards-compatible schema drift on table '{change_event.table_name}' operates forward-only."
        )
        return {
            "should_recompute_silver": should_recompute,
            "table_name": change_event.table_name,
            "version": change_event.new_schema_version,
            "change_type": change_event.change_type,
            "reason": reason,
        }
