from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sqlglot
from sqlglot import exp

from src.agentic_ops import AgentConfig, SelfHealAttempt, SelfHealStatus
from src.analytics_engine.olap import DuckDBWarehouse
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.serving.grammar import validate_sql_syntax
from src.stream_lakehouse.dlq_router import DLQRouter


class SelfHealingSQLLoop:
    DEFAULT_COLUMN_MAP: Dict[str, str] = {
        "user": "user_id",
        "tenant": "tenant_id",
        "schema": "schema_id",
        "latency": "latency_ms",
        "time": "timestamp",
        "created_date": "timestamp",
        "created_at": "timestamp",
        "error": "is_error",
        "anomaly": "is_anomaly",
        "tokens": "tokens_used",
        "complexity": "ast_complexity",
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        warehouse: Optional[DuckDBWarehouse] = None,
        validator: Optional[SQLPolicyASTValidator] = None,
        dlq_router: Optional[DLQRouter] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.warehouse = warehouse or DuckDBWarehouse()
        self.validator = validator or SQLPolicyASTValidator()
        self.dlq_router = dlq_router

    def parse_error_details(self, error_message: str) -> Tuple[str, Optional[str]]:
        msg = error_message.lower()

        col_patterns = [
            r"no such column:?\s*([a-zA-Z0-9_]+)",
            r'column\s+"?([a-zA-Z0-9_]+)"?\s+does not exist',
            r'referenced column\s+"?([a-zA-Z0-9_]+)"?\s+not found',
            r"column\s+not\s+found:\s*([a-zA-Z0-9_]+)",
        ]
        for pat in col_patterns:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return "COLUMN_NOT_FOUND", m.group(1).lower()

        tbl_patterns = [
            r"no such table:?\s*([a-zA-Z0-9_]+)",
            r'table\s+"?([a-zA-Z0-9_]+)"?\s+does not exist',
            r'table with name\s+"?([a-zA-Z0-9_]+)"?\s+does not exist',
        ]
        for pat in tbl_patterns:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return "TABLE_NOT_FOUND", m.group(1).lower()

        if "syntax" in msg or "parser" in msg:
            return "SYNTAX_ERROR", None

        return "UNKNOWN_DRIFT", None

    def rewrite_column_in_query(
        self,
        sql: str,
        missing_column: str,
        replacement_column: str,
    ) -> str:
        try:
            tree = sqlglot.parse_one(sql)

            def transform(node: exp.Expression) -> exp.Expression:
                if isinstance(node, exp.Column) and node.name.lower() == missing_column.lower():
                    return exp.Column(this=exp.to_identifier(replacement_column))
                return node

            transformed = tree.transform(transform)
            return transformed.sql()
        except Exception:
            pattern = re.compile(rf"\b{re.escape(missing_column)}\b", re.IGNORECASE)
            return pattern.sub(replacement_column, sql)

    def rewrite_table_in_query(
        self,
        sql: str,
        missing_table: str,
        replacement_table: str,
    ) -> str:
        try:
            tree = sqlglot.parse_one(sql)

            def transform(node: exp.Expression) -> exp.Expression:
                if isinstance(node, exp.Table) and node.name.lower() == missing_table.lower():
                    return exp.Table(this=exp.to_identifier(replacement_table))
                return node

            transformed = tree.transform(transform)
            return transformed.sql()
        except Exception:
            pattern = re.compile(rf"\b{re.escape(missing_table)}\b", re.IGNORECASE)
            return pattern.sub(replacement_table, sql)

    def heal_query(
        self,
        failing_sql: str,
        error_message: str,
        tenant_id: str,
        catalog_schemas: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[SelfHealStatus, Optional[str], List[SelfHealAttempt]]:
        attempts: List[SelfHealAttempt] = []
        drift_type, target_entity = self.parse_error_details(error_message)

        current_sql = failing_sql
        max_retries = self.config.max_self_heal_retries

        for attempt_idx in range(1, max_retries + 1):
            proposed_sql = current_sql
            ast_diff_desc = "No AST modification"

            if drift_type == "COLUMN_NOT_FOUND" and target_entity:
                if attempt_idx == 1:
                    replacement = self.DEFAULT_COLUMN_MAP.get(target_entity)
                    if not replacement and catalog_schemas:
                        for tbl_cols in catalog_schemas.values():
                            for c in tbl_cols:
                                if target_entity in c or c in target_entity:
                                    replacement = c
                                    break
                            if replacement:
                                break
                    replacement = replacement or "event_id"
                    proposed_sql = self.rewrite_column_in_query(current_sql, target_entity, replacement)
                    ast_diff_desc = f"Substituted missing column '{target_entity}' with catalog column '{replacement}'"
                elif attempt_idx == 2:
                    proposed_sql = self.rewrite_column_in_query(current_sql, target_entity, "1")
                    ast_diff_desc = f"Replaced missing column '{target_entity}' with constant literal 1"
                else:
                    proposed_sql = f"SELECT tenant_id, COUNT(*) AS count FROM fact_query_events WHERE tenant_id = '{tenant_id}' GROUP BY tenant_id;"
                    ast_diff_desc = "Fell back to canonical aggregated star-schema query"

            elif drift_type == "TABLE_NOT_FOUND" and target_entity:
                if attempt_idx == 1:
                    tbl_map = {
                        "events": "fact_query_events",
                        "queries": "fact_query_events",
                        "tenants": "dim_tenants",
                        "schemas": "dim_schemas",
                        "users": "dim_users",
                    }
                    replacement_table = tbl_map.get(target_entity, "fact_query_events")
                    proposed_sql = self.rewrite_table_in_query(current_sql, target_entity, replacement_table)
                    ast_diff_desc = f"Mapped missing table '{target_entity}' to warehouse table '{replacement_table}'"
                else:
                    proposed_sql = f"SELECT * FROM fact_query_events WHERE tenant_id = '{tenant_id}' LIMIT 10;"
                    ast_diff_desc = "Replaced query target with fact_query_events"
            else:
                proposed_sql = f"SELECT tenant_id, COUNT(*) AS query_volume FROM fact_query_events WHERE tenant_id = '{tenant_id}' GROUP BY tenant_id;"
                ast_diff_desc = "Synthesized safe standard query"

            syntax_check = validate_sql_syntax(proposed_sql)
            is_valid_syntax = syntax_check[0] if isinstance(syntax_check, tuple) else bool(syntax_check)
            policy_res = self.validator.validate(proposed_sql)
            is_policy_compliant = policy_res.is_valid and not policy_res.has_mutations

            exec_success = False
            exec_error = ""

            if is_valid_syntax and is_policy_compliant:
                try:
                    res_df = self.warehouse.query(proposed_sql)
                    exec_success = True
                except Exception as ex:
                    exec_error = str(ex)
                    exec_success = False
            else:
                exec_error = "Validation failure"

            attempts.append(
                SelfHealAttempt(
                    attempt_index=attempt_idx,
                    original_query=failing_sql,
                    error_message=exec_error if not exec_success else "",
                    detected_drift_type=drift_type,
                    ast_diff_summary=ast_diff_desc,
                    proposed_query=proposed_sql,
                    is_valid_syntax=is_valid_syntax,
                    is_policy_compliant=is_policy_compliant,
                    execution_success=exec_success,
                )
            )

            if exec_success:
                return SelfHealStatus.SUCCESS, proposed_sql, attempts

            current_sql = proposed_sql
            drift_type, target_entity = self.parse_error_details(exec_error)

        if self.dlq_router:
            try:
                import src.stream_lakehouse as slh

                mock_event = slh.StreamEvent(
                    tenant_id=tenant_id,
                    generated_sql=failing_sql,
                    is_valid=False,
                )
                if hasattr(self.dlq_router, "route"):
                    self.dlq_router.route(
                        event=mock_event,
                        failure_reason=f"Self-healing loop exhausted ({max_retries} attempts failed): {error_message}",
                        error_category="SELF_HEAL_EXHAUSTED",
                    )
                elif hasattr(self.dlq_router, "quarantine"):
                    self.dlq_router.quarantine(
                        event=mock_event,
                        reason=f"Self-healing loop exhausted ({max_retries} attempts failed): {error_message}",
                        error_class="SELF_HEAL_EXHAUSTED",
                    )
            except Exception:
                pass

        return SelfHealStatus.FAILED_ESCALATED, None, attempts
