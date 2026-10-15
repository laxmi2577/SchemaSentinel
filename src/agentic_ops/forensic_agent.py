from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional
import polars as pl

from src.agentic_ops import (
    AgentConfig,
    AnomalyType,
    ForensicReport,
    InvestigationStep,
    InvestigationTurn,
)
from src.analytics_engine.olap import DuckDBWarehouse
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.serving.grammar import validate_sql_syntax


class ForensicInvestigationAgent:
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        warehouse: Optional[DuckDBWarehouse] = None,
        validator: Optional[SQLPolicyASTValidator] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.warehouse = warehouse or DuckDBWarehouse()
        self.validator = validator or SQLPolicyASTValidator()

    def execute_turn(
        self,
        turn_index: int,
        step: InvestigationStep,
        tenant_id: str,
        hypothesis: str,
        sql_query: str,
    ) -> InvestigationTurn:
        t0 = datetime.datetime.now()

        syntax_check = validate_sql_syntax(sql_query)
        is_cfg = syntax_check[0] if isinstance(syntax_check, tuple) else bool(syntax_check)
        policy_res = self.validator.validate(sql_query)
        is_policy = policy_res.is_valid and not policy_res.has_mutations

        summary: Dict[str, Any] = {}
        if is_cfg and is_policy:
            try:
                res_df = self.warehouse.query(sql_query)
                summary["row_count"] = len(res_df)
                if not res_df.is_empty():
                    summary["data"] = res_df.to_dicts()[:5]
                summary["status"] = "success"
            except Exception as ex:
                summary["status"] = "execution_error"
                summary["error"] = str(ex)
        else:
            summary["status"] = "validation_failure"
            summary["is_cfg"] = is_cfg
            summary["is_policy"] = is_policy

        latency_ms = round((datetime.datetime.now() - t0).total_seconds() * 1000, 2)

        return InvestigationTurn(
            turn_index=turn_index,
            step=step,
            hypothesis=hypothesis,
            sql_query=sql_query,
            result_summary=summary,
            latency_ms=latency_ms,
            is_valid_cfg=is_cfg,
            is_policy_compliant=is_policy,
        )

    def investigate(
        self,
        tenant_id: str,
        anomaly_score: float,
        anomaly_type_hint: Optional[AnomalyType] = None,
    ) -> ForensicReport:
        turns: List[InvestigationTurn] = []
        inv_id = f"inv_{uuid.uuid4().hex[:10]}"

        turn_scope = self.execute_turn(
            turn_index=1,
            step=InvestigationStep.SCOPE,
            tenant_id=tenant_id,
            hypothesis=f"Isolate affected schemas and query counts for tenant {tenant_id}.",
            sql_query=f"SELECT schema_id, COUNT(*) AS event_count, AVG(latency_ms) AS avg_lat, SUM(CASE WHEN is_error THEN 1 ELSE 0 END) AS errors FROM fact_query_events WHERE tenant_id = '{tenant_id}' GROUP BY schema_id;",
        )
        turns.append(turn_scope)

        turn_profile = self.execute_turn(
            turn_index=2,
            step=InvestigationStep.PROFILE,
            tenant_id=tenant_id,
            hypothesis=f"Profile latency distribution and anomaly density for tenant {tenant_id}.",
            sql_query=f"SELECT QUANTILE_CONT(latency_ms, 0.5) AS p50, QUANTILE_CONT(latency_ms, 0.99) AS p99, AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) AS anomaly_rate, AVG(ast_complexity) AS avg_complexity FROM fact_query_events WHERE tenant_id = '{tenant_id}';",
        )
        turns.append(turn_profile)

        turn_drill = self.execute_turn(
            turn_index=3,
            step=InvestigationStep.DRILL_DOWN,
            tenant_id=tenant_id,
            hypothesis=f"Identify high-latency users and heavy token consumers for tenant {tenant_id}.",
            sql_query=f"SELECT user_id, COUNT(*) AS cnt, AVG(latency_ms) AS user_lat, SUM(tokens_used) AS total_tokens FROM fact_query_events WHERE tenant_id = '{tenant_id}' GROUP BY user_id ORDER BY user_lat DESC LIMIT 3;",
        )
        turns.append(turn_drill)

        turn_cross = self.execute_turn(
            turn_index=4,
            step=InvestigationStep.CROSS_REFERENCE,
            tenant_id=tenant_id,
            hypothesis=f"Check dimensional schema versions and metadata for tenant {tenant_id}.",
            sql_query=f"SELECT schema_id, version, total_columns, table_count FROM dim_schemas WHERE tenant_id = '{tenant_id}' LIMIT 1;",
        )
        turns.append(turn_cross)

        profile_data = turn_profile.result_summary.get("data", [{}])[0] if turn_profile.result_summary.get("data") else {}
        avg_complexity = float(profile_data.get("avg_complexity", 1.0))
        p99 = float(profile_data.get("p99", 50.0))

        if anomaly_type_hint is not None:
            inferred_type = anomaly_type_hint
            if anomaly_type_hint == AnomalyType.CARTESIAN_JOIN:
                root_cause = f"High AST complexity ({avg_complexity:.1f}) detected indicating unindexed joins or Cartesian product."
                recs = [
                    "Apply tenant column index on fact/dimension join keys",
                    "Inject mandatory WHERE clause predicates to bound table scan",
                ]
                requires_migration = False
            elif anomaly_type_hint == AnomalyType.LATENCY_SPIKE:
                root_cause = f"Severe p99 latency degradation ({p99:.1f}ms) breaching 100ms SLA."
                recs = [
                    "Scale compute warehouse or allocate dedicated tenant workers",
                    "Quarantine recurring slow queries into bounded buffer",
                ]
                requires_migration = False
            else:
                root_cause = "Subtle query distribution shift with schema evolution mismatch."
                recs = [
                    "Trigger automated self-healing SQL rewrite",
                    "Re-sync catalog schema version in DuckDB warehouse",
                ]
                requires_migration = True
        elif avg_complexity > 5.0:
            inferred_type = AnomalyType.CARTESIAN_JOIN
            root_cause = f"High AST complexity ({avg_complexity:.1f}) detected indicating unindexed joins or Cartesian product."
            recs = [
                "Apply tenant column index on fact/dimension join keys",
                "Inject mandatory WHERE clause predicates to bound table scan",
            ]
            requires_migration = False
        elif p99 > 150.0:
            inferred_type = AnomalyType.LATENCY_SPIKE
            root_cause = f"Severe p99 latency degradation ({p99:.1f}ms) breaching 100ms SLA."
            recs = [
                "Scale compute warehouse or allocate dedicated tenant workers",
                "Quarantine recurring slow queries into bounded buffer",
            ]
            requires_migration = False
        else:
            inferred_type = AnomalyType.SCHEMA_DRIFT
            root_cause = "Subtle query distribution shift with schema evolution mismatch."
            recs = [
                "Trigger automated self-healing SQL rewrite",
                "Re-sync catalog schema version in DuckDB warehouse",
            ]
            requires_migration = True

        turn_report = self.execute_turn(
            turn_index=5,
            step=InvestigationStep.REPORT,
            tenant_id=tenant_id,
            hypothesis="Synthesize investigation findings into root cause analysis.",
            sql_query=f"SELECT COUNT(*) AS total_anomalies FROM fact_query_events WHERE tenant_id = '{tenant_id}' AND is_anomaly = true;",
        )
        turns.append(turn_report)

        return ForensicReport(
            investigation_id=inv_id,
            tenant_id=tenant_id,
            anomaly_score=round(anomaly_score, 4),
            anomaly_type=inferred_type,
            root_cause=root_cause,
            confidence_score=0.92,
            turns=turns,
            recommendations=recs,
            requires_schema_migration=requires_migration,
            generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
