from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PolicyType(str, Enum):
    READ_ONLY = "READ_ONLY"
    TENANT_ISOLATION = "TENANT_ISOLATION"
    RESTRICTED_TABLE = "RESTRICTED_TABLE"
    FORBIDDEN_COLUMNS = "FORBIDDEN_COLUMNS"


class DecisionType(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"


class ColumnSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_type: str = "TEXT"
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_target: Optional[str] = None
    description: Optional[str] = None


class TableSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    columns: List[ColumnSchema]
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: Dict[str, str] = Field(default_factory=dict)


class DatabaseSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    db_id: str
    tables: Dict[str, TableSchema]
    description: Optional[str] = None


class PolicyRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    rule_type: PolicyType
    description: str
    target_tables: List[str] = Field(default_factory=list)
    required_predicates: List[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: DecisionType
    sql: Optional[str] = None
    rationale: str
    policy_violations: List[str] = Field(default_factory=list)


class SchemaFormatter:
    SYSTEM_PREAMBLE: str = (
        "You are SchemaSentinel, a secure, policy-aware database assistant. "
        "Your task is to generate deterministic, read-only SQL queries that strictly adhere to all database schemas "
        "and governance access policies. If the query cannot be executed without violating policy or schema, refuse it."
    )

    IM_START: str = "<|im_start|>"
    IM_END: str = "<|im_end|>"

    def to_ddl(self, schema: DatabaseSchema) -> str:
        statements: List[str] = []
        for table_name, table in schema.tables.items():
            col_defs: List[str] = []
            for col in table.columns:
                parts = [col.name, col.data_type]
                if col.is_primary_key and len(table.primary_keys) <= 1:
                    parts.append("PRIMARY KEY")
                col_defs.append("  " + " ".join(parts))

            if len(table.primary_keys) > 1:
                col_defs.append(f"  PRIMARY KEY ({', '.join(table.primary_keys)})")

            for src_col, target in table.foreign_keys.items():
                col_defs.append(f"  FOREIGN KEY ({src_col}) REFERENCES {target}")

            ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
            statements.append(ddl)
        return "\n\n".join(statements)

    def to_policy_block(self, policies: List[PolicyRule]) -> str:
        if not policies:
            return "- [READ_ONLY]: Only SELECT statements are permitted. Mutations and DDL are strictly forbidden."
        lines: List[str] = []
        for policy in policies:
            details = [f"- [{policy.rule_type.value}]: {policy.description}"]
            if policy.target_tables:
                details.append(f"  Applicable Tables: {', '.join(policy.target_tables)}")
            if policy.required_predicates:
                details.append(f"  Mandatory Predicates: {', '.join(policy.required_predicates)}")
            lines.append("\n".join(details))
        return "\n".join(lines)

    def format_prompt_context(
        self,
        schema: DatabaseSchema,
        policies: List[PolicyRule],
        user_query: str,
    ) -> str:
        ddl_text = self.to_ddl(schema)
        policy_text = self.to_policy_block(policies)

        system_content = (
            f"{self.SYSTEM_PREAMBLE}\n\n"
            f"DATABASE SCHEMA ({schema.db_id}):\n{ddl_text}\n\n"
            f"GOVERNANCE POLICIES:\n{policy_text}"
        )

        prompt = (
            f"{self.IM_START}system\n{system_content}{self.IM_END}\n"
            f"{self.IM_START}user\n{user_query.strip()}{self.IM_END}\n"
        )
        return prompt

    def format_completion(self, response: AssistantResponse) -> str:
        lines: List[str] = [response.decision.value]
        if response.decision == DecisionType.ALLOW and response.sql:
            cleaned_sql = response.sql.strip().rstrip(";") + ";"
            lines.append("```sql")
            lines.append(cleaned_sql)
            lines.append("```")
        elif response.decision == DecisionType.REFUSE and response.policy_violations:
            violations_str = ", ".join(response.policy_violations)
            lines.append(f"VIOLATIONS: {violations_str}")

        lines.append(f"EVIDENCE: {response.rationale.strip()}")
        return "\n".join(lines)

    def format_full_chatml(
        self,
        schema: DatabaseSchema,
        policies: List[PolicyRule],
        user_query: str,
        response: AssistantResponse,
    ) -> str:
        prompt = self.format_prompt_context(schema, policies, user_query)
        completion = self.format_completion(response)
        return f"{prompt}{self.IM_START}assistant\n{completion}{self.IM_END}"

    def format_inference_prompt(
        self,
        schema: DatabaseSchema,
        policies: List[PolicyRule],
        user_query: str,
    ) -> str:
        prompt = self.format_prompt_context(schema, policies, user_query)
        return f"{prompt}{self.IM_START}assistant\n"

    def from_spider_dict(self, db_meta: Dict[str, Any], db_id: str) -> DatabaseSchema:
        raw_tables: List[str] = db_meta.get("table_names_original", [])
        raw_cols: List[List[Any]] = db_meta.get("column_names_original", [])
        raw_types: List[str] = db_meta.get("column_types", [])
        raw_pks: List[int] = db_meta.get("primary_keys", [])
        raw_fks: List[List[int]] = db_meta.get("foreign_keys", [])

        tables: Dict[str, TableSchema] = {}
        pk_set = set(raw_pks)

        fk_map: Dict[int, int] = {}
        for src_idx, dest_idx in raw_fks:
            fk_map[src_idx] = dest_idx

        for t_idx, t_name in enumerate(raw_tables):
            cols: List[ColumnSchema] = []
            pk_names: List[str] = []
            fk_dict: Dict[str, str] = {}

            for c_idx, (col_t_idx, col_name) in enumerate(raw_cols):
                if col_t_idx != t_idx:
                    continue
                col_type = raw_types[c_idx].upper() if c_idx < len(raw_types) else "TEXT"
                is_pk = c_idx in pk_set
                if is_pk:
                    pk_names.append(col_name)

                is_fk = c_idx in fk_map
                fk_target = None
                if is_fk:
                    dest_c_idx = fk_map[c_idx]
                    if dest_c_idx < len(raw_cols):
                        dest_t_idx, dest_col_name = raw_cols[dest_c_idx]
                        if dest_t_idx >= 0 and dest_t_idx < len(raw_tables):
                            dest_t_name = raw_tables[dest_t_idx]
                            fk_target = f"{dest_t_name}({dest_col_name})"
                            fk_dict[col_name] = fk_target

                cols.append(
                    ColumnSchema(
                        name=col_name,
                        data_type=col_type,
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        foreign_target=fk_target,
                    )
                )

            tables[t_name] = TableSchema(
                name=t_name,
                columns=cols,
                primary_keys=pk_names,
                foreign_keys=fk_dict,
            )

        return DatabaseSchema(db_id=db_id, tables=tables)

    def from_bird_dict(self, db_meta: Dict[str, Any], db_id: str) -> DatabaseSchema:
        return self.from_spider_dict(db_meta, db_id)

    def extract_assistant_response(self, raw_text: str) -> AssistantResponse:
        text = raw_text.strip()
        decision = DecisionType.ALLOW if text.startswith("ALLOW") else DecisionType.REFUSE

        sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        sql = sql_match.group(1).strip() if sql_match else None

        violations_match = re.search(r"VIOLATIONS:\s*(.*?)(?=\nEVIDENCE:|\Z)", text, re.DOTALL)
        violations: List[str] = []
        if violations_match:
            raw_v = violations_match.group(1).strip()
            violations = [v.strip() for v in raw_v.split(",") if v.strip()]

        evidence_match = re.search(r"EVIDENCE:\s*(.*?)\Z", text, re.DOTALL)
        rationale = evidence_match.group(1).strip() if evidence_match else ""

        if not rationale and not sql and not violations:
            rationale = text

        return AssistantResponse(
            decision=decision,
            sql=sql,
            rationale=rationale,
            policy_violations=violations,
        )
