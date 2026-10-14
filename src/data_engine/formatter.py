from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from src.data_engine.schema_parser import (
    DatabaseCatalog,
    TableMetadata,
    ColumnMetadata,
    ASTSchemaParser,
)


class PolicyType(str, Enum):
    READ_ONLY = "READ_ONLY"
    TENANT_ISOLATION = "TENANT_ISOLATION"
    RESTRICTED_TABLE = "RESTRICTED_TABLE"
    FORBIDDEN_COLUMNS = "FORBIDDEN_COLUMNS"
    NO_NESTED_MUTATION = "NO_NESTED_MUTATION"


class DecisionType(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"


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


class ChatMLMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class ChatMLDialogue(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: List[ChatMLMessage]


class ColumnSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    data_type: str = "TEXT"
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_target: Optional[str] = None
    foreign_target_table: Optional[str] = None
    foreign_target_column: Optional[str] = None
    default_value: Optional[str] = None
    description: Optional[str] = None


class TableSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    columns: Any
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: Dict[str, str] = Field(default_factory=dict)
    raw_ddl: Optional[str] = None

    def get_columns_dict(self) -> Dict[str, Any]:
        if isinstance(self.columns, dict):
            return self.columns
        if isinstance(self.columns, list):
            return {c.name: c for c in self.columns}
        return {}

    def get_column(self, col_name: str) -> Optional[Any]:
        return self.get_columns_dict().get(col_name)

    def has_column(self, col_name: str) -> bool:
        return col_name in self.get_columns_dict()

    def column_names(self) -> List[str]:
        return list(self.get_columns_dict().keys())

    def to_ddl(self) -> str:
        col_lines: List[str] = []
        cols = self.get_columns_dict()
        for col in cols.values():
            tokens = [f"  {col.name}", col.data_type]
            if not col.is_nullable:
                tokens.append("NOT NULL")
            if col.is_primary_key and len(self.primary_keys) <= 1:
                tokens.append("PRIMARY KEY")
            if col.default_value is not None:
                tokens.append(f"DEFAULT {col.default_value}")
            col_lines.append(" ".join(tokens))

        if len(self.primary_keys) > 1:
            col_lines.append(f"  PRIMARY KEY ({', '.join(self.primary_keys)})")

        for col_name, target in self.foreign_keys.items():
            col_lines.append(f"  FOREIGN KEY ({col_name}) REFERENCES {target}")

        body = ",\n".join(col_lines)
        return f"CREATE TABLE {self.name} (\n{body}\n);"


class DatabaseSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    db_id: str
    tables: Dict[str, Any]
    dialect: str = "sqlite"

    def get_table(self, table_name: str) -> Optional[Any]:
        return self.tables.get(table_name)

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def table_names(self) -> List[str]:
        return list(self.tables.keys())

    def to_ddl(self) -> str:
        return "\n\n".join(table.to_ddl() for table in self.tables.values())

    def to_schema_prompt_str(self) -> str:
        sections: List[str] = []
        for table in self.tables.values():
            cols = table.get_columns_dict() if hasattr(table, "get_columns_dict") else table.columns
            col_descs = [f"{c.name} ({c.data_type})" for c in cols.values()]
            desc = f"Table {table.name}: {', '.join(col_descs)}"
            if table.primary_keys:
                desc += f" | Primary Keys: {', '.join(table.primary_keys)}"
            if table.foreign_keys:
                fks = [f"{k}->{v}" for k, v in table.foreign_keys.items()]
                desc += f" | Foreign Keys: {', '.join(fks)}"
            sections.append(desc)
        return "\n".join(sections)


class ChatMLFormatter:
    SYSTEM_PREAMBLE: str = (
        "You are SchemaSentinel, a secure, policy-aware database assistant. "
        "Your task is to generate deterministic, read-only SQL queries that strictly adhere to all database schemas "
        "and governance access policies. If the query cannot be executed without violating policy or schema, refuse it."
    )

    IM_START: str = "<|im_start|>"
    IM_END: str = "<|im_end|>"

    def format_policies_block(self, policies: List[PolicyRule]) -> str:
        if not policies:
            return "- [READ_ONLY]: Only SELECT queries are permitted. Data mutations and DDL statements are strictly forbidden."

        lines: List[str] = []
        for policy in policies:
            entry = [f"- [{policy.rule_type.value}]: {policy.description}"]
            if policy.target_tables:
                entry.append(f"  Applicable Tables: {', '.join(policy.target_tables)}")
            if policy.required_predicates:
                entry.append(f"  Mandatory Predicates: {', '.join(policy.required_predicates)}")
            lines.append("\n".join(entry))
        return "\n".join(lines)

    def format_system_message(
        self,
        catalog: Optional[Any] = None,
        policies: Optional[List[PolicyRule]] = None,
        schema: Optional[Any] = None,
    ) -> str:
        cat = catalog if catalog is not None else schema
        pols = policies or []
        policy_text = self.format_policies_block(pols)
        if cat is None:
            if not policy_text:
                return self.SYSTEM_PREAMBLE
            return f"{self.SYSTEM_PREAMBLE}\n\nGOVERNANCE POLICIES:\n{policy_text}"

        ddl_text = cat.to_ddl() if hasattr(cat, "to_ddl") else str(cat)
        db_id = getattr(cat, "db_id", "default")
        return (
            f"{self.SYSTEM_PREAMBLE}\n\n"
            f"DATABASE SCHEMA ({db_id}):\n{ddl_text}\n\n"
            f"GOVERNANCE POLICIES:\n{policy_text}"
        )

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

    def format_prompt_context(
        self,
        catalog: Optional[Any] = None,
        policies: Optional[List[PolicyRule]] = None,
        user_query: str = "",
        schema: Optional[Any] = None,
    ) -> str:
        system_content = self.format_system_message(catalog=catalog, policies=policies, schema=schema)
        return (
            f"{self.IM_START}system\n{system_content}{self.IM_END}\n"
            f"{self.IM_START}user\n{user_query.strip()}{self.IM_END}\n"
        )

    def format_full_dialogue(
        self,
        catalog: Optional[Any] = None,
        policies: Optional[List[PolicyRule]] = None,
        user_query: str = "",
        response: Optional[AssistantResponse] = None,
        schema: Optional[Any] = None,
    ) -> str:
        prompt = self.format_prompt_context(catalog=catalog, policies=policies, user_query=user_query, schema=schema)
        completion = self.format_completion(response) if response is not None else ""
        return f"{prompt}{self.IM_START}assistant\n{completion}{self.IM_END}"

    def format_inference_prompt(
        self,
        catalog: Optional[Any] = None,
        policies: Optional[List[PolicyRule]] = None,
        user_query: str = "",
        schema: Optional[Any] = None,
    ) -> str:
        prompt = self.format_prompt_context(catalog=catalog, policies=policies, user_query=user_query, schema=schema)
        return f"{prompt}{self.IM_START}assistant\n"

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

    def to_ddl(self, catalog: Any) -> str:
        return catalog.to_ddl()

    def to_policy_block(self, policies: List[PolicyRule]) -> str:
        return self.format_policies_block(policies)

    def format_full_chatml(
        self,
        catalog: Optional[Any] = None,
        policies: Optional[List[PolicyRule]] = None,
        user_query: str = "",
        response: Optional[AssistantResponse] = None,
        schema: Optional[Any] = None,
    ) -> str:
        return self.format_full_dialogue(
            catalog=catalog,
            policies=policies,
            user_query=user_query,
            response=response,
            schema=schema,
        )

    def from_spider_dict(self, spider_dict: Dict[str, Any], db_id: str = "") -> DatabaseSchema:
        parser = ASTSchemaParser()
        cat = parser.from_spider_tables_dict(spider_dict, db_id=db_id)
        schema_tables: Dict[str, TableSchema] = {}
        for t_name, t_meta in cat.tables.items():
            schema_cols: Dict[str, ColumnSchema] = {}
            for c_name, c_meta in t_meta.columns.items():
                schema_cols[c_name] = ColumnSchema(
                    name=c_meta.name,
                    data_type=c_meta.data_type,
                    is_nullable=c_meta.is_nullable,
                    is_primary_key=c_meta.is_primary_key,
                    is_foreign_key=c_meta.is_foreign_key,
                    foreign_target=t_meta.foreign_keys.get(c_name),
                    foreign_target_table=c_meta.foreign_target_table,
                    foreign_target_column=c_meta.foreign_target_column,
                    default_value=c_meta.default_value,
                    description=c_meta.description,
                )
            schema_tables[t_name] = TableSchema(
                name=t_name,
                columns=schema_cols,
                primary_keys=list(t_meta.primary_keys),
                foreign_keys=dict(t_meta.foreign_keys),
                raw_ddl=t_meta.raw_ddl,
            )
        return DatabaseSchema(db_id=cat.db_id, tables=schema_tables, dialect=cat.dialect)


SchemaFormatter = ChatMLFormatter
