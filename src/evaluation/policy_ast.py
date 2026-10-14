from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from src.data_engine.formatter import PolicyRule, PolicyType


class PolicyViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_type: PolicyType
    rule_id: Optional[str] = None
    message: str
    offending_node: Optional[str] = None


class PolicyValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_valid: bool
    violations: List[PolicyViolation] = Field(default_factory=list)
    extracted_tables: List[str] = Field(default_factory=list)
    extracted_columns: List[str] = Field(default_factory=list)
    has_mutations: bool = False
    has_tenant_filter: bool = True


class SQLPolicyASTValidator:
    FORBIDDEN_MUTATION_TYPES: Tuple[type, ...] = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
        exp.Pragma,
    )

    def __init__(
        self,
        default_dialect: str = "sqlite",
        tenant_column: Optional[str] = None,
        mandatory_tenant_check: bool = False,
    ) -> None:
        self.default_dialect = default_dialect
        self.tenant_column = tenant_column
        self.mandatory_tenant_check = mandatory_tenant_check

    def parse_statements(self, sql: str, dialect: Optional[str] = None) -> List[exp.Expression]:
        target_dialect = dialect or self.default_dialect
        return [stmt for stmt in sqlglot.parse(sql, read=target_dialect) if stmt is not None]

    def extract_table_names(self, statement: exp.Expression) -> List[str]:
        tables: Set[str] = set()
        for tbl in statement.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name.lower())
        return sorted(list(tables))

    def extract_column_names(self, statement: exp.Expression) -> List[str]:
        cols: Set[str] = set()
        for col in statement.find_all(exp.Column):
            if col.name:
                cols.add(col.name.lower())
        return sorted(list(cols))

    def check_for_mutations(self, statement: exp.Expression) -> List[str]:
        mutations: List[str] = []
        if isinstance(statement, self.FORBIDDEN_MUTATION_TYPES):
            mutations.append(statement.key.upper())

        for forbidden_cls in self.FORBIDDEN_MUTATION_TYPES:
            for node in statement.find_all(forbidden_cls):
                if node != statement:
                    mutations.append(f"NESTED_{node.key.upper()}")

        return mutations

    def has_tenant_predicate(
        self,
        statement: exp.Expression,
        tenant_column: str = "tenant_id",
    ) -> bool:
        target_col = tenant_column.lower()

        for col in statement.find_all(exp.Column):
            if col.name and col.name.lower() == target_col:
                parent = col.find_ancestor(exp.Where, exp.Join)
                if parent is not None:
                    return True
        return False

    def validate(
        self,
        sql: str,
        policies: Optional[List[PolicyRule]] = None,
        dialect: Optional[str] = None,
    ) -> PolicyValidationResult:
        target_dialect = dialect or self.default_dialect
        violations: List[PolicyViolation] = []

        try:
            statements = self.parse_statements(sql, dialect=target_dialect)
        except ParseError as e:
            violations.append(
                PolicyViolation(
                    rule_type=PolicyType.READ_ONLY,
                    message=f"SQL syntax parse failure: {str(e)}",
                    offending_node="SYNTAX_ERROR",
                )
            )
            return PolicyValidationResult(
                is_valid=False,
                violations=violations,
                has_mutations=False,
                has_tenant_filter=False,
            )

        if not statements:
            violations.append(
                PolicyViolation(
                    rule_type=PolicyType.READ_ONLY,
                    message="Empty SQL string provided",
                    offending_node="EMPTY",
                )
            )
            return PolicyValidationResult(
                is_valid=False,
                violations=violations,
                has_mutations=False,
                has_tenant_filter=False,
            )

        all_tables: Set[str] = set()
        all_cols: Set[str] = set()
        has_mutations = False
        active_policies = list(policies) if policies is not None else [
            PolicyRule(
                rule_id="DEFAULT_RO",
                rule_type=PolicyType.READ_ONLY,
                description="Default read-only policy.",
            )
        ]
        if self.mandatory_tenant_check and not any(p.rule_type == PolicyType.TENANT_ISOLATION for p in active_policies):
            t_col = self.tenant_column or "tenant_id"
            active_policies.append(
                PolicyRule(
                    rule_id="TENANT_AUTO",
                    rule_type=PolicyType.TENANT_ISOLATION,
                    description="Mandatory tenant isolation.",
                    required_predicates=[f"{t_col} = :tenant_val"],
                )
            )

        for stmt in statements:
            tables = self.extract_table_names(stmt)
            cols = self.extract_column_names(stmt)
            all_tables.update(tables)
            all_cols.update(cols)

            mutations = self.check_for_mutations(stmt)
            if mutations:
                has_mutations = True
                for mut in mutations:
                    violations.append(
                        PolicyViolation(
                            rule_type=PolicyType.READ_ONLY,
                            message=f"Unauthorized mutation or administrative statement detected: {mut}",
                            offending_node=mut,
                        )
                    )

            for policy in active_policies:
                if policy.rule_type == PolicyType.RESTRICTED_TABLE:
                    forbidden_tables = [t.lower() for t in policy.target_tables]
                    for t in tables:
                        if t in forbidden_tables:
                            violations.append(
                                PolicyViolation(
                                    rule_type=PolicyType.RESTRICTED_TABLE,
                                    rule_id=policy.rule_id,
                                    message=f"Access to restricted table '{t}' is denied",
                                    offending_node=t,
                                )
                            )

                elif policy.rule_type == PolicyType.FORBIDDEN_COLUMNS:
                    forbidden_columns = [c.lower() for c in policy.target_tables]
                    for c in cols:
                        if c in forbidden_columns:
                            violations.append(
                                PolicyViolation(
                                    rule_type=PolicyType.FORBIDDEN_COLUMNS,
                                    rule_id=policy.rule_id,
                                    message=f"Access to forbidden column '{c}' is denied",
                                    offending_node=c,
                                )
                            )

                elif policy.rule_type == PolicyType.TENANT_ISOLATION:
                    target_scope = [t.lower() for t in policy.target_tables]
                    overlaps_target = any(t in target_scope for t in tables) if target_scope else True

                    if overlaps_target:
                        tenant_col = "tenant_id"
                        if policy.required_predicates:
                            first_pred = policy.required_predicates[0].lower()
                            match = re.match(r"^(\w+)\s*=", first_pred)
                            if match:
                                tenant_col = match.group(1)

                        if not self.has_tenant_predicate(stmt, tenant_column=tenant_col):
                            violations.append(
                                PolicyViolation(
                                    rule_type=PolicyType.TENANT_ISOLATION,
                                    rule_id=policy.rule_id,
                                    message=f"Missing mandatory tenant isolation predicate on column '{tenant_col}'",
                                    offending_node=tenant_col,
                                )
                            )

        is_valid = len(violations) == 0
        has_tenant = not any(v.rule_type == PolicyType.TENANT_ISOLATION for v in violations)

        return PolicyValidationResult(
            is_valid=is_valid,
            violations=violations,
            extracted_tables=sorted(list(all_tables)),
            extracted_columns=sorted(list(all_cols)),
            has_mutations=has_mutations,
            has_tenant_filter=has_tenant,
        )

    def assert_read_only(self, sql: str, dialect: Optional[str] = None) -> bool:
        result = self.validate(
            sql=sql,
            policies=[
                PolicyRule(
                    rule_id="RO_ASSERT",
                    rule_type=PolicyType.READ_ONLY,
                    description="Read-only assertion",
                )
            ],
            dialect=dialect,
        )
        return not result.has_mutations and result.is_valid

    def validate_query(
        self,
        sql: str,
        policies: Optional[List[PolicyRule]] = None,
        dialect: Optional[str] = None,
    ) -> PolicyValidationResult:
        return self.validate(sql=sql, policies=policies, dialect=dialect)


def validate_sql_policy(
    sql: str,
    policies: Optional[List[PolicyRule]] = None,
    dialect: str = "sqlite",
) -> PolicyValidationResult:
    validator = SQLPolicyASTValidator(default_dialect=dialect)
    return validator.validate(sql=sql, policies=policies, dialect=dialect)
