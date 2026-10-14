import pytest
from src.data_engine.formatter import (
    ColumnSchema,
    TableSchema,
    DatabaseSchema,
    PolicyRule,
    PolicyType,
    DecisionType,
    AssistantResponse,
    SchemaFormatter,
)


def test_schema_ddl_generation():
    col1 = ColumnSchema(name="id", data_type="INTEGER", is_primary_key=True)
    col2 = ColumnSchema(
        name="user_id",
        data_type="INTEGER",
        is_foreign_key=True,
        foreign_target="users(id)",
    )
    col3 = ColumnSchema(name="tenant_id", data_type="VARCHAR(64)")

    table = TableSchema(
        name="orders",
        columns=[col1, col2, col3],
        primary_keys=["id"],
        foreign_keys={"user_id": "users(id)"},
    )

    schema = DatabaseSchema(db_id="retail_db", tables={"orders": table})
    formatter = SchemaFormatter()
    ddl = formatter.to_ddl(schema)

    assert "CREATE TABLE orders (" in ddl
    assert "id INTEGER PRIMARY KEY" in ddl
    assert "FOREIGN KEY (user_id) REFERENCES users(id)" in ddl


def test_policy_block_formatting():
    rule1 = PolicyRule(
        rule_id="R1",
        rule_type=PolicyType.READ_ONLY,
        description="Only SELECT queries are authorized.",
    )
    rule2 = PolicyRule(
        rule_id="R2",
        rule_type=PolicyType.TENANT_ISOLATION,
        description="Enforce tenant boundary filter.",
        target_tables=["orders", "users"],
        required_predicates=["tenant_id = :tenant_id"],
    )

    formatter = SchemaFormatter()
    block = formatter.to_policy_block([rule1, rule2])

    assert "[READ_ONLY]: Only SELECT queries are authorized." in block
    assert "[TENANT_ISOLATION]: Enforce tenant boundary filter." in block
    assert "Mandatory Predicates: tenant_id = :tenant_id" in block


def test_full_chatml_and_inference_prompt():
    col = ColumnSchema(name="account_id", data_type="INTEGER", is_primary_key=True)
    table = TableSchema(name="accounts", columns=[col], primary_keys=["account_id"])
    schema = DatabaseSchema(db_id="bank_db", tables={"accounts": table})

    rule = PolicyRule(
        rule_id="R_RO",
        rule_type=PolicyType.READ_ONLY,
        description="Read-only access.",
    )

    formatter = SchemaFormatter()
    inf_prompt = formatter.format_inference_prompt(
        schema=schema,
        policies=[rule],
        user_query="How many accounts are there?",
    )

    assert inf_prompt.startswith("<|im_start|>system")
    assert "<|im_start|>user\nHow many accounts are there?<|im_end|>" in inf_prompt
    assert inf_prompt.endswith("<|im_start|>assistant\n")

    response = AssistantResponse(
        decision=DecisionType.ALLOW,
        sql="SELECT COUNT(*) FROM accounts;",
        rationale="Counts all records in accounts table with read-only query.",
    )

    full_chatml = formatter.format_full_chatml(
        schema=schema,
        policies=[rule],
        user_query="How many accounts are there?",
        response=response,
    )

    assert "ALLOW\n```sql\nSELECT COUNT(*) FROM accounts;\n```" in full_chatml
    assert "EVIDENCE: Counts all records" in full_chatml
    assert full_chatml.endswith("<|im_end|>")


def test_extract_assistant_response_roundtrip():
    formatter = SchemaFormatter()
    raw_allow = (
        "ALLOW\n"
        "```sql\n"
        "SELECT id, tenant_id FROM orders WHERE tenant_id = 't-01';\n"
        "```\n"
        "EVIDENCE: Scoped strictly to specified tenant_id filter."
    )

    parsed_allow = formatter.extract_assistant_response(raw_allow)
    assert parsed_allow.decision == DecisionType.ALLOW
    assert parsed_allow.sql == "SELECT id, tenant_id FROM orders WHERE tenant_id = 't-01';"
    assert "Scoped strictly" in parsed_allow.rationale

    raw_refuse = (
        "REFUSE\n"
        "VIOLATIONS: READ_ONLY, TENANT_ISOLATION\n"
        "EVIDENCE: Attempted UPDATE mutation without required tenant predicate."
    )

    parsed_refuse = formatter.extract_assistant_response(raw_refuse)
    assert parsed_refuse.decision == DecisionType.REFUSE
    assert parsed_refuse.sql is None
    assert "READ_ONLY" in parsed_refuse.policy_violations
    assert "TENANT_ISOLATION" in parsed_refuse.policy_violations


def test_from_spider_dict_parsing():
    spider_db = {
        "table_names_original": ["department", "student"],
        "column_names_original": [
            [-1, "*"],
            [0, "dept_id"],
            [0, "dept_name"],
            [1, "stud_id"],
            [1, "stud_name"],
            [1, "dept_id"],
        ],
        "column_types": ["text", "number", "text", "number", "text", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[5, 1]],
    }

    formatter = SchemaFormatter()
    schema = formatter.from_spider_dict(spider_db, db_id="university")

    assert schema.db_id == "university"
    assert "department" in schema.tables
    assert "student" in schema.tables

    dept_table = schema.tables["department"]
    assert dept_table.primary_keys == ["dept_id"]

    student_table = schema.tables["student"]
    assert student_table.primary_keys == ["stud_id"]
    assert student_table.foreign_keys == {"dept_id": "department(dept_id)"}
