import pytest
from src.data_engine.formatter import PolicyRule, PolicyType
from src.evaluation.policy_ast import (
    PolicyViolation,
    PolicyValidationResult,
    SQLPolicyASTValidator,
    validate_sql_policy,
)


def test_read_only_allowed_queries():
    validator = SQLPolicyASTValidator()

    queries = [
        "SELECT id, name FROM users WHERE id = 1;",
        "SELECT dept_id, count(*), avg(salary) FROM employees GROUP BY dept_id HAVING count(*) > 5;",
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'COMPLETED';",
        "WITH high_earners AS (SELECT id, salary FROM employees WHERE salary > 100000) SELECT * FROM high_earners;",
        "SELECT * FROM products WHERE price > (SELECT avg(price) FROM products);",
    ]

    for q in queries:
        result = validator.validate(q)
        assert result.is_valid is True
        assert result.has_mutations is False
        assert len(result.violations) == 0
        assert validator.assert_read_only(q) is True


def test_mutation_rejection_direct():
    validator = SQLPolicyASTValidator()

    destructive_queries = [
        "DROP TABLE users;",
        "DELETE FROM employees WHERE dept_id = 10;",
        "UPDATE accounts SET balance = balance + 1000 WHERE id = 42;",
        "INSERT INTO audit_logs (event) VALUES ('hacked');",
        "ALTER TABLE customers ADD COLUMN compromised TEXT;",
        "CREATE TABLE backdoors (cmd TEXT);",
    ]

    for q in destructive_queries:
        result = validator.validate(q)
        assert result.is_valid is False
        assert result.has_mutations is True
        assert len(result.violations) >= 1
        assert validator.assert_read_only(q) is False


def test_nested_mutation_rejection_in_cte():
    validator = SQLPolicyASTValidator()
    cte_mutation = (
        "WITH deleted_rows AS (DELETE FROM sessions WHERE expired = 1 RETURNING user_id) "
        "SELECT * FROM deleted_rows;"
    )

    result = validator.validate(cte_mutation, dialect="postgres")
    assert result.is_valid is False
    assert result.has_mutations is True


def test_mandatory_tenant_isolation_predicate():
    policy = PolicyRule(
        rule_id="TENANT_GUARD",
        rule_type=PolicyType.TENANT_ISOLATION,
        description="Mandatory tenant isolation on orders and customers.",
        target_tables=["orders", "customers"],
        required_predicates=["tenant_id = :tenant_id"],
    )
    validator = SQLPolicyASTValidator()

    leaked_query = "SELECT order_id, amount FROM orders WHERE amount > 100;"
    result_leak = validator.validate(leaked_query, policies=[policy])
    assert result_leak.is_valid is False
    assert result_leak.has_tenant_filter is False
    assert any(v.rule_type == PolicyType.TENANT_ISOLATION for v in result_leak.violations)

    isolated_query = (
        "SELECT order_id, amount FROM orders WHERE tenant_id = 'T-42' AND amount > 100;"
    )
    result_ok = validator.validate(isolated_query, policies=[policy])
    assert result_ok.is_valid is True
    assert result_ok.has_tenant_filter is True

    join_isolated_query = (
        "SELECT o.order_id, c.name FROM orders o "
        "JOIN customers c ON o.customer_id = c.id "
        "WHERE o.tenant_id = :tenant_id;"
    )
    result_join_ok = validator.validate(join_isolated_query, policies=[policy])
    assert result_join_ok.is_valid is True


def test_restricted_table_access():
    policy = PolicyRule(
        rule_id="BLOCK_SECRETS",
        rule_type=PolicyType.RESTRICTED_TABLE,
        description="Credentials table is restricted from query access.",
        target_tables=["credentials", "internal_salaries"],
    )
    validator = SQLPolicyASTValidator()

    blocked_query = "SELECT username, password_hash FROM credentials;"
    result = validator.validate(blocked_query, policies=[policy])
    assert result.is_valid is False
    assert any(v.rule_type == PolicyType.RESTRICTED_TABLE for v in result.violations)

    safe_query = "SELECT username, display_name FROM public_profiles;"
    result_safe = validator.validate(safe_query, policies=[policy])
    assert result_safe.is_valid is True


def test_forbidden_columns_access():
    policy = PolicyRule(
        rule_id="MASK_PII",
        rule_type=PolicyType.FORBIDDEN_COLUMNS,
        description="SSN and credit card tokens are prohibited from projections.",
        target_tables=["ssn", "credit_card_token"],
    )
    validator = SQLPolicyASTValidator()

    violation_query = "SELECT customer_id, ssn, email FROM customers WHERE customer_id = 5;"
    result = validator.validate(violation_query, policies=[policy])
    assert result.is_valid is False
    assert any(v.rule_type == PolicyType.FORBIDDEN_COLUMNS for v in result.violations)

    allowed_query = "SELECT customer_id, email FROM customers WHERE customer_id = 5;"
    result_allowed = validator.validate(allowed_query, policies=[policy])
    assert result_allowed.is_valid is True
