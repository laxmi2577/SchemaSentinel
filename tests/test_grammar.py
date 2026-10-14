import pytest
from src.serving.grammar import (
    SQL_EBNF_GRAMMAR,
    SQL_READ_ONLY_CFG,
    FORBIDDEN_MUTATION_KEYWORDS,
    SQLSyntaxReport,
    SQLGrammarValidator,
    load_lark_parser,
    validate_sql_syntax,
    is_read_only_syntax,
    extract_query_tables,
    extract_query_columns,
)


def test_grammar_constants_definitions():
    assert isinstance(SQL_EBNF_GRAMMAR, str)
    assert len(SQL_EBNF_GRAMMAR) > 500
    assert SQL_READ_ONLY_CFG == SQL_EBNF_GRAMMAR
    assert "start: query_stmt" in SQL_EBNF_GRAMMAR
    assert "select_query" in SQL_EBNF_GRAMMAR
    assert "from_clause" in SQL_EBNF_GRAMMAR
    assert "where_clause" in SQL_EBNF_GRAMMAR
    assert "join_clause" in SQL_EBNF_GRAMMAR

    assert "drop" in FORBIDDEN_MUTATION_KEYWORDS
    assert "delete" in FORBIDDEN_MUTATION_KEYWORDS
    assert "truncate" in FORBIDDEN_MUTATION_KEYWORDS
    assert "alter" in FORBIDDEN_MUTATION_KEYWORDS
    assert "update" in FORBIDDEN_MUTATION_KEYWORDS
    assert "insert" in FORBIDDEN_MUTATION_KEYWORDS


def test_load_lark_parser_instantiation():
    parser = load_lark_parser()
    try:
        import lark

        assert parser is not None
    except ImportError:
        assert parser is None


def test_validate_valid_read_only_queries():
    queries = [
        "SELECT id, name FROM users WHERE id = 1;",
        "SELECT o.id, c.name FROM orders AS o JOIN customers AS c ON o.customer_id = c.id WHERE o.total > 100;",
        "SELECT region, SUM(revenue) FROM sales GROUP BY region HAVING SUM(revenue) > 500 ORDER BY SUM(revenue) DESC LIMIT 10;",
        "WITH active_users AS (SELECT id FROM users WHERE is_active = 1) SELECT * FROM active_users;",
        "SELECT id, CASE WHEN total > 100 THEN 'VIP' ELSE 'STANDARD' END AS tier FROM orders WHERE status IN ('PAID', 'SHIPPED') AND total BETWEEN 50 AND 500 AND notes LIKE '%urgent%' AND shipped_at IS NOT NULL;",
        "SELECT COUNT(DISTINCT customer_id) FROM transactions WHERE amount > 0;",
        "SELECT 1 AS probe;",
    ]

    for q in queries:
        is_valid, err = validate_sql_syntax(q)
        assert is_valid is True, f"Failed on valid query: {q}, error: {err}"
        assert is_read_only_syntax(q) is True


def test_validate_forbidden_mutation_statements():
    mutations = [
        "DROP TABLE users;",
        "DELETE FROM orders WHERE id = 10;",
        "UPDATE accounts SET balance = 0 WHERE id = 1;",
        "INSERT INTO audit_log (msg) VALUES ('breach');",
        "TRUNCATE TABLE customers;",
        "ALTER TABLE users ADD COLUMN is_admin INT;",
        "CREATE TABLE backdoor (id INT);",
        "SELECT * FROM users; DROP TABLE orders;",
    ]

    for m in mutations:
        is_valid, err = validate_sql_syntax(m)
        assert is_valid is False, f"Mutation query was erroneously validated as read-only: {m}"
        assert is_read_only_syntax(m) is False
        assert err is not None


def test_validate_malformed_syntax():
    malformed = [
        "",
        "   ",
        "SELECT FROM WHERE;",
        "SELECT , FROM table;",
        "JOIN orders ON;",
    ]

    for m in malformed:
        is_valid, err = validate_sql_syntax(m)
        assert is_valid is False
        assert err is not None


def test_extract_query_tables_and_columns():
    query = "SELECT u.id, u.email, o.total FROM users AS u JOIN orders AS o ON u.id = o.user_id WHERE u.status = 'ACTIVE';"

    tables = extract_query_tables(query)
    assert "users" in tables
    assert "orders" in tables

    columns = extract_query_columns(query)
    assert "id" in columns
    assert "email" in columns
    assert "total" in columns
    assert "status" in columns

    empty_tables = extract_query_tables("")
    assert empty_tables == []
    empty_cols = extract_query_columns("")
    assert empty_cols == []


def test_sql_grammar_validator_class():
    validator = SQLGrammarValidator()

    valid_sql = "SELECT id, title FROM books WHERE year >= 2000 ORDER BY title ASC LIMIT 5;"
    report = validator.validate(valid_sql)

    assert isinstance(report, SQLSyntaxReport)
    assert report.is_valid is True
    assert report.is_read_only is True
    assert report.has_forbidden_mutations is False
    assert report.error_message is None
    assert "books" in report.tables
    assert "id" in report.columns or "title" in report.columns

    assert validator.is_valid(valid_sql) is True
    assert "books" in validator.extract_tables(valid_sql)
    assert "title" in validator.extract_columns(valid_sql)

    invalid_sql = "UPDATE books SET price = 0;"
    bad_report = validator.validate(invalid_sql)
    assert bad_report.is_valid is False
    assert bad_report.is_read_only is False
    assert bad_report.has_forbidden_mutations is True
    assert "Prohibited mutation" in bad_report.error_message
    assert validator.is_valid(invalid_sql) is False
