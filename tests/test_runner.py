import pytest
from src.data_engine.schema_parser import (
    ColumnMetadata,
    TableMetadata,
    DatabaseCatalog,
)
from src.evaluation.runner import (
    ExecutionStatus,
    SandboxConfig,
    SQLiteSandbox,
    execute_query_isolated,
)


def test_successful_isolated_execution():
    ddl = """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL
    );
    """
    records = {
        "employees": [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000.0},
            {"id": 2, "name": "Bob", "department": "Marketing", "salary": 85000.0},
            {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 140000.0},
        ]
    }
    query = "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC;"

    result = execute_query_isolated(
        query=query,
        ddl=ddl,
        seed_records=records,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.columns == ["name", "salary"]
    assert result.row_count == 2
    assert result.rows == [["Charlie", 140000.0], ["Alice", 120000.0]]
    assert result.error_message is None
    assert result.execution_time_ms > 0.0
    assert result.is_truncated is False


def test_seeding_from_database_catalog():
    col1 = ColumnMetadata(name="sku", data_type="TEXT", is_primary_key=True)
    col2 = ColumnMetadata(name="stock", data_type="INTEGER")
    table = TableMetadata(name="inventory", columns={"sku": col1, "stock": col2}, primary_keys=["sku"])
    catalog = DatabaseCatalog(db_id="warehouse", tables={"inventory": table})

    sandbox = SQLiteSandbox()
    conn = sandbox.create_connection()
    try:
        sandbox.seed_from_catalog(conn, catalog)
        sandbox.seed_data(conn, "inventory", [{"sku": "A101", "stock": 50}, {"sku": "B202", "stock": 0}])

        result = sandbox.execute_query(conn, "SELECT COUNT(*) FROM inventory WHERE stock > 0;")
        assert result.status == ExecutionStatus.SUCCESS
        assert result.rows == [[1]]
    finally:
        conn.close()


def test_syntax_error_detection():
    ddl = "CREATE TABLE test (id INTEGER);"
    result = execute_query_isolated(
        query="SELECT non_existent_column FROM test;",
        ddl=ddl,
    )

    assert result.status == ExecutionStatus.SYNTAX_ERROR
    assert result.rows == []
    assert result.error_message is not None
    assert "no such column" in result.error_message.lower()


def test_mutation_denied_by_authorizer():
    ddl = "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);"
    seed = {"users": [{"id": 1, "email": "admin@corp.com"}]}

    queries = [
        "DROP TABLE users;",
        "DELETE FROM users WHERE id = 1;",
        "UPDATE users SET email = 'hacked@corp.com' WHERE id = 1;",
        "INSERT INTO users (id, email) VALUES (2, 'injected@corp.com');",
    ]

    for q in queries:
        result = execute_query_isolated(
            query=q,
            ddl=ddl,
            seed_records=seed,
            config=SandboxConfig(enforce_read_only=True),
        )
        assert result.status == ExecutionStatus.MUTATION_DENIED
        assert result.error_message is not None
        assert "unauthorized mutation" in result.error_message.lower()


def test_max_rows_truncation():
    ddl = "CREATE TABLE numbers (n INTEGER);"
    records = {"numbers": [{"n": i} for i in range(50)]}

    config = SandboxConfig(max_rows=10)
    result = execute_query_isolated(
        query="SELECT n FROM numbers ORDER BY n ASC;",
        ddl=ddl,
        seed_records=records,
        config=config,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.row_count == 10
    assert len(result.rows) == 10
    assert result.rows[0] == [0]
    assert result.rows[-1] == [9]
    assert result.is_truncated is True


def test_execution_timeout_guard():
    config = SandboxConfig(
        timeout_seconds=0.1,
        opcode_check_interval=50,
    )
    runaway_query = "WITH RECURSIVE loop(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM loop) SELECT count(*) FROM loop;"

    result = execute_query_isolated(
        query=runaway_query,
        config=config,
    )

    assert result.status == ExecutionStatus.TIMEOUT
    assert result.error_message is not None
    assert "timed out" in result.error_message.lower()
