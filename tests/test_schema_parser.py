import pytest
from src.data_engine.schema_parser import (
    ColumnMetadata,
    TableMetadata,
    DatabaseCatalog,
    ASTSchemaParser,
)


def test_parse_ddl_single_table():
    ddl = """
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY,
        username VARCHAR(64) NOT NULL,
        email TEXT,
        is_active BOOLEAN DEFAULT 1
    );
    """
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.parse_ddl(ddl, db_id="auth_db")

    assert catalog.db_id == "auth_db"
    assert catalog.has_table("users")

    table = catalog.get_table("users")
    assert table is not None
    assert table.name == "users"
    assert table.primary_keys == ["user_id"]
    assert table.has_column("username")
    assert table.has_column("email")
    assert table.has_column("is_active")

    user_col = table.get_column("user_id")
    assert user_col is not None
    assert user_col.is_primary_key is True

    username_col = table.get_column("username")
    assert username_col is not None
    assert username_col.is_nullable is False


def test_parse_ddl_multiple_tables_with_foreign_keys():
    ddl = """
    CREATE TABLE departments (
        dept_id INTEGER PRIMARY KEY,
        dept_name TEXT NOT NULL
    );

    CREATE TABLE employees (
        emp_id INTEGER PRIMARY KEY,
        emp_name TEXT NOT NULL,
        dept_id INTEGER,
        FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
    );
    """
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.parse_ddl(ddl, db_id="hr_db")

    assert len(catalog.tables) == 2
    assert catalog.has_table("departments")
    assert catalog.has_table("employees")

    emp_table = catalog.get_table("employees")
    assert emp_table is not None
    assert "dept_id" in emp_table.foreign_keys
    assert emp_table.foreign_keys["dept_id"] == "departments(dept_id)"

    dept_col = emp_table.get_column("dept_id")
    assert dept_col is not None
    assert dept_col.is_foreign_key is True
    assert dept_col.foreign_target_table == "departments"
    assert dept_col.foreign_target_column == "dept_id"

    assert parser.validate_catalog(catalog) is True


def test_parse_composite_primary_key():
    ddl = """
    CREATE TABLE project_assignments (
        project_id INTEGER,
        employee_id INTEGER,
        role TEXT,
        PRIMARY KEY (project_id, employee_id)
    );
    """
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.parse_ddl(ddl, db_id="projects_db")

    table = catalog.get_table("project_assignments")
    assert table is not None
    assert table.primary_keys == ["project_id", "employee_id"]

    p_col = table.get_column("project_id")
    e_col = table.get_column("employee_id")
    assert p_col is not None and p_col.is_primary_key is True
    assert e_col is not None and e_col.is_primary_key is True


def test_from_spider_tables_dict():
    spider_meta = {
        "table_names_original": ["author", "book"],
        "column_names_original": [
            [-1, "*"],
            [0, "author_id"],
            [0, "author_name"],
            [1, "book_id"],
            [1, "title"],
            [1, "author_id"],
        ],
        "column_types": ["text", "number", "text", "number", "text", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[5, 1]],
    }
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.from_spider_tables_dict(spider_meta, db_id="library_db")

    assert catalog.db_id == "library_db"
    assert catalog.has_table("author")
    assert catalog.has_table("book")

    author_table = catalog.get_table("author")
    book_table = catalog.get_table("book")
    assert author_table is not None
    assert book_table is not None

    assert author_table.primary_keys == ["author_id"]
    assert book_table.primary_keys == ["book_id"]
    assert book_table.foreign_keys == {"author_id": "author(author_id)"}

    assert parser.validate_catalog(catalog) is True


def test_validate_catalog_broken_foreign_key():
    broken_table = TableMetadata(
        name="orders",
        columns={
            "order_id": ColumnMetadata(name="order_id", is_primary_key=True),
            "customer_id": ColumnMetadata(name="customer_id", is_foreign_key=True),
        },
        primary_keys=["order_id"],
        foreign_keys={"customer_id": "missing_customers(id)"},
    )
    broken_catalog = DatabaseCatalog(db_id="broken_db", tables={"orders": broken_table})
    parser = ASTSchemaParser()

    assert parser.validate_catalog(broken_catalog) is False


def test_catalog_to_ddl_and_prompt_formatting():
    col1 = ColumnMetadata(name="id", data_type="INTEGER", is_primary_key=True)
    col2 = ColumnMetadata(name="name", data_type="TEXT", is_nullable=False)
    table = TableMetadata(name="categories", columns={"id": col1, "name": col2}, primary_keys=["id"])
    catalog = DatabaseCatalog(db_id="inventory", tables={"categories": table})

    ddl = catalog.to_ddl()
    assert "CREATE TABLE categories (" in ddl
    assert "id INTEGER PRIMARY KEY" in ddl
    assert "name TEXT NOT NULL" in ddl

    prompt_str = catalog.to_schema_prompt_str()
    assert "Table categories:" in prompt_str
    assert "id (INTEGER)" in prompt_str
    assert "Primary Keys: id" in prompt_str
