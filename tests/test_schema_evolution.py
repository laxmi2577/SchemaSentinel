from __future__ import annotations

import pytest

import src.stream_lakehouse as slh


def test_extract_table_and_columns():
    detector = slh.SchemaEvolutionDetector()
    ddl = "CREATE TABLE users (id INT, email TEXT, created_at TIMESTAMP);"
    table, cols = detector.extract_table_and_columns(ddl)

    assert table == "users"
    assert "id" in cols
    assert "email" in cols
    assert "created_at" in cols
    assert cols["id"] == "INT"


def test_diff_ddl_additive_column():
    detector = slh.SchemaEvolutionDetector()
    old_ddl = "CREATE TABLE customers (id INT, name TEXT);"
    new_ddl = "CREATE TABLE customers (id INT, name TEXT, credit_limit FLOAT);"

    diff = detector.diff_ddl(old_ddl, new_ddl)
    assert diff.table_name == "customers"
    assert diff.added_columns == ["credit_limit"]
    assert len(diff.dropped_columns) == 0
    assert len(diff.altered_columns) == 0
    assert diff.is_breaking is False
    assert "credit_limit" in diff.diff_summary


def test_diff_ddl_dropped_column():
    detector = slh.SchemaEvolutionDetector()
    old_ddl = "CREATE TABLE accounts (id INT, account_number TEXT, balance FLOAT);"
    new_ddl = "CREATE TABLE accounts (id INT, balance FLOAT);"

    diff = detector.diff_ddl(old_ddl, new_ddl)
    assert diff.table_name == "accounts"
    assert diff.dropped_columns == ["account_number"]
    assert diff.is_breaking is True
    assert "Dropped columns" in diff.diff_summary


def test_diff_ddl_altered_column_type():
    detector = slh.SchemaEvolutionDetector()
    old_ddl = "CREATE TABLE orders (id INT, amount TEXT);"
    new_ddl = "CREATE TABLE orders (id INT, amount DECIMAL);"

    diff = detector.diff_ddl(old_ddl, new_ddl)
    assert diff.table_name == "orders"
    assert "amount" in diff.altered_columns
    assert diff.altered_columns["amount"]["from"] == "TEXT"
    assert diff.altered_columns["amount"]["to"] == "DECIMAL"
    assert diff.is_breaking is True


def test_create_change_event():
    detector = slh.SchemaEvolutionDetector()
    old_ddl = "CREATE TABLE items (id INT, price FLOAT);"
    new_ddl = "CREATE TABLE items (id INT, price FLOAT, sku TEXT);"

    event = detector.create_change_event(
        tenant_id="tenant_10",
        old_ddl=old_ddl,
        new_ddl=new_ddl,
        previous_version="1.0",
        new_version="1.1",
    )
    assert event.tenant_id == "tenant_10"
    assert event.table_name == "items"
    assert event.change_type == "ADD_COLUMN"
    assert event.is_breaking is False
    assert event.new_schema_version == "1.1"
    assert "sku" in event.ast_diff_summary["added_columns"]


def test_detect_breaking_changes():
    detector = slh.SchemaEvolutionDetector()
    breaking_event = slh.SchemaChangeEvent(
        tenant_id="t1",
        table_name="tbl",
        change_type="DROP_COLUMN",
        raw_ddl="CREATE TABLE tbl (id INT);",
        is_breaking=True,
    )
    assert detector.detect_breaking_changes(breaking_event) is True

    non_breaking_event = slh.SchemaChangeEvent(
        tenant_id="t1",
        table_name="tbl",
        change_type="ADD_COLUMN",
        raw_ddl="CREATE TABLE tbl (id INT, val TEXT);",
        is_breaking=False,
    )
    assert detector.detect_breaking_changes(non_breaking_event) is False


def test_evaluate_silver_recomputation_trigger():
    detector = slh.SchemaEvolutionDetector()
    breaking_event = slh.SchemaChangeEvent(
        tenant_id="t_omega",
        table_name="orders",
        change_type="DROP_COLUMN",
        raw_ddl="CREATE TABLE orders (id INT);",
        is_breaking=True,
    )
    res_breaking = detector.evaluate_silver_recomputation_trigger(breaking_event)
    assert res_breaking["should_recompute_silver"] is True
    assert "requires full Silver layer recomputation" in res_breaking["reason"]

    additive_event = slh.SchemaChangeEvent(
        tenant_id="t_omega",
        table_name="orders",
        change_type="ADD_COLUMN",
        raw_ddl="CREATE TABLE orders (id INT, notes TEXT);",
        is_breaking=False,
    )
    res_additive = detector.evaluate_silver_recomputation_trigger(additive_event)
    assert res_additive["should_recompute_silver"] is False
    assert "operates forward-only" in res_additive["reason"]
