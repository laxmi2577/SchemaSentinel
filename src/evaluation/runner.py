from __future__ import annotations

import sqlite3
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field
from src.data_engine.schema_parser import DatabaseCatalog


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    MUTATION_DENIED = "MUTATION_DENIED"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"


class SandboxConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeout_seconds: float = 3.0
    max_rows: int = 1000
    max_bytes: int = 5 * 1024 * 1024
    enforce_read_only: bool = True
    opcode_check_interval: int = 1000


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ExecutionStatus
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    is_truncated: bool = False


class SQLiteSandbox:
    FORBIDDEN_ACTION_CODES: Set[int] = {
        getattr(sqlite3, "SQLITE_INSERT", 18),
        getattr(sqlite3, "SQLITE_UPDATE", 23),
        getattr(sqlite3, "SQLITE_DELETE", 9),
        getattr(sqlite3, "SQLITE_DROP_TABLE", 11),
        getattr(sqlite3, "SQLITE_CREATE_TABLE", 2),
        getattr(sqlite3, "SQLITE_ALTER_TABLE", 26),
        getattr(sqlite3, "SQLITE_ATTACH", 24),
        getattr(sqlite3, "SQLITE_DETACH", 25),
        getattr(sqlite3, "SQLITE_DROP_INDEX", 10),
        getattr(sqlite3, "SQLITE_CREATE_INDEX", 1),
        getattr(sqlite3, "SQLITE_DROP_VIEW", 17),
        getattr(sqlite3, "SQLITE_CREATE_VIEW", 8),
        getattr(sqlite3, "SQLITE_DROP_TRIGGER", 16),
        getattr(sqlite3, "SQLITE_CREATE_TRIGGER", 7),
    }

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()

    def create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.isolation_level = None
        return conn

    def seed_schema(self, conn: sqlite3.Connection, ddl: str) -> None:
        conn.executescript(ddl)

    def seed_from_catalog(self, conn: sqlite3.Connection, catalog: DatabaseCatalog) -> None:
        ddl = catalog.to_ddl()
        self.seed_schema(conn, ddl)

    def seed_data(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        records: List[Dict[str, Any]],
    ) -> None:
        if not records:
            return

        columns = list(records[0].keys())
        col_names = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

        values = [[row.get(c) for c in columns] for row in records]
        conn.executemany(insert_sql, values)

    def _authorizer_callback(
        self,
        action_code: int,
        arg1: Optional[str],
        arg2: Optional[str],
        db_name: Optional[str],
        trigger_name: Optional[str],
    ) -> int:
        if self.config.enforce_read_only and action_code in self.FORBIDDEN_ACTION_CODES:
            return getattr(sqlite3, "SQLITE_DENY", 1)
        return getattr(sqlite3, "SQLITE_OK", 0)

    def execute_query(
        self,
        conn: sqlite3.Connection,
        query: str,
    ) -> ExecutionResult:
        start_time = time.perf_counter()
        timeout_occurred = False

        def progress_handler() -> int:
            nonlocal timeout_occurred
            elapsed = time.perf_counter() - start_time
            if elapsed > self.config.timeout_seconds:
                timeout_occurred = True
                return 1
            return 0

        conn.set_progress_handler(progress_handler, self.config.opcode_check_interval)
        if self.config.enforce_read_only:
            conn.set_authorizer(self._authorizer_callback)

        cursor = conn.cursor()
        try:
            cursor.execute(query)
            description = cursor.description
            columns = [desc[0] for desc in description] if description else []

            collected_rows: List[List[Any]] = []
            total_bytes = 0
            is_truncated = False

            while True:
                row = cursor.fetchone()
                if row is None:
                    break

                if len(collected_rows) >= self.config.max_rows:
                    is_truncated = True
                    break

                row_list = list(row)
                row_bytes = sys.getsizeof(row_list)
                total_bytes += row_bytes

                if total_bytes > self.config.max_bytes:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return ExecutionResult(
                        status=ExecutionStatus.MEMORY_LIMIT_EXCEEDED,
                        columns=columns,
                        rows=collected_rows,
                        row_count=len(collected_rows),
                        execution_time_ms=round(elapsed_ms, 3),
                        error_message=f"Result set exceeded maximum byte buffer ({self.config.max_bytes} bytes)",
                        is_truncated=True,
                    )

                collected_rows.append(row_list)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                columns=columns,
                rows=collected_rows,
                row_count=len(collected_rows),
                execution_time_ms=round(elapsed_ms, 3),
                error_message=None,
                is_truncated=is_truncated,
            )

        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(e)
            if timeout_occurred or "interrupted" in err_msg.lower():
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    execution_time_ms=round(elapsed_ms, 3),
                    error_message=f"Query execution timed out after {self.config.timeout_seconds}s",
                )
            if "syntax" in err_msg.lower() or "no such" in err_msg.lower():
                return ExecutionResult(
                    status=ExecutionStatus.SYNTAX_ERROR,
                    execution_time_ms=round(elapsed_ms, 3),
                    error_message=err_msg,
                )
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                execution_time_ms=round(elapsed_ms, 3),
                error_message=err_msg,
            )

        except sqlite3.DatabaseError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(e)
            if "not authorized" in err_msg.lower():
                return ExecutionResult(
                    status=ExecutionStatus.MUTATION_DENIED,
                    execution_time_ms=round(elapsed_ms, 3),
                    error_message="Unauthorized mutation or administrative operation denied by sandbox policy",
                )
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                execution_time_ms=round(elapsed_ms, 3),
                error_message=err_msg,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                execution_time_ms=round(elapsed_ms, 3),
                error_message=str(e),
            )

        finally:
            try:
                conn.set_progress_handler(None, 0)
                conn.set_authorizer(None)
                cursor.close()
            except Exception:
                pass

    def execute_isolated(
        self,
        query: str,
        ddl: Optional[str] = None,
        seed_records: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        catalog: Optional[DatabaseCatalog] = None,
    ) -> ExecutionResult:
        conn = self.create_connection()
        try:
            if catalog is not None:
                self.seed_from_catalog(conn, catalog)
            elif ddl is not None:
                self.seed_schema(conn, ddl)

            if seed_records:
                for tbl_name, recs in seed_records.items():
                    self.seed_data(conn, tbl_name, recs)

            return self.execute_query(conn, query)
        finally:
            conn.close()


def execute_query_isolated(
    query: str,
    ddl: Optional[str] = None,
    seed_records: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    catalog: Optional[DatabaseCatalog] = None,
    config: Optional[SandboxConfig] = None,
) -> ExecutionResult:
    sandbox = SQLiteSandbox(config=config)
    return sandbox.execute_isolated(
        query=query,
        ddl=ddl,
        seed_records=seed_records,
        catalog=catalog,
    )
