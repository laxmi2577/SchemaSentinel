from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class CatalogSyncConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    gold_dir: str = "data/lakehouse/gold"
    silver_dir: str = "data/lakehouse/silver"
    bronze_dir: str = "data/lakehouse/bronze"
    db_path: str = ":memory:"
    enable_hive_partitioning: bool = True


class TableSyncRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    table_name: str
    source_path_glob: str
    row_count: int
    column_names: List[str]
    is_view: bool = True
    registered_at: float = Field(default_factory=time.time)


class CatalogSyncReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_tables_registered: int = 0
    registrations: Dict[str, TableSyncRegistration] = Field(default_factory=dict)
    sync_duration_ms: float = 0.0
    catalog_status: str = "healthy"


class DuckDBCatalogSync:
    def __init__(
        self,
        config: Optional[CatalogSyncConfig] = None,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        self.config = config or CatalogSyncConfig()
        if connection is not None:
            self._conn = connection
            self._owns_connection = False
        else:
            self._conn = duckdb.connect(self.config.db_path)
            self._owns_connection = True
        self.registrations: Dict[str, TableSyncRegistration] = {}

    def register_parquet_table(
        self,
        table_name: str,
        path_glob: str,
        as_view: bool = True,
    ) -> Optional[TableSyncRegistration]:
        normalized_glob = str(path_glob).replace("\\", "/")
        matching_files = list(Path(normalized_glob.split("/*")[0]).rglob("*.parquet"))
        if not matching_files:
            return None

        hive_flag = "true" if self.config.enable_hive_partitioning else "false"
        view_sql = (
            f"CREATE OR REPLACE VIEW {table_name} AS "
            f"SELECT * FROM read_parquet('{normalized_glob}', hive_partitioning={hive_flag})"
        )
        self._conn.execute(view_sql)

        count_res = self._conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        row_count = count_res[0] if count_res else 0

        desc_res = self._conn.execute(f"DESCRIBE {table_name}").fetchall()
        columns = [str(row[0]) for row in desc_res]

        registration = TableSyncRegistration(
            table_name=table_name,
            source_path_glob=normalized_glob,
            row_count=row_count,
            column_names=columns,
            is_view=as_view,
        )
        self.registrations[table_name] = registration
        return registration

    def sync_gold_catalog(self) -> Dict[str, TableSyncRegistration]:
        gold_path = Path(self.config.gold_dir)
        if not gold_path.exists():
            return {}

        results: Dict[str, TableSyncRegistration] = {}
        for entry in gold_path.iterdir():
            if entry.is_dir():
                parquet_files = list(entry.rglob("*.parquet"))
                if parquet_files:
                    glob_pattern = f"{entry}/**/*.parquet"
                    reg = self.register_parquet_table(entry.name, glob_pattern)
                    if reg:
                        results[entry.name] = reg
        return results

    def sync_silver_catalog(self) -> Optional[TableSyncRegistration]:
        silver_path = Path(self.config.silver_dir)
        if not silver_path.exists():
            return None
        parquet_files = list(silver_path.rglob("*.parquet"))
        if not parquet_files:
            return None
        glob_pattern = f"{silver_path}/**/*.parquet"
        return self.register_parquet_table("silver_events", glob_pattern)

    def sync_bronze_catalog(self) -> Optional[TableSyncRegistration]:
        bronze_path = Path(self.config.bronze_dir)
        if not bronze_path.exists():
            return None
        parquet_files = list(bronze_path.rglob("*.parquet"))
        if not parquet_files:
            return None
        glob_pattern = f"{bronze_path}/**/*.parquet"
        return self.register_parquet_table("bronze_events", glob_pattern)

    def sync_all(self) -> CatalogSyncReport:
        t0 = time.perf_counter()
        self.sync_bronze_catalog()
        self.sync_silver_catalog()
        self.sync_gold_catalog()
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        return CatalogSyncReport(
            total_tables_registered=len(self.registrations),
            registrations=dict(self.registrations),
            sync_duration_ms=duration_ms,
            catalog_status="healthy",
        )

    def query(self, sql: str) -> pl.DataFrame:
        rel = self._conn.execute(sql)
        return rel.pl()

    def execute(self, sql: str) -> Any:
        return self._conn.execute(sql)

    def get_registered_tables(self) -> List[str]:
        return list(self.registrations.keys())

    def close(self) -> None:
        if self._owns_connection and self._conn is not None:
            self._conn.close()
