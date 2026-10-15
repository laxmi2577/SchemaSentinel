from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import duckdb
import polars as pl
from pydantic import BaseModel

from src.analytics_engine import OLAPConfig


class DuckDBWarehouse:
    def __init__(self, config: Optional[OLAPConfig] = None) -> None:
        self.config = config or OLAPConfig()
        self._init_connection()
        self.initialize_star_schema()

    def _init_connection(self) -> None:
        if self.config.db_path != ":memory:":
            Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(
            database=self.config.db_path,
            read_only=self.config.read_only,
        )
        self.con.execute(f"SET memory_limit='{self.config.memory_limit}';")
        self.con.execute(f"SET threads TO {self.config.threads};")

    def initialize_star_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_tenants (
                tenant_id VARCHAR PRIMARY KEY,
                tier VARCHAR DEFAULT 'starter',
                region VARCHAR DEFAULT 'us-east-1',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_schemas (
                schema_id VARCHAR PRIMARY KEY,
                tenant_id VARCHAR,
                database_name VARCHAR DEFAULT 'default_db',
                table_count INTEGER DEFAULT 1,
                total_columns INTEGER DEFAULT 5,
                version INTEGER DEFAULT 1
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_time_periods (
                period_key INTEGER PRIMARY KEY,
                date_day DATE,
                hour INTEGER,
                day_of_week INTEGER,
                is_weekend BOOLEAN
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_users (
                user_id VARCHAR PRIMARY KEY,
                tenant_id VARCHAR,
                role VARCHAR DEFAULT 'analyst',
                department VARCHAR DEFAULT 'engineering'
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_query_events (
                event_id VARCHAR PRIMARY KEY,
                tenant_id VARCHAR,
                schema_id VARCHAR,
                user_id VARCHAR,
                timestamp TIMESTAMP,
                latency_ms DOUBLE,
                is_error BOOLEAN,
                is_anomaly BOOLEAN,
                tokens_used INTEGER,
                tables_accessed INTEGER,
                ast_complexity DOUBLE
            );
            """
        )

    def register_gold_parquet_views(self, gold_dir: Optional[str] = None) -> Dict[str, str]:
        target_dir = Path(gold_dir or self.config.gold_parquet_dir)
        registered = {}
        if not target_dir.exists():
            return registered

        for subpath in target_dir.iterdir():
            if subpath.is_dir():
                table_name = f"gold_{subpath.name}"
                parquet_glob = str(subpath / "**" / "*.parquet").replace("\\", "/")
                view_sql = f"""
                    CREATE OR REPLACE VIEW {table_name} AS
                    SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning={str(self.config.enable_hive_partitioning).lower()});
                """
                try:
                    self.con.execute(view_sql)
                    registered[table_name] = parquet_glob
                except Exception:
                    pass

        return registered

    def insert_events(self, events: Union[pl.DataFrame, Sequence[Dict[str, Any]]]) -> int:
        if isinstance(events, list) and not events:
            return 0

        if isinstance(events, list):
            df = pl.DataFrame(events)
        else:
            df = events

        if df.is_empty():
            return 0

        tenants = (
            df.select("tenant_id")
            .unique()
            .with_columns(
                pl.lit("starter").alias("tier"),
                pl.lit("us-east-1").alias("region"),
                pl.lit(datetime.datetime.now()).alias("created_at"),
            )
        )
        self.con.register("tmp_tenants", tenants.to_arrow())
        self.con.execute(
            """
            INSERT INTO dim_tenants (tenant_id, tier, region, created_at)
            SELECT tenant_id, tier, region, created_at FROM tmp_tenants
            ON CONFLICT (tenant_id) DO NOTHING;
            """
        )
        self.con.unregister("tmp_tenants")

        schema_col = "schema_id" if "schema_id" in df.columns else "tenant_id"
        select_cols = [
            pl.col(schema_col).alias("schema_id"),
            pl.col("tenant_id"),
        ]
        if "tables_accessed" in df.columns:
            select_cols.append(pl.col("tables_accessed").cast(pl.Int32).alias("table_count"))
        else:
            select_cols.append(pl.lit(1).alias("table_count"))

        schemas = (
            df.select(select_cols)
            .unique(subset=["schema_id"])
            .with_columns(
                pl.lit("default_db").alias("database_name"),
                pl.lit(5).alias("total_columns"),
                pl.lit(1).alias("version"),
            )
        )
        self.con.register("tmp_schemas", schemas.to_arrow())
        self.con.execute(
            """
            INSERT INTO dim_schemas (schema_id, tenant_id, database_name, table_count, total_columns, version)
            SELECT schema_id, tenant_id, database_name, table_count, total_columns, version FROM tmp_schemas
            ON CONFLICT (schema_id) DO NOTHING;
            """
        )
        self.con.unregister("tmp_schemas")

        if "user_id" in df.columns:
            users = (
                df.select("user_id", "tenant_id")
                .unique()
                .with_columns(
                    pl.lit("analyst").alias("role"),
                    pl.lit("engineering").alias("department"),
                )
            )
            self.con.register("tmp_users", users.to_arrow())
            self.con.execute(
                """
                INSERT INTO dim_users (user_id, tenant_id, role, department)
                SELECT user_id, tenant_id, role, department FROM tmp_users
                ON CONFLICT (user_id) DO NOTHING;
                """
            )
            self.con.unregister("tmp_users")

        normalized_cols = []
        col_names = df.columns

        normalized_cols.append(
            pl.col("event_id").cast(pl.Utf8)
            if "event_id" in col_names
            else pl.Series("event_id", [f"evt_{i}" for i in range(len(df))])
        )
        normalized_cols.append(pl.col("tenant_id").cast(pl.Utf8))
        normalized_cols.append(
            pl.col("schema_id").cast(pl.Utf8)
            if "schema_id" in col_names
            else pl.col("tenant_id").alias("schema_id")
        )
        normalized_cols.append(
            pl.col("user_id").cast(pl.Utf8)
            if "user_id" in col_names
            else pl.lit("user_system").alias("user_id")
        )
        normalized_cols.append(
            pl.col("timestamp").cast(pl.Datetime)
            if "timestamp" in col_names
            else pl.lit(datetime.datetime.now()).alias("timestamp")
        )
        normalized_cols.append(
            pl.col("latency_ms").cast(pl.Float64)
            if "latency_ms" in col_names
            else pl.lit(10.0).alias("latency_ms")
        )
        normalized_cols.append(
            pl.col("is_error").cast(pl.Boolean)
            if "is_error" in col_names
            else pl.lit(False).alias("is_error")
        )
        normalized_cols.append(
            pl.col("is_anomaly").cast(pl.Boolean)
            if "is_anomaly" in col_names
            else pl.lit(False).alias("is_anomaly")
        )
        normalized_cols.append(
            pl.col("tokens_used").cast(pl.Int32)
            if "tokens_used" in col_names
            else pl.lit(20).alias("tokens_used")
        )
        normalized_cols.append(
            pl.col("tables_accessed").cast(pl.Int32)
            if "tables_accessed" in col_names
            else pl.lit(1).alias("tables_accessed")
        )
        normalized_cols.append(
            pl.col("ast_complexity").cast(pl.Float64)
            if "ast_complexity" in col_names
            else pl.lit(1.0).alias("ast_complexity")
        )

        fact_df = df.select(normalized_cols)
        self.con.register("tmp_facts", fact_df.to_arrow())
        self.con.execute(
            """
            INSERT INTO fact_query_events
            SELECT * FROM tmp_facts
            ON CONFLICT (event_id) DO NOTHING;
            """
        )
        self.con.unregister("tmp_facts")
        return len(df)

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> pl.DataFrame:
        if params:
            cur = self.con.execute(sql, params)
        else:
            cur = self.con.execute(sql)
        try:
            return cur.pl()
        except Exception:
            try:
                return pl.from_arrow(cur.arrow())
            except Exception:
                return pl.DataFrame()

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> None:
        if params:
            self.con.execute(sql, params)
        else:
            self.con.execute(sql)

    def get_star_schema_summary(self) -> Dict[str, int]:
        tables = [
            "dim_tenants",
            "dim_schemas",
            "dim_time_periods",
            "dim_users",
            "fact_query_events",
        ]
        summary = {}
        for tbl in tables:
            res = self.con.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()
            summary[tbl] = int(res[0]) if res else 0
        return summary

    def get_tenant_performance_matrix(
        self,
        tenant_id: Optional[str] = None,
    ) -> pl.DataFrame:
        sql = """
            SELECT
                f.tenant_id,
                t.tier,
                COUNT(f.event_id) AS total_queries,
                ROUND(AVG(f.latency_ms), 3) AS mean_latency_ms,
                ROUND(QUANTILE_CONT(f.latency_ms, 0.50), 3) AS p50_latency_ms,
                ROUND(QUANTILE_CONT(f.latency_ms, 0.90), 3) AS p90_latency_ms,
                ROUND(QUANTILE_CONT(f.latency_ms, 0.99), 3) AS p99_latency_ms,
                ROUND(AVG(CASE WHEN f.is_error THEN 1.0 ELSE 0.0 END), 4) AS error_rate,
                ROUND(AVG(CASE WHEN f.is_anomaly THEN 1.0 ELSE 0.0 END), 4) AS anomaly_rate,
                SUM(f.tokens_used) AS total_tokens,
                ROUND(AVG(f.ast_complexity), 2) AS avg_complexity
            FROM fact_query_events f
            LEFT JOIN dim_tenants t ON f.tenant_id = t.tenant_id
        """
        if tenant_id:
            sql += f" WHERE f.tenant_id = '{tenant_id}'"
        sql += " GROUP BY f.tenant_id, t.tier ORDER BY total_queries DESC;"
        return self.query(sql)

    def close(self) -> None:
        if hasattr(self, "con"):
            self.con.close()
