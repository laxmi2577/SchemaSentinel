from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import sqlglot
from sqlglot import exp


class ColumnMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_type: str = "TEXT"
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_target_table: Optional[str] = None
    foreign_target_column: Optional[str] = None
    default_value: Optional[str] = None
    description: Optional[str] = None


class TableMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    columns: Dict[str, ColumnMetadata]
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: Dict[str, str] = Field(default_factory=dict)
    raw_ddl: Optional[str] = None

    def get_column(self, col_name: str) -> Optional[ColumnMetadata]:
        return self.columns.get(col_name)

    def has_column(self, col_name: str) -> bool:
        return col_name in self.columns

    def column_names(self) -> List[str]:
        return list(self.columns.keys())

    def to_ddl(self) -> str:
        col_lines: List[str] = []
        for col in self.columns.values():
            tokens = [f"  {col.name}", col.data_type]
            if not col.is_nullable:
                tokens.append("NOT NULL")
            if col.is_primary_key and len(self.primary_keys) <= 1:
                tokens.append("PRIMARY KEY")
            if col.default_value is not None:
                tokens.append(f"DEFAULT {col.default_value}")
            col_lines.append(" ".join(tokens))

        if len(self.primary_keys) > 1:
            col_lines.append(f"  PRIMARY KEY ({', '.join(self.primary_keys)})")

        for col_name, target in self.foreign_keys.items():
            col_lines.append(f"  FOREIGN KEY ({col_name}) REFERENCES {target}")

        body = ",\n".join(col_lines)
        return f"CREATE TABLE {self.name} (\n{body}\n);"


class DatabaseCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    db_id: str
    tables: Dict[str, TableMetadata]
    dialect: str = "sqlite"

    def get_table(self, table_name: str) -> Optional[TableMetadata]:
        return self.tables.get(table_name)

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def table_names(self) -> List[str]:
        return list(self.tables.keys())

    def to_ddl(self) -> str:
        return "\n\n".join(table.to_ddl() for table in self.tables.values())

    def to_schema_prompt_str(self) -> str:
        sections: List[str] = []
        for table in self.tables.values():
            col_descs = [f"{c.name} ({c.data_type})" for c in table.columns.values()]
            desc = f"Table {table.name}: {', '.join(col_descs)}"
            if table.primary_keys:
                desc += f" | Primary Keys: {', '.join(table.primary_keys)}"
            if table.foreign_keys:
                fks = [f"{k}->{v}" for k, v in table.foreign_keys.items()]
                desc += f" | Foreign Keys: {', '.join(fks)}"
            sections.append(desc)
        return "\n".join(sections)


class ASTSchemaParser:
    def __init__(self, default_dialect: str = "sqlite") -> None:
        self.default_dialect = default_dialect

    def parse_ddl(
        self,
        ddl_text: str,
        db_id: str = "default_db",
        dialect: Optional[str] = None,
    ) -> DatabaseCatalog:
        target_dialect = dialect or self.default_dialect
        statements = sqlglot.parse(ddl_text, read=target_dialect)
        tables: Dict[str, TableMetadata] = {}

        for statement in statements:
            if not statement or not isinstance(statement, exp.Create):
                continue

            schema_node = statement.this
            if not isinstance(schema_node, exp.Schema):
                continue

            table_node = schema_node.this
            table_name = table_node.name if hasattr(table_node, "name") else str(table_node)
            raw_statement_ddl = statement.sql(dialect=target_dialect)

            columns: Dict[str, ColumnMetadata] = {}
            primary_keys: List[str] = []
            foreign_keys: Dict[str, str] = {}

            for element in schema_node.expressions:
                if isinstance(element, exp.ColumnDef):
                    col_name = element.name
                    data_type = element.kind.sql(dialect=target_dialect).upper() if element.kind else "TEXT"
                    is_pk = False
                    is_nullable = True
                    is_fk = False
                    target_table: Optional[str] = None
                    target_column: Optional[str] = None
                    default_val: Optional[str] = None

                    for constraint in element.constraints:
                        kind = constraint.kind
                        kind_name = kind.__class__.__name__
                        constraint_sql = constraint.sql(dialect=target_dialect).lower()

                        if isinstance(kind, exp.PrimaryKeyColumnConstraint) or kind_name == "PrimaryKeyColumnConstraint" or "primary key" in constraint_sql:
                            is_pk = True
                        if isinstance(kind, exp.NotNullColumnConstraint) or kind_name == "NotNullColumnConstraint" or "not null" in constraint_sql:
                            is_nullable = False
                        if isinstance(kind, exp.DefaultColumnConstraint) or kind_name == "DefaultColumnConstraint" or "default" in constraint_sql:
                            default_node = getattr(kind, "this", None) or getattr(constraint, "this", None)
                            if default_node:
                                default_val = default_node.sql(dialect=target_dialect)
                        if isinstance(kind, exp.Reference) or kind_name == "Reference":
                            is_fk = True
                            ref_node = kind.this
                            if isinstance(ref_node, exp.Schema):
                                ref_table_node = ref_node.this
                                target_table = ref_table_node.name if hasattr(ref_table_node, "name") else str(ref_table_node)
                                if ref_node.expressions and hasattr(ref_node.expressions[0], "name"):
                                    target_column = ref_node.expressions[0].name
                            elif isinstance(ref_node, exp.Table):
                                target_table = ref_node.name
                                if kind.expressions and hasattr(kind.expressions[0], "name"):
                                    target_column = kind.expressions[0].name
                            else:
                                target_table = ref_node.name if hasattr(ref_node, "name") else str(ref_node)
                                if kind.expressions and hasattr(kind.expressions[0], "name"):
                                    target_column = kind.expressions[0].name
                            if target_table and target_column:
                                foreign_keys[col_name] = f"{target_table}({target_column})"

                    if is_pk and col_name not in primary_keys:
                        primary_keys.append(col_name)

                    columns[col_name] = ColumnMetadata(
                        name=col_name,
                        data_type=data_type,
                        is_nullable=is_nullable,
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        foreign_target_table=target_table,
                        foreign_target_column=target_column,
                        default_value=default_val,
                    )

                elif isinstance(element, exp.PrimaryKey):
                    for pk_col in element.expressions:
                        col_pk_name = pk_col.name if hasattr(pk_col, "name") else str(pk_col)
                        if col_pk_name not in primary_keys:
                            primary_keys.append(col_pk_name)
                        if col_pk_name in columns:
                            old_col = columns[col_pk_name]
                            columns[col_pk_name] = ColumnMetadata(
                                name=old_col.name,
                                data_type=old_col.data_type,
                                is_nullable=False,
                                is_primary_key=True,
                                is_foreign_key=old_col.is_foreign_key,
                                foreign_target_table=old_col.foreign_target_table,
                                foreign_target_column=old_col.foreign_target_column,
                                default_value=old_col.default_value,
                            )

                elif isinstance(element, exp.ForeignKey):
                    src_cols = [c.name for c in element.expressions if hasattr(c, "name")]
                    reference = element.find(exp.Reference)
                    if reference and reference.this and src_cols:
                        ref_node = reference.this
                        if isinstance(ref_node, exp.Schema):
                            ref_table_node = ref_node.this
                            ref_table = ref_table_node.name if hasattr(ref_table_node, "name") else str(ref_table_node)
                            ref_cols = [c.name for c in ref_node.expressions if hasattr(c, "name")]
                        elif isinstance(ref_node, exp.Table):
                            ref_table = ref_node.name
                            ref_cols = [c.name for c in reference.expressions if hasattr(c, "name")] if reference.expressions else []
                        else:
                            ref_table = ref_node.name if hasattr(ref_node, "name") else str(ref_node)
                            ref_cols = [c.name for c in reference.expressions if hasattr(c, "name")] if reference.expressions else []

                        for i, src_c in enumerate(src_cols):
                            ref_c = ref_cols[i] if i < len(ref_cols) else "id"
                            foreign_keys[src_c] = f"{ref_table}({ref_c})"
                            if src_c in columns:
                                old_c = columns[src_c]
                                columns[src_c] = ColumnMetadata(
                                    name=old_c.name,
                                    data_type=old_c.data_type,
                                    is_nullable=old_c.is_nullable,
                                    is_primary_key=old_c.is_primary_key,
                                    is_foreign_key=True,
                                    foreign_target_table=ref_table,
                                    foreign_target_column=ref_c,
                                    default_value=old_c.default_value,
                                )

            tables[table_name] = TableMetadata(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                raw_ddl=raw_statement_ddl,
            )

        return DatabaseCatalog(db_id=db_id, tables=tables, dialect=target_dialect)

    def from_spider_tables_dict(
        self,
        db_meta: Dict[str, Any],
        db_id: str,
    ) -> DatabaseCatalog:
        raw_tables: List[str] = db_meta.get("table_names_original", [])
        raw_cols: List[List[Any]] = db_meta.get("column_names_original", [])
        raw_types: List[str] = db_meta.get("column_types", [])
        raw_pks: List[int] = db_meta.get("primary_keys", [])
        raw_fks: List[List[int]] = db_meta.get("foreign_keys", [])

        tables: Dict[str, TableMetadata] = {}
        pk_set = set(raw_pks)

        fk_map: Dict[int, int] = {}
        for src_idx, dest_idx in raw_fks:
            fk_map[src_idx] = dest_idx

        for t_idx, t_name in enumerate(raw_tables):
            cols: Dict[str, ColumnMetadata] = {}
            pk_names: List[str] = []
            fk_dict: Dict[str, str] = {}

            for c_idx, (col_t_idx, col_name) in enumerate(raw_cols):
                if col_t_idx != t_idx:
                    continue

                col_type = raw_types[c_idx].upper() if c_idx < len(raw_types) else "TEXT"
                is_pk = c_idx in pk_set
                if is_pk:
                    pk_names.append(col_name)

                is_fk = c_idx in fk_map
                target_tbl: Optional[str] = None
                target_col: Optional[str] = None
                if is_fk:
                    dest_c_idx = fk_map[c_idx]
                    if dest_c_idx < len(raw_cols):
                        dest_t_idx, dest_col_name = raw_cols[dest_c_idx]
                        if 0 <= dest_t_idx < len(raw_tables):
                            target_tbl = raw_tables[dest_t_idx]
                            target_col = dest_col_name
                            fk_dict[col_name] = f"{target_tbl}({target_col})"

            for c_idx, (col_t_idx, col_name) in enumerate(raw_cols):
                if col_t_idx != t_idx:
                    continue

                col_type = raw_types[c_idx].upper() if c_idx < len(raw_types) else "TEXT"
                is_pk = c_idx in pk_set
                is_fk = c_idx in fk_map
                target_tbl = None
                target_col = None
                if is_fk:
                    dest_c_idx = fk_map[c_idx]
                    if dest_c_idx < len(raw_cols):
                        dest_t_idx, dest_col_name = raw_cols[dest_c_idx]
                        if 0 <= dest_t_idx < len(raw_tables):
                            target_tbl = raw_tables[dest_t_idx]
                            target_col = dest_col_name

                cols[col_name] = ColumnMetadata(
                    name=col_name,
                    data_type=col_type,
                    is_nullable=not is_pk,
                    is_primary_key=is_pk,
                    is_foreign_key=is_fk,
                    foreign_target_table=target_tbl,
                    foreign_target_column=target_col,
                )

            table_meta = TableMetadata(
                name=t_name,
                columns=cols,
                primary_keys=pk_names,
                foreign_keys=fk_dict,
            )
            tables[t_name] = table_meta

        return DatabaseCatalog(db_id=db_id, tables=tables, dialect=self.default_dialect)

    def validate_catalog(self, catalog: DatabaseCatalog) -> bool:
        for table in catalog.tables.values():
            for src_col, target in table.foreign_keys.items():
                if not table.has_column(src_col):
                    return False
                match = re.match(r"^(\w+)\((\w+)\)$", target)
                if not match:
                    return False
                target_table_name, target_col_name = match.groups()
                target_table = catalog.get_table(target_table_name)
                if not target_table:
                    return False
                if not target_table.has_column(target_col_name):
                    return False
        return True
