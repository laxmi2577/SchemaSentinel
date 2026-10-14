from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field
import sqlglot
from sqlglot import exp

SQL_EBNF_GRAMMAR = r"""
start: query_stmt [";"]

?query_stmt: with_clause? set_query

with_clause: WITH [ RECURSIVE ] cte_binding ("," cte_binding)*
cte_binding: CNAME AS "(" query_stmt ")"

?set_query: select_query (( UNION [ ALL ] | INTERSECT | EXCEPT ) select_query)*

select_query: SELECT [ DISTINCT | ALL ] select_list [ from_clause ] [ where_clause ] [ groupby_clause ] [ having_clause ] [ orderby_clause ] [ limit_clause ]

select_list: select_item ("," select_item)*
select_item: expr [ [ AS ] alias ] | STAR | table_prefix STAR
table_prefix: (CNAME ".")
alias: CNAME | ESCAPED_IDENTIFIER

from_clause: FROM table_ref (join_clause)*
?table_ref: table_source [ [ AS ] alias ]
?table_source: table_name | "(" query_stmt ")"
table_name: (schema_name ".")? CNAME
schema_name: CNAME

join_clause: [ join_type ] JOIN table_ref [ ON expr | USING "(" CNAME ("," CNAME)* ")" ]
join_type: INNER | LEFT [ OUTER ] | RIGHT [ OUTER ] | FULL [ OUTER ] | CROSS

where_clause: WHERE expr
groupby_clause: GROUP BY expr ("," expr)*
having_clause: HAVING expr
orderby_clause: ORDER BY order_item ("," order_item)*
order_item: expr [ ASC | DESC ] [ NULLS ( FIRST | LAST ) ]
limit_clause: LIMIT ( INT | PARAMETER ) [ OFFSET ( INT | PARAMETER ) ] | LIMIT ( INT | PARAMETER ) "," ( INT | PARAMETER )

?expr: or_test
?or_test: and_test ( OR and_test )*
?and_test: not_test ( AND not_test )*
?not_test: NOT not_test -> not_expr | comparison

?comparison: atom_expr comp_op atom_expr -> comp_expr
           | atom_expr [ NOT ] IN "(" ( query_stmt | in_expr_list ) ")" -> in_expr
           | atom_expr [ NOT ] BETWEEN atom_expr AND atom_expr -> between_expr
           | atom_expr [ NOT ] LIKE atom_expr -> like_expr
           | atom_expr IS [ NOT ] NULL -> is_null_expr
           | [ NOT ] EXISTS "(" query_stmt ")" -> exists_expr
           | atom_expr

!comp_op: "=" | "!=" | "<>" | "<=" | ">=" | "<" | ">"

in_expr_list: expr ("," expr)*

?atom_expr: ( "+" | "-" ) atom_expr -> unary_expr
          | func_call
          | case_expr
          | cast_expr
          | column_ref
          | literal
          | "(" expr ")"
          | "(" query_stmt ")" -> scalar_subquery

column_ref: (table_prefix)? ( CNAME | ESCAPED_IDENTIFIER )
func_call: func_name "(" [ [ DISTINCT ] func_args ] ")"
func_name: CNAME
func_args: ( expr ("," expr)* | STAR )

case_expr: CASE [ expr ] ( WHEN expr THEN expr )+ [ ELSE expr ] END
cast_expr: CAST "(" expr AS data_type ")"

data_type: CNAME [ "(" INT ("," INT)? ")" ]

?literal: NUMBER -> num_literal
        | STRING -> str_literal
        | TRUE -> true_literal
        | FALSE -> false_literal
        | NULL -> null_literal
        | PARAMETER -> param_literal

SELECT.2: "SELECT"i
FROM.2: "FROM"i
WHERE.2: "WHERE"i
JOIN.2: "JOIN"i
ON.2: "ON"i
USING.2: "USING"i
GROUP.2: "GROUP"i
ORDER.2: "ORDER"i
BY.2: "BY"i
HAVING.2: "HAVING"i
LIMIT.2: "LIMIT"i
OFFSET.2: "OFFSET"i
UNION.2: "UNION"i
INTERSECT.2: "INTERSECT"i
EXCEPT.2: "EXCEPT"i
ALL.2: "ALL"i
DISTINCT.2: "DISTINCT"i
WITH.2: "WITH"i
RECURSIVE.2: "RECURSIVE"i
CASE.2: "CASE"i
WHEN.2: "WHEN"i
THEN.2: "THEN"i
ELSE.2: "ELSE"i
END.2: "END"i
CAST.2: "CAST"i
AS.2: "AS"i
AND.2: "AND"i
OR.2: "OR"i
NOT.2: "NOT"i
IN.2: "IN"i
BETWEEN.2: "BETWEEN"i
LIKE.2: "LIKE"i
IS.2: "IS"i
NULL.2: "NULL"i
TRUE.2: "TRUE"i
FALSE.2: "FALSE"i
EXISTS.2: "EXISTS"i
ASC.2: "ASC"i
DESC.2: "DESC"i
NULLS.2: "NULLS"i
FIRST.2: "FIRST"i
LAST.2: "LAST"i
INNER.2: "INNER"i
LEFT.2: "LEFT"i
RIGHT.2: "RIGHT"i
FULL.2: "FULL"i
CROSS.2: "CROSS"i
OUTER.2: "OUTER"i

STAR: "*"
PARAMETER: /:[a-zA-Z_][a-zA-Z0-9_]*/ | "?"
ESCAPED_IDENTIFIER: /`[^`]+`/ | /"[^"]+"/ | /\[[^\]]+\]/
STRING: /'([^'\\]|\\.)*'/ | /"([^"\\]|\\.)*"/
CNAME: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.INT
%import common.NUMBER
%import common.WS
%ignore WS
"""

SQL_READ_ONLY_CFG = SQL_EBNF_GRAMMAR

FORBIDDEN_MUTATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "drop",
        "delete",
        "truncate",
        "alter",
        "update",
        "insert",
        "create",
        "replace",
        "grant",
        "revoke",
        "merge",
        "upsert",
    }
)


class SQLSyntaxReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    sql: str
    is_valid: bool
    is_read_only: bool
    error_message: Optional[str] = None
    tables: List[str] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    has_forbidden_mutations: bool = False


def load_lark_parser(
    parser_type: str = "earley",
    lexer: str = "auto",
) -> Optional[Any]:
    try:
        import lark

        return lark.Lark(
            SQL_EBNF_GRAMMAR,
            start="start",
            parser=parser_type,
            lexer=lexer,
            propagate_positions=True,
        )
    except Exception:
        return None


def _check_forbidden_keywords(sql: str) -> Tuple[bool, Optional[str]]:
    tokens = re.findall(r"\b[a-zA-Z_]+\b", sql.lower())
    for token in tokens:
        if token in FORBIDDEN_MUTATION_KEYWORDS:
            return True, f"Prohibited mutation keyword detected: {token.upper()}"
    return False, None


def validate_sql_syntax(
    sql: str,
    parser: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    cleaned = sql.strip()
    if not cleaned:
        return False, "Empty SQL statement."

    has_mutation, mutation_err = _check_forbidden_keywords(cleaned)
    if has_mutation:
        return False, mutation_err

    p = parser if parser is not None else load_lark_parser()
    if p is not None:
        try:
            p.parse(cleaned)
        except Exception as e:
            return False, f"Lark parse error: {e}"

    try:
        parsed_trees = sqlglot.parse(cleaned)
        if not parsed_trees or any(t is None for t in parsed_trees):
            return False, "Failed to parse SQL statement."
        for tree in parsed_trees:
            if not isinstance(tree, (exp.Select, exp.Union)):
                return False, f"Non-SELECT statement not permitted: {type(tree).__name__}"
        return True, None
    except Exception as e:
        return False, f"SQL syntax error: {e}"


def is_read_only_syntax(sql: str, parser: Optional[Any] = None) -> bool:
    is_valid, _ = validate_sql_syntax(sql, parser=parser)
    return is_valid


def extract_query_tables(sql: str) -> List[str]:
    cleaned = sql.strip()
    if not cleaned:
        return []

    tables: Set[str] = set()
    try:
        for expression in sqlglot.parse(cleaned):
            if expression:
                for table in expression.find_all(exp.Table):
                    if table.name:
                        tables.add(table.name.lower())
    except Exception:
        pass
    return sorted(list(tables))


def extract_query_columns(sql: str) -> List[str]:
    cleaned = sql.strip()
    if not cleaned:
        return []

    columns: Set[str] = set()
    try:
        for expression in sqlglot.parse(cleaned):
            if expression:
                for column in expression.find_all(exp.Column):
                    if column.name:
                        columns.add(column.name.lower())
    except Exception:
        pass
    return sorted(list(columns))


class SQLGrammarValidator:
    def __init__(self, parser: Optional[Any] = None) -> None:
        self.parser = parser or load_lark_parser()

    def validate(self, sql: str) -> SQLSyntaxReport:
        cleaned = sql.strip()
        has_mutation, mutation_err = _check_forbidden_keywords(cleaned)
        if has_mutation:
            return SQLSyntaxReport(
                sql=cleaned,
                is_valid=False,
                is_read_only=False,
                error_message=mutation_err,
                has_forbidden_mutations=True,
            )

        is_valid, err = validate_sql_syntax(cleaned, parser=self.parser)
        tables = extract_query_tables(cleaned) if is_valid else []
        columns = extract_query_columns(cleaned) if is_valid else []

        return SQLSyntaxReport(
            sql=cleaned,
            is_valid=is_valid,
            is_read_only=is_valid,
            error_message=err,
            tables=tables,
            columns=columns,
            has_forbidden_mutations=False,
        )

    def is_valid(self, sql: str) -> bool:
        return self.validate(sql).is_valid

    def extract_tables(self, sql: str) -> List[str]:
        return extract_query_tables(sql)

    def extract_columns(self, sql: str) -> List[str]:
        return extract_query_columns(sql)
