import pytest
from src.data_engine.schema_parser import (
    ASTSchemaParser,
    DatabaseCatalog,
)
from src.data_engine.formatter import PolicyRule, PolicyType
from src.evaluation.runner import (
    ExecutionStatus,
    SQLiteSandbox,
    execute_query_isolated,
)
from src.evaluation.equivalence import (
    ResultSetComparator,
    EquivalenceConfig,
    compare_result_sets,
)
from src.evaluation.policy_ast import (
    SQLPolicyASTValidator,
    validate_sql_policy,
)
from src.evaluation.red_team import (
    AdversarialFuzzer,
    AdversarialDefenseEvaluator,
    DefenseStatus,
)


def test_end_to_end_evaluation_pipeline_golden_match():
    ddl = """
    CREATE TABLE stadium (
        stadium_id INTEGER PRIMARY KEY,
        location TEXT NOT NULL,
        name TEXT NOT NULL,
        capacity INTEGER NOT NULL
    );

    CREATE TABLE concert (
        concert_id INTEGER PRIMARY KEY,
        concert_name TEXT NOT NULL,
        stadium_id INTEGER NOT NULL,
        year INTEGER NOT NULL,
        FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id)
    );
    """
    records = {
        "stadium": [
            {"stadium_id": 1, "location": "London", "name": "Wembley", "capacity": 90000},
            {"stadium_id": 2, "location": "Paris", "name": "Stade de France", "capacity": 80000},
            {"stadium_id": 3, "location": "Berlin", "name": "Olympiastadion", "capacity": 74000},
        ],
        "concert": [
            {"concert_id": 101, "concert_name": "Rock Fest", "stadium_id": 1, "year": 2023},
            {"concert_id": 102, "concert_name": "Pop Gala", "stadium_id": 2, "year": 2023},
            {"concert_id": 103, "concert_name": "Jazz Night", "stadium_id": 1, "year": 2024},
        ],
    }

    gold_sql = (
        "SELECT s.name, count(c.concert_id) AS total_concerts "
        "FROM stadium s JOIN concert c ON s.stadium_id = c.stadium_id "
        "GROUP BY s.stadium_id, s.name "
        "HAVING count(c.concert_id) > 1;"
    )

    pred_sql_equivalent = (
        "SELECT s.name, count(*) "
        "FROM stadium s INNER JOIN concert c ON s.stadium_id = c.stadium_id "
        "GROUP BY s.name "
        "HAVING count(*) > 1;"
    )

    sandbox = SQLiteSandbox()
    res_gold = sandbox.execute_isolated(gold_sql, ddl=ddl, seed_records=records)
    res_pred = sandbox.execute_isolated(pred_sql_equivalent, ddl=ddl, seed_records=records)

    assert res_gold.status == ExecutionStatus.SUCCESS
    assert res_pred.status == ExecutionStatus.SUCCESS

    comparator = ResultSetComparator(EquivalenceConfig(order_agnostic=True))
    report = comparator.compare_execution_results(res_pred, res_gold)

    assert report.is_equivalent is True
    assert report.predicted_row_count == 1
    assert report.gold_row_count == 1

    validator = SQLPolicyASTValidator()
    policy_res = validator.validate(pred_sql_equivalent)
    assert policy_res.is_valid is True
    assert policy_res.has_mutations is False


def test_end_to_end_evaluation_pipeline_semantic_mismatch():
    ddl = "CREATE TABLE products (id INTEGER, price REAL, category TEXT);"
    records = {
        "products": [
            {"id": 1, "price": 10.0, "category": "books"},
            {"id": 2, "price": 50.0, "category": "electronics"},
            {"id": 3, "price": 15.0, "category": "books"},
        ]
    }

    gold_sql = "SELECT id FROM products WHERE category = 'books' AND price > 12.0;"
    pred_sql_wrong = "SELECT id FROM products WHERE category = 'books';"

    sandbox = SQLiteSandbox()
    res_gold = sandbox.execute_isolated(gold_sql, ddl=ddl, seed_records=records)
    res_pred = sandbox.execute_isolated(pred_sql_wrong, ddl=ddl, seed_records=records)

    comparator = ResultSetComparator()
    report = comparator.compare_execution_results(res_pred, res_gold)

    assert report.is_equivalent is False
    assert report.predicted_row_count == 2
    assert report.gold_row_count == 1


def test_end_to_end_evaluation_pipeline_red_team_containment():
    ddl = "CREATE TABLE accounts (id INTEGER PRIMARY KEY, tenant_id TEXT, balance REAL);"
    policies = [
        PolicyRule(
            rule_id="SEC_RO",
            rule_type=PolicyType.READ_ONLY,
            description="Read-only operations strictly enforced.",
        ),
        PolicyRule(
            rule_id="SEC_TENANT",
            rule_type=PolicyType.TENANT_ISOLATION,
            description="Tenant isolation required on accounts.",
            target_tables=["accounts"],
            required_predicates=["tenant_id = :tenant_id"],
        ),
    ]

    clean_sql = "SELECT id, balance FROM accounts WHERE tenant_id = 't-acme';"
    fuzzer = AdversarialFuzzer()
    attacks = fuzzer.fuzz_mutations(clean_sql)
    attacks.extend(fuzzer.fuzz_tenant_leakage(clean_sql, tenant_column="tenant_id"))

    evaluator = AdversarialDefenseEvaluator()
    audit = evaluator.audit_attacks(attacks, policies=policies, ddl=ddl)

    assert audit.total_attacks >= 5
    assert audit.blocked_count == audit.total_attacks
    assert audit.breached_count == 0
    assert audit.defense_rate == 1.0
