import pytest
from src.data_engine.formatter import PolicyRule, PolicyType
from src.data_engine.ingestion import RawTextToSQLSample, BenchmarkSource, BenchmarkSplit
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.evaluation.runner import SQLiteSandbox
from src.evaluation.red_team import (
    AttackCategory,
    AdversarialAttack,
    DefenseStatus,
    RedTeamDefenseResult,
    RedTeamAuditSummary,
    AdversarialFuzzer,
    AdversarialDefenseEvaluator,
    inject_sql_attacks,
)


def test_fuzz_mutations_generation():
    fuzzer = AdversarialFuzzer()
    base_sql = "SELECT id, name FROM users WHERE id = 10;"

    attacks = fuzzer.fuzz_mutations(base_sql)
    assert len(attacks) == 3
    for a in attacks:
        assert a.category == AttackCategory.UNAUTHORIZED_MUTATION
        assert a.expected_policy == PolicyType.READ_ONLY
        assert a.mutated_sql is not None

    mutated_sqls = [a.mutated_sql for a in attacks]
    assert any("DROP TABLE" in s for s in mutated_sqls)
    assert any("DELETE FROM" in s for s in mutated_sqls)
    assert any("UPDATE" in s for s in mutated_sqls)


def test_fuzz_tenant_leakage_generation():
    fuzzer = AdversarialFuzzer()
    base_sql = "SELECT id, total FROM orders WHERE tenant_id = 't-01' AND total > 50;"

    attacks = fuzzer.fuzz_tenant_leakage(base_sql, tenant_column="tenant_id")
    assert len(attacks) == 2
    for a in attacks:
        assert a.category == AttackCategory.TENANT_LEAKAGE
        assert a.expected_policy == PolicyType.TENANT_ISOLATION
        assert a.mutated_sql is not None

    tautology_attack = attacks[0]
    assert "1=1" in tautology_attack.mutated_sql


def test_fuzz_restricted_schema():
    fuzzer = AdversarialFuzzer()
    attacks = fuzzer.fuzz_restricted_schema(
        forbidden_table="auth_tokens",
        forbidden_column="secret_salt",
    )

    assert len(attacks) == 2
    assert attacks[0].category == AttackCategory.RESTRICTED_SCHEMA_PROBE
    assert "auth_tokens" in attacks[0].mutated_sql
    assert attacks[0].expected_policy == PolicyType.RESTRICTED_TABLE

    assert attacks[1].category == AttackCategory.RESTRICTED_SCHEMA_PROBE
    assert "secret_salt" in attacks[1].mutated_sql
    assert attacks[1].expected_policy == PolicyType.FORBIDDEN_COLUMNS


def test_fuzz_prompt_injections():
    fuzzer = AdversarialFuzzer()
    attacks = fuzzer.fuzz_prompt_injections("Show sales summary for region West.")

    assert len(attacks) >= 4
    for a in attacks:
        assert a.category == AttackCategory.PROMPT_INJECTION
        assert "Show sales summary for region West." in a.adversarial_prompt
        assert a.mutated_sql is None


def test_comprehensive_fuzz_suite():
    sample = RawTextToSQLSample(
        sample_id="fuzz_01",
        db_id="corp_db",
        question="How many active employees are there?",
        query="SELECT count(*) FROM employees WHERE tenant_id = 't-99' AND status = 'ACTIVE';",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.TRAIN,
    )
    attacks = inject_sql_attacks(sample.query, sample.question, tenant_column="tenant_id")
    assert len(attacks) >= 9

    categories = {a.category for a in attacks}
    assert AttackCategory.UNAUTHORIZED_MUTATION in categories
    assert AttackCategory.TENANT_LEAKAGE in categories
    assert AttackCategory.RESTRICTED_SCHEMA_PROBE in categories
    assert AttackCategory.PROMPT_INJECTION in categories


def test_adversarial_defense_evaluator_audit():
    ddl = "CREATE TABLE customers (id INTEGER PRIMARY KEY, tenant_id TEXT, name TEXT);"
    policies = [
        PolicyRule(
            rule_id="RO_01",
            rule_type=PolicyType.READ_ONLY,
            description="Strict read only",
        ),
        PolicyRule(
            rule_id="TEN_01",
            rule_type=PolicyType.TENANT_ISOLATION,
            description="Mandatory tenant predicate",
            target_tables=["customers"],
            required_predicates=["tenant_id = :tenant_id"],
        ),
        PolicyRule(
            rule_id="REST_01",
            rule_type=PolicyType.RESTRICTED_TABLE,
            description="Credentials restricted",
            target_tables=["credentials"],
        ),
    ]

    evaluator = AdversarialDefenseEvaluator()
    sample = RawTextToSQLSample(
        sample_id="audit_s1",
        db_id="retail",
        question="Count total customers.",
        query="SELECT count(*) FROM customers WHERE tenant_id = 'acme';",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.TRAIN,
    )
    fuzzer = AdversarialFuzzer()
    attacks = fuzzer.generate_comprehensive_fuzz_suite(sample, tenant_column="tenant_id")

    audit = evaluator.audit_attacks(attacks, policies=policies, ddl=ddl)

    assert audit.total_attacks == len(attacks)
    assert audit.blocked_count == audit.total_attacks
    assert audit.breached_count == 0
    assert audit.defense_rate == 1.0
