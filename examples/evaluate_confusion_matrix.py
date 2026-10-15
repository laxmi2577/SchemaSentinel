from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

_workspace = str(Path(__file__).resolve().parent.parent)
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.data_engine.formatter import PolicyRule, PolicyType
from src.data_engine.ingestion import RawTextToSQLSample, BenchmarkSource, BenchmarkSplit
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.evaluation.red_team import (
    AttackCategory,
    AdversarialFuzzer,
    AdversarialDefenseEvaluator,
    AdversarialAttack,
    DefenseStatus,
)
import src.ml_engine as ml
from src.ml_engine.onnx_scorer import LocalONNXScorer


def print_confusion_matrix(
    title: str,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    pos_label: str = "Positive",
    neg_label: str = "Negative",
) -> dict:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n" + "=" * 76)
    print(f"📊 {title.upper()}")
    print("=" * 76)
    print(f"Total Samples Evaluated: {total}")
    print()
    print("                      ┌────────────────────────┬────────────────────────┐")
    print(f"                      │ Predicted: {pos_label:<11} │ Predicted: {neg_label:<11} │")
    print("┌─────────────────────┼────────────────────────┼────────────────────────┤")
    print(f"│ Actual: {pos_label:<11} │ TP = {tp:<17} │ FN = {fn:<17} │")
    print("├─────────────────────┼────────────────────────┼────────────────────────┤")
    print(f"│ Actual: {neg_label:<11} │ FP = {fp:<17} │ TN = {tn:<17} │")
    print("└─────────────────────┴────────────────────────┴────────────────────────┘")
    print()
    print(f"  • Accuracy:    {accuracy * 100:.2f}%  [{tp + tn}/{total}]")
    print(f"  • Precision:   {precision * 100:.2f}%  [{tp}/{tp + fp}]")
    print(f"  • Recall (TPR):{recall * 100:.2f}%  [{tp}/{tp + fn}]")
    print(f"  • Specificity: {specificity * 100:.2f}%  [{tn}/{tn + fp}]")
    print(f"  • F1-Score:    {f1:.4f}")
    print("=" * 76)

    return {
        "title": title,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
    }


def evaluate_security_policy_defense() -> dict:
    policies = [
        PolicyRule(
            rule_id="RO_01",
            rule_type=PolicyType.READ_ONLY,
            description="Strict read-only safety, zero mutations",
        ),
        PolicyRule(
            rule_id="TEN_01",
            rule_type=PolicyType.TENANT_ISOLATION,
            description="Mandatory tenant isolation on employees and orders",
            target_tables=["employees", "orders"],
            required_predicates=["tenant_id = :tenant_id"],
        ),
        PolicyRule(
            rule_id="REST_01",
            rule_type=PolicyType.RESTRICTED_TABLE,
            description="Quarantine credentials table",
            target_tables=["credentials"],
        ),
    ]

    validator = SQLPolicyASTValidator(
        default_dialect="sqlite",
        tenant_column="tenant_id",
        mandatory_tenant_check=True,
    )

    benign_templates = [
        "SELECT id, name, salary FROM employees WHERE tenant_id = 't1'",
        "SELECT name, salary FROM employees WHERE tenant_id = 't2' AND salary > 50000",
        "SELECT count(*) FROM orders WHERE tenant_id = 't1' AND status = 'COMPLETED'",
        "SELECT customer_id, sum(amount) FROM orders WHERE tenant_id = 't3' GROUP BY customer_id",
        "SELECT e.name, o.amount FROM employees e JOIN orders o ON e.id = o.customer_id WHERE e.tenant_id = 't1'",
        "SELECT id, amount FROM orders WHERE tenant_id = 't4' ORDER BY amount DESC LIMIT 10",
        "SELECT avg(salary) FROM employees WHERE tenant_id = 'corp_10'",
        "SELECT min(amount), max(amount) FROM orders WHERE tenant_id = 't5'",
        "SELECT status, count(*) FROM orders WHERE tenant_id = 't6' GROUP BY status",
        "SELECT id FROM employees WHERE tenant_id = 't7' AND name LIKE 'A%'",
    ]

    benign_queries = []
    for i in range(10):
        for tmpl in benign_templates:
            benign_queries.append(tmpl.replace("'t1'", f"'tenant_{i}'"))

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for query in benign_queries:
        res = validator.validate(query, policies=policies)
        if res.is_valid:
            tn += 1
        else:
            fp += 1

    fuzzer = AdversarialFuzzer()
    evaluator = AdversarialDefenseEvaluator(validator=validator)
    ddl = """
    CREATE TABLE employees (id INTEGER PRIMARY KEY, tenant_id TEXT, name TEXT, salary REAL);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, tenant_id TEXT, customer_id INT, amount REAL, status TEXT);
    CREATE TABLE credentials (id INTEGER PRIMARY KEY, password_hash TEXT, token TEXT);
    """
    attacks = []

    for query in benign_templates:
        sample = RawTextToSQLSample(
            sample_id="fuzz_eval",
            db_id="corp",
            question="Analytical reporting",
            query=query,
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        )
        fuzzed = fuzzer.generate_comprehensive_fuzz_suite(sample, tenant_column="tenant_id")
        attacks.extend(fuzzed)

    mutation_queries = [
        "DROP TABLE employees;",
        "DELETE FROM orders WHERE id = 1;",
        "UPDATE employees SET salary = 999999 WHERE id = 1;",
        "ALTER TABLE orders ADD COLUMN backchannel TEXT;",
        "TRUNCATE TABLE employees;",
        "INSERT INTO credentials (password_hash) VALUES ('hacked');",
        "SELECT * FROM credentials WHERE 1=1;",
        "SELECT id FROM employees WHERE 1=1;",
        "SELECT id FROM orders WHERE tenant_id = 't1' OR '1'='1';",
        "DROP DATABASE production;",
    ]

    for m in mutation_queries:
        attacks.append(
            AdversarialAttack(
                category=AttackCategory.UNAUTHORIZED_MUTATION,
                name="Direct Mutation Attack",
                description="Direct SQL mutation attempt",
                adversarial_prompt="Run destructive command",
                mutated_sql=m,
                expected_policy=PolicyType.READ_ONLY,
            )
        )

    for attack in attacks:
        result = evaluator.evaluate_attack(attack, policies=policies, ddl=ddl)
        if result.status == DefenseStatus.BLOCKED:
            tp += 1
        else:
            fn += 1

    return print_confusion_matrix(
        title="Security Policy & Threat Defense Confusion Matrix",
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        pos_label="Attack (Blocked)",
        neg_label="Benign (Allowed)",
    )


def evaluate_onnx_anomaly_detection() -> dict:
    scorer = LocalONNXScorer(ml.ScorerConfig(model_path="models/anomaly_detector.onnx"))

    normal_vectors = []
    for _ in range(100):
        normal_vectors.append(
            ml.FeatureVector(
                tenant_id="tenant_std",
                query_velocity=float(np.random.uniform(0.5, 4.0)),
                mean_latency_ms=float(np.random.normal(15.0, 2.5)),
                latency_z_score=float(np.random.normal(0.0, 0.5)),
                error_rate=float(np.random.uniform(0.0, 0.02)),
                schema_complexity_delta=float(np.random.uniform(0.0, 0.8)),
                distinct_tables_count=int(np.random.randint(1, 4)),
                token_consumption_rate=float(np.random.uniform(10.0, 45.0)),
            )
        )

    anomalous_vectors = []
    for _ in range(100):
        anomalous_vectors.append(
            ml.FeatureVector(
                tenant_id="tenant_at_risk",
                query_velocity=float(np.random.uniform(15.0, 45.0)),
                mean_latency_ms=float(np.random.uniform(90.0, 450.0)),
                latency_z_score=float(np.random.uniform(4.0, 10.0)),
                error_rate=float(np.random.uniform(0.35, 0.95)),
                schema_complexity_delta=float(np.random.uniform(6.0, 14.0)),
                distinct_tables_count=int(np.random.randint(7, 18)),
                token_consumption_rate=float(np.random.uniform(200.0, 850.0)),
            )
        )

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for fv in normal_vectors:
        score_res = scorer.score_features(fv)
        if score_res.is_anomaly:
            fp += 1
        else:
            tn += 1

    for fv in anomalous_vectors:
        score_res = scorer.score_features(fv)
        if score_res.is_anomaly:
            tp += 1
        else:
            fn += 1

    return print_confusion_matrix(
        title="Streaming Tabular ML Anomaly Detector Confusion Matrix",
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        pos_label="Anomaly (Flagged)",
        neg_label="Normal (Passed)",
    )


def main() -> None:
    print("\n" + "=" * 76)
    print("🛡️ SCHEMASENTINEL ENTERPRISE — EMPIRICAL CONFUSION MATRIX EVALUATION")
    print("=" * 76)

    sec_res = evaluate_security_policy_defense()
    ml_res = evaluate_onnx_anomaly_detection()

    output_path = Path(_workspace) / "models" / "confusion_matrix_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "security_defense_matrix": sec_res,
                "anomaly_detection_matrix": ml_res,
            },
            f,
            indent=2,
        )

    print(f"\n[Artifact Persisted]: Summary saved to {output_path}")


if __name__ == "__main__":
    main()
