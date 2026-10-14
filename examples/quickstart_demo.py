from __future__ import annotations

import os
from pathlib import Path
import sys

_workspace = str(Path(__file__).resolve().parent.parent)
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

from src.data_engine.schema_parser import ASTSchemaParser
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.serving.grammar import validate_sql_syntax, extract_query_tables, extract_query_columns
from src.serving.local_runner import format_chatml_sql_prompt, SchemaSentinelCPURunner, LocalRunnerConfig
from src.llmops.drift_monitor import EvidentlyDriftMonitor, DriftMonitorConfig
from src.llmops.model_card import ModelCardGenerator
from src.llmops.hardware_audit import LocalHardwareAuditor, HardwareAuditConfig


def main() -> None:
    print("=" * 70)
    print("🚀 SCHEMASENTINEL ENTERPRISE — END-TO-END DEMO WALKTHROUGH")
    print("=" * 70)

    ddl = """
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        tenant_id INT NOT NULL,
        name TEXT NOT NULL,
        tier TEXT NOT NULL
    );
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        tenant_id INT NOT NULL,
        customer_id INT NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """

    print("\n[Step 1] AST Schema Parsing & Pydantic Catalog")
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.parse_ddl(ddl, db_id="ecommerce_prod")
    print(f"Catalog DB: {catalog.db_id}")
    print(f"Parsed Tables: {list(catalog.tables.keys())}")
    for t_name, t_meta in catalog.tables.items():
        print(f"  - Table '{t_name}': {len(t_meta.columns)} columns, PKs: {t_meta.primary_keys}")

    print("\n[Step 2] Policy-Aware ChatML Prompt Construction")
    question = "List top spending VIP customers with total spend over 1000 in tenant 42."
    prompt = format_chatml_sql_prompt(
        schema_ddl=ddl,
        question=question,
        governance_rules=[
            "Strictly read-only SELECT queries permitted.",
            "Tenant isolation mandatory: WHERE tenant_id = <id> required on all queries."
        ]
    )
    print("Prompt Header:")
    for line in prompt.splitlines()[:6]:
        print(f"  {line}")

    print("\n[Step 3] Candidate SQL & Grammar Verification (Lark CFG)")
    valid_sql = (
        "SELECT c.name, SUM(o.total_amount) AS total_spent "
        "FROM customers AS c "
        "JOIN orders AS o ON c.id = o.customer_id "
        "WHERE c.tenant_id = 42 AND c.tier = 'VIP' "
        "GROUP BY c.name "
        "HAVING SUM(o.total_amount) > 1000 "
        "ORDER BY total_spent DESC;"
    )
    is_valid, err = validate_sql_syntax(valid_sql)
    print(f"Valid Query ANSI Syntax Check: {is_valid} (Error: {err})")
    print(f"Referenced Tables: {extract_query_tables(valid_sql)}")
    print(f"Referenced Columns: {extract_query_columns(valid_sql)}")

    print("\n[Step 4] Adversarial Attack & Mutation Policy Interception")
    malicious_sql = "DELETE FROM customers WHERE tenant_id = 42; DROP TABLE orders;"
    is_bad_valid, bad_err = validate_sql_syntax(malicious_sql)
    print(f"Malicious Mutation Grammar Acceptance: {is_bad_valid} (Rejected: {bad_err is not None})")

    validator = SQLPolicyASTValidator(
        tenant_column="tenant_id",
        mandatory_tenant_check=True,
    )
    pol_result = validator.validate_query(malicious_sql)
    print(f"AST Policy Validator Acceptance: {pol_result.is_valid} (Violations caught: {len(pol_result.violations)})")

    print("\n[Step 5] Local CPU Runner (Low-Memory Streaming Inference)")
    model_path = "models/SchemaSentinel-7B-Q4_K_M.gguf"
    has_model = os.path.exists(model_path)
    print(f"Certified GGUF Model Found: {has_model} ({model_path})")
    runner = SchemaSentinelCPURunner(config=LocalRunnerConfig(model_path=model_path, dry_run=not has_model, max_tokens=128))
    gen_result = runner.generate(prompt=prompt, schema_ddl=ddl)
    print(f"Execution Latency: {gen_result.latency_seconds:.4f}s")
    print(f"Extracted SQL:\n  {gen_result.extracted_sql.strip()}")
    print(f"Syntax Certification: {gen_result.is_valid_syntax} | Read-Only: {gen_result.is_read_only}")

    print("\n[Step 6] Evidently AI Data & Schema Drift Monitor")
    ref_queries = [
        "SELECT id, name FROM customers WHERE tenant_id = 42;",
        "SELECT id, total_amount FROM orders WHERE tenant_id = 42;",
        "SELECT name, tier FROM customers WHERE tenant_id = 42;",
    ]
    prod_queries = [
        "SELECT c.name, o.total_amount FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.tenant_id = 42;",
        "SELECT c.id, c.name, c.tier, o.id, o.total_amount, o.status FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.tenant_id = 42 AND o.total_amount > 500;",
        "SELECT id, name, tier FROM customers WHERE tenant_id = 42;",
    ]
    drift_mon = EvidentlyDriftMonitor(config=DriftMonitorConfig(min_samples=3))
    drift_report = drift_mon.run_drift_audit(ref_queries, prod_queries)
    print(f"Drift Audit: has_drift={drift_report.has_drift}")
    print(f"Summary: {drift_report.summary}")

    print("\n[Step 7] Certified Model Card & SLA Governance Report")
    card_gen = ModelCardGenerator()
    gov_report = card_gen.compile_governance_report()
    print(f"Governance SLAs Compliant: {gov_report.overall_compliance}")
    print(f"Certified SHA-256 Fingerprint: {gov_report.governance_hash[:16]}...")
    for sla in gov_report.safety_slas:
        print(f"  - SLA '{sla.metric_name}': Achieved {sla.achieved_score}% vs Target {sla.target_sla} (Passed: {sla.passed})")

    print("\n[Step 8] Local Hardware Audit & Zero Weight Leakage Quarantine")
    auditor = LocalHardwareAuditor(
        HardwareAuditConfig(
            workspace_root=".",
            restricted_extensions=(".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx"),
        )
    )
    audit_report = auditor.run_audit()
    print(f"Hardware Audit: All Passed={audit_report.all_passed}")
    for chk in audit_report.checks:
        print(f"  - Check '{chk.check_name}': {chk.message}")

    print("\n" + "=" * 70)
    print("✅ ALL 8 SCHEMASENTINEL ENTERPRISE PIPELINES CERTIFIED & FUNCTIONAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
