from __future__ import annotations

import os
from pathlib import Path
import sys

_workspace = str(Path(__file__).resolve().parent.parent)
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

from src.data_engine.schema_parser import ASTSchemaParser
from src.evaluation.policy_ast import SQLPolicyASTValidator
from src.serving.grammar import validate_sql_syntax
from src.serving.local_runner import (
    LocalRunnerConfig,
    SchemaSentinelCPURunner,
    format_chatml_sql_prompt,
)


def main() -> None:
    model_path = "models/SchemaSentinel-7B-Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    print("=" * 65)
    print("🛡️ SCHEMASENTINEL ENTERPRISE — INTERACTIVE SQL GENERATOR")
    print(f"Loaded Custom Model: {model_path} ({os.path.getsize(model_path) / (1024**3):.2f} GB)")
    print("=" * 65)

    sample_ddl = """
    CREATE TABLE employees (
        id INT PRIMARY KEY,
        tenant_id INT NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL
    );
    CREATE TABLE projects (
        id INT PRIMARY KEY,
        tenant_id INT NOT NULL,
        lead_id INT NOT NULL,
        budget REAL NOT NULL,
        FOREIGN KEY (lead_id) REFERENCES employees(id)
    );
    """
    print("\n[Active Database Schema]:")
    print(sample_ddl.strip())
    print("\nEnter your plain-English question below (or type 'exit' to quit):")

    runner = SchemaSentinelCPURunner(
        config=LocalRunnerConfig(
            model_path=model_path,
            max_tokens=256,
        )
    )
    if runner.is_live:
        print("\n⚡ Execution Mode: LIVE Neural CPU Inference (SchemaSentinel weights active)")
    else:
        print(f"\n⚠️ Execution Mode: OFFLINE SIMULATION ({runner.init_error})")
        print("   To enable REAL 4.36 GB neural CPU inference, install the prebuilt wheel:")
        print("   & \"D:\\PYTHON\\python.exe\" -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")

    ast_validator = SQLPolicyASTValidator(
        tenant_column="tenant_id",
        mandatory_tenant_check=True,
    )

    while True:
        try:
            print("-" * 65)
            question = input("\n💬 Your Question > ").strip()
            if not question or question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            prompt = format_chatml_sql_prompt(
                schema_ddl=sample_ddl,
                question=question,
                governance_rules=[
                    "Strictly read-only SELECT queries.",
                    "Tenant isolation mandatory: WHERE tenant_id = <id> required.",
                ],
            )

            print("\n⚙️ Generating safe SQL with SchemaSentinel...")
            result = runner.generate(prompt=prompt)
            sql = result.extracted_sql.strip()

            print("\n📄 Generated SQL:")
            print(f"  {sql}\n")

            is_cfg_valid, cfg_err = validate_sql_syntax(sql)
            pol_res = ast_validator.validate_query(sql)

            print(f"🔒 CFG Grammar Valid: {is_cfg_valid}")
            print(f"🛡️ Security Policy Valid: {pol_res.is_valid} (Violations: {len(pol_res.violations)})")
            if not pol_res.is_valid:
                for v in pol_res.violations:
                    print(f"   ⚠️ Blocked: {v.message}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
