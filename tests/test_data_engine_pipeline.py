import pytest
from src.data_engine.schema_parser import (
    ASTSchemaParser,
    DatabaseCatalog,
)
from src.data_engine.ingestion import (
    BenchmarkSource,
    BenchmarkSplit,
    RawTextToSQLSample,
)
from src.data_engine.deduplication import (
    MinHashConfig,
    BenchmarkContaminationDetector,
)
from src.data_engine.formatter import (
    ChatMLFormatter,
    PolicyRule,
    PolicyType,
    DecisionType,
    AssistantResponse,
)
from src.data_engine.collator import (
    CollatorConfig,
    CompletionOnlyCollator,
)


def test_end_to_end_data_engine_allow_pipeline():
    ddl = """
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        tenant_id VARCHAR(32) NOT NULL,
        name TEXT NOT NULL,
        email TEXT
    );

    CREATE TABLE transactions (
        tx_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        tenant_id VARCHAR(32) NOT NULL,
        amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.parse_ddl(ddl, db_id="fintech_db")

    assert catalog.db_id == "fintech_db"
    assert len(catalog.tables) == 2
    assert parser.validate_catalog(catalog) is True

    benchmark_dev_sample = RawTextToSQLSample(
        sample_id="dev_gold_01",
        db_id="fintech_db",
        question="What is the total transaction volume for tenant T-100?",
        query="SELECT SUM(amount) FROM transactions WHERE tenant_id = 'T-100';",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.DEV,
    )
    detector = BenchmarkContaminationDetector(MinHashConfig(threshold=0.85))
    detector.register_benchmark_samples([benchmark_dev_sample])

    raw_train_batch = [
        RawTextToSQLSample(
            sample_id="train_leaked_01",
            db_id="fintech_db",
            question="What is the total transaction volume for tenant T-100?",
            query="SELECT SUM(amount) FROM transactions WHERE tenant_id = 'T-100';",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
        RawTextToSQLSample(
            sample_id="train_valid_02",
            db_id="fintech_db",
            question="Find customer names who have transactions over 500 dollars.",
            query="SELECT c.name FROM customers c JOIN transactions t ON c.customer_id = t.customer_id WHERE t.amount > 500 AND c.tenant_id = :tenant_id;",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
        RawTextToSQLSample(
            sample_id="train_dup_03",
            db_id="fintech_db",
            question="Find customer names who have transactions over 500 dollars.",
            query="SELECT c.name FROM customers c JOIN transactions t ON c.customer_id = t.customer_id WHERE t.amount > 500 AND c.tenant_id = :tenant_id;",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
    ]

    sanitized_samples, summary = detector.sanitize_training_stream(
        raw_train_batch,
        deduplicate_internal=True,
    )

    assert summary.total_records == 3
    assert summary.quarantined_benchmark_leakages == 1
    assert summary.internal_duplicates_dropped == 1
    assert summary.unique_records == 1
    assert len(sanitized_samples) == 1
    assert sanitized_samples[0].sample_id == "train_valid_02"

    policies = [
        PolicyRule(
            rule_id="SEC_RO",
            rule_type=PolicyType.READ_ONLY,
            description="Only SELECT queries are authorized.",
        ),
        PolicyRule(
            rule_id="SEC_TENANT",
            rule_type=PolicyType.TENANT_ISOLATION,
            description="Tenant isolation predicate mandatory on all queries.",
            target_tables=["customers", "transactions"],
            required_predicates=["tenant_id = :tenant_id"],
        ),
    ]
    formatter = ChatMLFormatter()
    gold_response = AssistantResponse(
        decision=DecisionType.ALLOW,
        sql=sanitized_samples[0].query,
        rationale="Enforces read-only join with parameterized tenant boundary predicate.",
    )

    formatted_chatml = formatter.format_full_chatml(
        catalog=catalog,
        policies=policies,
        user_query=sanitized_samples[0].question,
        response=gold_response,
    )

    assert "<|im_start|>system" in formatted_chatml
    assert "DATABASE SCHEMA (fintech_db):" in formatted_chatml
    assert "CREATE TABLE customers" in formatted_chatml
    assert "CREATE TABLE transactions" in formatted_chatml
    assert "[TENANT_ISOLATION]: Tenant isolation predicate mandatory" in formatted_chatml
    assert "<|im_start|>user\nFind customer names who have transactions over 500 dollars.<|im_end|>" in formatted_chatml
    assert "<|im_start|>assistant\nALLOW\n```sql" in formatted_chatml
    assert "EVIDENCE: Enforces read-only join" in formatted_chatml

    mock_prompt_tokens = [151644, 8948, 198] + list(range(100, 150)) + [151645, 198]
    mock_response_template = [151644, 77091, 198]
    mock_completion_tokens = [501, 502, 503, 504, 151645]

    sample_input_ids = mock_prompt_tokens + mock_response_template + mock_completion_tokens

    collator = CompletionOnlyCollator(
        CollatorConfig(
            response_template_ids=mock_response_template,
            pad_token_id=151643,
            ignore_index=-100,
            pad_to_multiple_of=8,
        )
    )

    batch = collator.collate_batch([{"input_ids": sample_input_ids}], return_tensors=False)

    prompt_and_template_len = len(mock_prompt_tokens) + len(mock_response_template)
    total_unpadded_len = len(sample_input_ids)
    batch_input_ids = batch["input_ids"][0]
    batch_labels = batch["labels"][0]
    batch_mask = batch["attention_mask"][0]

    assert len(batch_input_ids) % 8 == 0
    assert len(batch_labels) == len(batch_input_ids)
    assert len(batch_mask) == len(batch_input_ids)

    for i in range(prompt_and_template_len):
        assert batch_labels[i] == -100

    for i, expected_tok in enumerate(mock_completion_tokens):
        assert batch_labels[prompt_and_template_len + i] == expected_tok

    for i in range(total_unpadded_len, len(batch_labels)):
        assert batch_labels[i] == -100
        assert batch_mask[i] == 0
        assert batch_input_ids[i] == 151643


def test_end_to_end_data_engine_refuse_pipeline():
    ddl = "CREATE TABLE audit_logs (log_id INTEGER PRIMARY KEY, msg TEXT);"
    parser = ASTSchemaParser(default_dialect="sqlite")
    catalog = parser.parse_ddl(ddl, db_id="audit_db")

    policies = [
        PolicyRule(
            rule_id="RO_01",
            rule_type=PolicyType.READ_ONLY,
            description="Mutations and deletions are strictly forbidden.",
        )
    ]
    formatter = ChatMLFormatter()
    refusal_response = AssistantResponse(
        decision=DecisionType.REFUSE,
        sql=None,
        rationale="Request requires TRUNCATE or DROP TABLE, which violates read-only governance.",
        policy_violations=["READ_ONLY"],
    )

    chatml = formatter.format_full_chatml(
        catalog=catalog,
        policies=policies,
        user_query="Purge all audit logs immediately.",
        response=refusal_response,
    )

    assert "REFUSE\nVIOLATIONS: READ_ONLY\nEVIDENCE: Request requires TRUNCATE" in chatml
    parsed_back = formatter.extract_assistant_response(
        chatml.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
    )

    assert parsed_back.decision == DecisionType.REFUSE
    assert parsed_back.sql is None
    assert "READ_ONLY" in parsed_back.policy_violations
    assert "Request requires TRUNCATE" in parsed_back.rationale
