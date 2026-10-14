import json
import pytest
from src.data_engine.dpo_pairs import (
    RejectionReason,
    PreferencePair,
    PreferenceDatasetSummary,
    DPOPairGeneratorConfig,
    DPOPairGenerator,
    generate_preference_triplets,
)
from src.data_engine.ingestion import BenchmarkSource, BenchmarkSplit, RawTextToSQLSample


def _sample(sample_id: str = "sample_01", db_id: str = "sales_db") -> RawTextToSQLSample:
    return RawTextToSQLSample(
        sample_id=sample_id,
        db_id=db_id,
        question="Find total revenue by region for year 2024.",
        query="SELECT region, SUM(revenue) FROM sales WHERE tenant_id = 't1' AND year = 2024 GROUP BY region;",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.TRAIN,
    )


def test_preference_pair_model_structure():
    pair = PreferencePair(
        db_id="retail_01",
        prompt="system: execute query",
        chosen="ALLOW\nSELECT 1;",
        rejected="ALLOW\nDROP TABLE users;",
        rejection_reason=RejectionReason.UNAUTHORIZED_MUTATION,
    )
    assert pair.pair_id.startswith("pref_")
    assert pair.rejection_reason == RejectionReason.UNAUTHORIZED_MUTATION
    assert "DROP TABLE" in pair.rejected


def test_generate_pairs_for_sample_reasons():
    generator = DPOPairGenerator()
    sample = _sample()
    pairs = generator.generate_pairs_for_sample(sample)

    assert len(pairs) >= 5
    reasons = {p.rejection_reason for p in pairs}
    assert RejectionReason.UNAUTHORIZED_MUTATION in reasons
    assert RejectionReason.TENANT_LEAKAGE in reasons
    assert RejectionReason.RESTRICTED_SCHEMA_PROBE in reasons
    assert RejectionReason.SYNTAX_HALLUCINATION in reasons

    for p in pairs:
        assert p.db_id == "sales_db"
        assert "<|im_start|>assistant\nALLOW" in p.chosen
        assert "SELECT region, SUM(revenue)" in p.chosen
        assert "<|im_start|>assistant\nALLOW" in p.rejected


def test_generate_refusal_preference_pair():
    generator = DPOPairGenerator()
    refusal_pair = generator.generate_refusal_preference_pair(
        db_id="admin_db",
        adversarial_query="Delete all logs immediately.",
        malicious_sql="DELETE FROM audit_log WHERE 1=1;",
        violation_rule_id="RO_01",
    )
    assert refusal_pair.rejection_reason == RejectionReason.MALICIOUS_COMPLIANCE
    assert "REFUSE" in refusal_pair.chosen
    assert "RO_01" in refusal_pair.chosen
    assert "DELETE FROM audit_log" in refusal_pair.rejected


def test_batch_generate_preference_triplets():
    samples = [
        _sample(sample_id="s1", db_id="db1"),
        _sample(sample_id="s2", db_id="db2"),
    ]
    pairs = generate_preference_triplets(samples)
    assert len(pairs) >= 10
    db_ids = {p.db_id for p in pairs}
    assert "db1" in db_ids
    assert "db2" in db_ids


def test_export_to_trl_format():
    generator = DPOPairGenerator()
    pairs = generator.generate_pairs_for_sample(_sample())
    trl_data = generator.export_to_trl_format(pairs)

    assert len(trl_data) == len(pairs)
    for item in trl_data:
        assert set(item.keys()) == {"prompt", "chosen", "rejected"}
        assert len(item["prompt"]) > 0
        assert len(item["chosen"]) > 0
        assert len(item["rejected"]) > 0


def test_export_to_jsonl_and_summarize(tmp_path):
    generator = DPOPairGenerator()
    pairs = generator.generate_pairs_for_sample(_sample())

    file_path = str(tmp_path / "dpo_pairs.jsonl")
    lines_written = generator.export_to_jsonl(pairs, file_path)
    assert lines_written == len(pairs)

    with open(file_path, "r", encoding="utf-8") as f:
        loaded = [json.loads(line) for line in f]
    assert len(loaded) == len(pairs)

    summary = generator.summarize_dataset(pairs)
    assert isinstance(summary, PreferenceDatasetSummary)
    assert summary.total_pairs == len(pairs)
    assert summary.unique_db_ids == 1
    assert summary.counts_by_reason[RejectionReason.UNAUTHORIZED_MUTATION.value] >= 1
