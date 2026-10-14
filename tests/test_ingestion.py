import pytest
from src.data_engine.ingestion import (
    BenchmarkSplit,
    BenchmarkSource,
    RawTextToSQLSample,
    DatasetStreamConfig,
    SpiderDatasetStreamer,
    BIRDDatasetStreamer,
    UnifiedDataEngineStreamer,
)


def test_spider_streamer_parse_item():
    streamer = SpiderDatasetStreamer()
    raw_item = {
        "db_id": "concert_singer",
        "question": "How many singers do we have?",
        "query": "SELECT count(*) FROM singer",
        "difficulty": "easy",
    }
    sample = streamer.parse_raw_item(raw_item, BenchmarkSplit.TRAIN, index=42)

    assert sample.sample_id == "spider_train_concert_singer_42"
    assert sample.db_id == "concert_singer"
    assert sample.question == "How many singers do we have?"
    assert sample.query == "SELECT count(*) FROM singer"
    assert sample.source == BenchmarkSource.SPIDER
    assert sample.split == BenchmarkSplit.TRAIN
    assert sample.difficulty == "easy"


def test_spider_streamer_filtering_and_max_samples():
    records = [
        {"db_id": "db_a", "question": "Q1", "query": "SELECT 1"},
        {"db_id": "db_b", "question": "Q2", "query": "SELECT 2"},
        {"db_id": "db_a", "question": "Q3", "query": "SELECT 3"},
        {"db_id": "db_a", "question": "Q4", "query": "SELECT 4"},
    ]
    config = DatasetStreamConfig(max_samples=2, filter_db_ids=["db_a"])
    streamer = SpiderDatasetStreamer(config=config)

    samples = list(streamer.stream_from_records(records, BenchmarkSplit.DEV))
    assert len(samples) == 2
    assert samples[0].question == "Q1"
    assert samples[1].question == "Q3"
    assert all(s.db_id == "db_a" for s in samples)


def test_bird_streamer_parse_item_with_evidence():
    streamer = BIRDDatasetStreamer()
    raw_item = {
        "db_id": "financial",
        "question": "What is the average transaction amount for account 1?",
        "SQL": "SELECT avg(amount) FROM trans WHERE account_id = 1",
        "evidence": "trans refers to the transactions table.",
        "difficulty": "moderate",
    }
    sample = streamer.parse_raw_item(raw_item, BenchmarkSplit.TRAIN, index=10)

    assert sample.sample_id == "bird_train_financial_10"
    assert sample.db_id == "financial"
    assert sample.query == "SELECT avg(amount) FROM trans WHERE account_id = 1"
    assert sample.evidence == "trans refers to the transactions table."
    assert sample.source == BenchmarkSource.BIRD
    assert sample.difficulty == "moderate"


def test_bird_streamer_schema_catalog_parsing():
    streamer = BIRDDatasetStreamer()
    schemas_data = [
        {
            "db_id": "hospital",
            "table_names_original": ["doctor", "patient"],
            "column_names_original": [
                [-1, "*"],
                [0, "doc_id"],
                [0, "doc_name"],
                [1, "pat_id"],
                [1, "pat_name"],
            ],
            "column_types": ["text", "number", "text", "number", "text"],
            "primary_keys": [1, 3],
            "foreign_keys": [],
        }
    ]
    catalogs = streamer.parse_schemas_from_dict_list(schemas_data)
    assert "hospital" in catalogs
    catalog = catalogs["hospital"]
    assert catalog.has_table("doctor")
    assert catalog.has_table("patient")


def test_unified_streamer_mixed_samples():
    spider_data = [
        {"db_id": "s_db", "question": "SQ1", "query": "SELECT 1"},
        {"db_id": "s_db", "question": "SQ2", "query": "SELECT 2"},
    ]
    bird_data = [
        {"db_id": "b_db", "question": "BQ1", "SQL": "SELECT 10"},
        {"db_id": "b_db", "question": "BQ2", "SQL": "SELECT 20"},
    ]

    config = DatasetStreamConfig(max_samples=3)
    unified = UnifiedDataEngineStreamer(config=config)
    mixed = list(unified.stream_mixed_samples(spider_data, bird_data, BenchmarkSplit.TRAIN))

    assert len(mixed) == 3
    assert mixed[0].source == BenchmarkSource.SPIDER
    assert mixed[1].source == BenchmarkSource.BIRD
    assert mixed[2].source == BenchmarkSource.SPIDER
