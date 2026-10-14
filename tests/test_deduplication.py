import pytest
from src.data_engine.ingestion import (
    BenchmarkSource,
    BenchmarkSplit,
    RawTextToSQLSample,
)
from src.data_engine.deduplication import (
    MinHashConfig,
    MinHashSignature,
    LSHIndex,
    BenchmarkContaminationDetector,
)


def test_minhash_signature_identical_and_disjoint():
    config = MinHashConfig(num_perm=128, seed=42)
    hasher = MinHashSignature(config)

    text_a = "Find all departments located in the San Francisco building."
    text_b = "Find all departments located in the San Francisco building."
    text_c = "Calculate maximum employee salaries across each fiscal quarter."

    sig_a = hasher.compute_signature(text_a)
    sig_b = hasher.compute_signature(text_b)
    sig_c = hasher.compute_signature(text_c)

    assert len(sig_a) == 128
    assert sig_a == sig_b
    assert hasher.estimate_jaccard(sig_a, sig_b) == 1.0
    assert hasher.estimate_jaccard(sig_a, sig_c) < 0.2


def test_minhash_near_duplicates_similarity():
    config = MinHashConfig(num_perm=128, seed=42)
    hasher = MinHashSignature(config)

    text1 = "What is the total number of students enrolled in computer science?"
    text2 = "What is the total number of students enrolled in the computer science department?"

    sig1 = hasher.compute_signature(text1)
    sig2 = hasher.compute_signature(text2)
    sim = hasher.estimate_jaccard(sig1, sig2)

    assert sim >= 0.60
    assert sim < 1.0


def test_lsh_index_candidate_retrieval():
    config = MinHashConfig(num_perm=128, num_bands=16, seed=42)
    hasher = MinHashSignature(config)
    lsh = LSHIndex(config)

    q1 = "What is the average transaction amount for customers located in New York City?"
    q2 = "  what is the average transaction amount for customers located in new york city?  "
    q3 = "Show flight delays for Chicago O'Hare airport yesterday."

    sig1 = hasher.compute_signature(q1)
    sig2 = hasher.compute_signature(q2)
    sig3 = hasher.compute_signature(q3)

    lsh.insert("sample_1", sig1)
    lsh.insert("sample_3", sig3)

    candidates = lsh.query_candidates(sig2)
    assert "sample_1" in candidates
    assert "sample_3" not in candidates


def test_benchmark_contamination_detector_leakage():
    detector = BenchmarkContaminationDetector(MinHashConfig(threshold=0.80))

    bench_sample = RawTextToSQLSample(
        sample_id="spider_dev_01",
        db_id="car_1",
        question="Which car model has the highest horsepower produced after 2010?",
        query="SELECT model FROM cars WHERE year > 2010 ORDER BY hp DESC LIMIT 1",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.DEV,
    )
    detector.register_benchmark_samples([bench_sample])

    clean_train = RawTextToSQLSample(
        sample_id="train_01",
        db_id="flight_2",
        question="What is the average duration of domestic flights?",
        query="SELECT avg(duration) FROM flights",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.TRAIN,
    )
    assert detector.check_leakage(clean_train) is None

    leaked_train = RawTextToSQLSample(
        sample_id="train_02",
        db_id="car_1",
        question="Which car model has the highest horsepower produced after 2010?",
        query="SELECT model FROM cars WHERE year > 2010 ORDER BY hp DESC LIMIT 1",
        source=BenchmarkSource.SPIDER,
        split=BenchmarkSplit.TRAIN,
    )
    violation = detector.check_leakage(leaked_train)
    assert violation is not None
    assert violation.train_sample_id == "train_02"
    assert violation.benchmark_sample_id == "spider_dev_01"
    assert violation.similarity >= 0.80


def test_sanitize_training_stream_end_to_end():
    detector = BenchmarkContaminationDetector(MinHashConfig(threshold=0.85))

    bench_sample = RawTextToSQLSample(
        sample_id="gold_bench_01",
        db_id="store",
        question="Find customer names who purchased laptops.",
        query="SELECT c.name FROM customers c JOIN orders o ...",
        source=BenchmarkSource.BIRD,
        split=BenchmarkSplit.TEST,
    )
    detector.register_benchmark_samples([bench_sample])

    train_samples = [
        RawTextToSQLSample(
            sample_id="t1",
            db_id="store",
            question="Find customer names who purchased laptops.",
            query="SELECT c.name ...",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
        RawTextToSQLSample(
            sample_id="t2",
            db_id="store",
            question="List all product names and inventory levels.",
            query="SELECT name, stock FROM products",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
        RawTextToSQLSample(
            sample_id="t3",
            db_id="store",
            question="List all product names and inventory levels.",
            query="SELECT name, stock FROM products",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
        RawTextToSQLSample(
            sample_id="t4",
            db_id="store",
            question="Calculate total revenue generated in December.",
            query="SELECT sum(amount) FROM sales ...",
            source=BenchmarkSource.SPIDER,
            split=BenchmarkSplit.TRAIN,
        ),
    ]

    sanitized, summary = detector.sanitize_training_stream(train_samples, deduplicate_internal=True)

    assert summary.total_records == 4
    assert summary.quarantined_benchmark_leakages == 1
    assert summary.internal_duplicates_dropped == 1
    assert summary.unique_records == 2

    assert len(sanitized) == 2
    assert sanitized[0].sample_id == "t2"
    assert sanitized[1].sample_id == "t4"
