from pathlib import Path
import pytest
from src.ml_engine import (
    AnomalyScore,
    FeatureVector,
    LocalONNXScorer,
    ScorerConfig,
)


def test_scorer_config_defaults():
    cfg = ScorerConfig()
    assert cfg.model_path == "models/anomaly_detector.onnx"
    assert cfg.anomaly_threshold == 0.50
    assert cfg.calibrated_threshold == 0.65
    assert cfg.execution_provider == "CPUExecutionProvider"


def test_score_single_feature_vector(tmp_path: Path):
    model_file = tmp_path / "models" / "scorer_test.onnx"
    cfg = ScorerConfig(model_path=str(model_file))
    scorer = LocalONNXScorer(cfg)

    fv = FeatureVector(
        tenant_id="tenant_eval",
        query_velocity=2.5,
        mean_latency_ms=18.0,
        latency_z_score=0.2,
        error_rate=0.0,
        schema_complexity_delta=0.0,
        distinct_tables_count=2,
        token_consumption_rate=45.0,
    )
    score = scorer.score_features(fv)

    assert isinstance(score, AnomalyScore)
    assert score.tenant_id == "tenant_eval"
    assert 0.0 <= score.anomaly_score <= 1.0
    assert isinstance(score.is_anomaly, bool)
    assert score.inference_latency_ms >= 0.0
    assert "mean_latency_ms" in score.contributing_features


def test_score_batch(tmp_path: Path):
    model_file = tmp_path / "models" / "batch_scorer_test.onnx"
    scorer = LocalONNXScorer(ScorerConfig(model_path=str(model_file)))

    vectors = [
        FeatureVector(tenant_id="t1", mean_latency_ms=15.0),
        FeatureVector(tenant_id="t2", mean_latency_ms=250.0, latency_z_score=5.0, error_rate=0.8),
    ]

    scores = scorer.score_batch(vectors)
    assert len(scores) == 2
    assert scores[0].tenant_id == "t1"
    assert scores[1].tenant_id == "t2"
    assert scores[0].inference_latency_ms >= 0.0


def test_calibrate_threshold(tmp_path: Path):
    model_file = tmp_path / "models" / "calib_test.onnx"
    scorer = LocalONNXScorer(ScorerConfig(model_path=str(model_file)))

    val_vectors = [
        FeatureVector(tenant_id=f"t_{i}", mean_latency_ms=10.0 + i)
        for i in range(15)
    ]
    calibrated = scorer.calibrate_threshold(val_vectors, target_fpr=0.10)
    assert 0.0 <= calibrated <= 1.0
    assert scorer.calibrated_threshold == calibrated


def test_check_latency_sla(tmp_path: Path):
    model_file = tmp_path / "models" / "sla_test.onnx"
    scorer = LocalONNXScorer(ScorerConfig(model_path=str(model_file)))

    fv = FeatureVector(tenant_id="tenant_sla", mean_latency_ms=20.0)
    meets_sla = scorer.check_latency_sla(fv, n_warmup=2, n_iterations=20, p99_sla_ms=15.0)
    assert meets_sla is True


def test_empty_batch(tmp_path: Path):
    model_file = tmp_path / "models" / "empty_test.onnx"
    scorer = LocalONNXScorer(ScorerConfig(model_path=str(model_file)))
    assert scorer.score_batch([]) == []
    assert scorer.calibrate_threshold([]) == scorer.calibrated_threshold
