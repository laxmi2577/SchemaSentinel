import time
from pathlib import Path
import numpy as np
import polars as pl
import pytest
from src.ml_engine import (
    FeatureStoreConfig,
    FeatureVector,
    StreamingFeatureStore,
)
from src.stream_lakehouse import StreamEvent


def test_feature_store_config_defaults():
    cfg = FeatureStoreConfig()
    assert 60 in cfg.windows_seconds
    assert 300 in cfg.windows_seconds
    assert 86400 in cfg.windows_seconds
    assert cfg.offline_storage_dir == "data/feature_store"


def test_ingest_event_and_compute_features():
    fs = StreamingFeatureStore()
    now = time.time()
    events = [
        StreamEvent(
            tenant_id="tenant_alpha",
            timestamp=now - 10.0,
            latency_ms=20.0,
            tokens_used=50,
            is_valid=True,
            tables_accessed=["users", "orders"],
        ),
        StreamEvent(
            tenant_id="tenant_alpha",
            timestamp=now - 5.0,
            latency_ms=40.0,
            tokens_used=70,
            is_valid=True,
            tables_accessed=["orders"],
        ),
    ]
    fs.ingest_batch(events)

    fv = fs.compute_features(tenant_id="tenant_alpha", window_seconds=60, current_time=now)
    assert isinstance(fv, FeatureVector)
    assert fv.tenant_id == "tenant_alpha"
    assert fv.query_velocity == round(2.0 / 60.0, 4)
    assert fv.mean_latency_ms == 30.0
    assert fv.distinct_tables_count == 2
    assert fv.error_rate == 0.0
    assert fv.token_consumption_rate == round(120.0 / 60.0, 2)


def test_compute_multi_window_features():
    fs = StreamingFeatureStore()
    now = time.time()
    fs.ingest_event(
        StreamEvent(
            tenant_id="tenant_multi",
            timestamp=now - 20.0,
            latency_ms=15.0,
            tokens_used=30,
        )
    )

    windows_dict = fs.compute_multi_window_features(tenant_id="tenant_multi", current_time=now)
    assert 60 in windows_dict
    assert 300 in windows_dict
    assert 900 in windows_dict
    assert windows_dict[60].mean_latency_ms == 15.0
    assert windows_dict[300].mean_latency_ms == 15.0


def test_rolling_z_score_calculation():
    fs = StreamingFeatureStore()
    now = time.time()

    for i in range(20):
        fs.ingest_event(
            StreamEvent(
                tenant_id="tenant_z",
                timestamp=now - 50.0 + i,
                latency_ms=10.0 + (i % 2),
                tokens_used=10,
            )
        )

    fs.ingest_event(
        StreamEvent(
            tenant_id="tenant_z",
            timestamp=now - 2.0,
            latency_ms=100.0,
            tokens_used=10,
        )
    )

    fv = fs.compute_features(tenant_id="tenant_z", window_seconds=5, current_time=now)
    assert fv.latency_z_score > 3.0


def test_to_numpy_matrix():
    fs = StreamingFeatureStore()
    vectors = [
        FeatureVector(
            tenant_id="t1",
            query_velocity=1.5,
            mean_latency_ms=25.0,
            latency_z_score=0.5,
            error_rate=0.01,
            schema_complexity_delta=1.0,
            distinct_tables_count=3,
            token_consumption_rate=50.0,
        ),
        FeatureVector(
            tenant_id="t2",
            query_velocity=2.0,
            mean_latency_ms=35.0,
            latency_z_score=1.5,
            error_rate=0.05,
            schema_complexity_delta=2.0,
            distinct_tables_count=5,
            token_consumption_rate=80.0,
        ),
    ]

    mat = fs.to_numpy_matrix(vectors)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (2, 7)
    assert mat.dtype == np.float32

    empty_mat = fs.to_numpy_matrix([])
    assert empty_mat.shape == (0, 7)


def test_persist_and_read_offline_features(tmp_path: Path):
    fs = StreamingFeatureStore(
        FeatureStoreConfig(offline_storage_dir=str(tmp_path / "offline_fs"))
    )
    vectors = [
        FeatureVector(tenant_id="tenant_persist", mean_latency_ms=12.5),
        FeatureVector(tenant_id="tenant_persist", mean_latency_ms=25.0),
    ]

    paths = fs.persist_offline_features(vectors, date_str="2026-10-15")
    assert len(paths) == 1
    assert Path(paths[0]).exists()

    df = fs.read_offline_features(tenant_id="tenant_persist")
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 2
    assert "mean_latency_ms" in df.columns


def test_empty_tenant_features():
    fs = StreamingFeatureStore()
    fv = fs.compute_features("non_existent_tenant")
    assert fv.query_velocity == 0.0
    assert fv.mean_latency_ms == 0.0
    assert fv.distinct_tables_count == 0
