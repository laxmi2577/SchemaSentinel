import importlib
import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

tabular_module = importlib.import_module("colab.04_tabular_ml_trainer")
TabularTrainerConfig = tabular_module.TabularTrainerConfig
TabularTrainingSummary = tabular_module.TabularTrainingSummary
TabularMLTrainer = tabular_module.TabularMLTrainer
parse_args = tabular_module.parse_args
main = tabular_module.main
detect_environment = tabular_module.detect_environment
mount_google_drive = tabular_module.mount_google_drive
ensure_dependencies = tabular_module.ensure_dependencies
DriveMountConfig = tabular_module.DriveMountConfig


def test_trainer_config_defaults():
    cfg = TabularTrainerConfig()
    assert cfg.output_onnx_path == "models/anomaly_detector.onnx"
    assert cfg.focal_loss_gamma == 2.0
    assert cfg.focal_loss_alpha == 0.25
    assert cfg.max_model_size_bytes == 15 * 1024 * 1024
    assert cfg.save_drive_copy is True
    assert cfg.save_summary_json is True


def test_generate_synthetic_dataset():
    trainer = TabularMLTrainer(TabularTrainerConfig(n_samples=100, contamination=0.10))
    X, y = trainer.generate_synthetic_dataset(n_samples=100)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (100, 7)
    assert y.shape == (100,)
    assert set(np.unique(y)).issubset({0, 1})
    assert np.sum(y == 1) == 10


def test_focal_loss_computation():
    trainer = TabularMLTrainer()
    y_true = np.array([1, 0, 1, 0])

    good_preds = np.array([0.9, 0.1, 0.85, 0.05])
    bad_preds = np.array([0.1, 0.9, 0.2, 0.8])

    loss_good = trainer.compute_focal_loss(y_true, good_preds)
    loss_bad = trainer.compute_focal_loss(y_true, bad_preds)

    assert loss_good < loss_bad
    assert loss_good > 0.0


def test_hyperparameter_optimization():
    trainer = TabularMLTrainer()
    X, y = trainer.generate_synthetic_dataset(n_samples=50)
    best_params = trainer.optimize_hyperparameters(X, y, n_trials=2)

    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert best_params["n_estimators"] in (5, 10, 15)


def test_train_and_export_dry_run(tmp_path: Path):
    onnx_file = tmp_path / "models" / "test_detector.onnx"
    summary_json = tmp_path / "reports" / "summary.json"
    cfg = TabularTrainerConfig(
        output_onnx_path=str(onnx_file),
        summary_json_path=str(summary_json),
        n_samples=40,
        n_trials=2,
        dry_run=True,
    )
    trainer = TabularMLTrainer(cfg)
    summary = trainer.train_and_export(dry_run=True)

    assert isinstance(summary, TabularTrainingSummary)
    assert summary.model_type == "RandomForest_AnomalyDetector"
    assert onnx_file.exists()
    assert summary.onnx_size_bytes > 0
    assert summary.onnx_size_bytes < cfg.max_model_size_bytes
    assert summary.is_within_size_budget is True
    assert summary.train_accuracy >= 0.70
    assert summary_json.exists()

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_type"] == "RandomForest_AnomalyDetector"
    assert data["is_within_size_budget"] is True


def test_ensure_dependencies():
    results = ensure_dependencies(auto_install=False)
    assert isinstance(results, dict)
    assert "onnx" in results


def test_detect_environment():
    env = detect_environment(dry_run=True)
    assert isinstance(env, dict)
    assert "python_version" in env
    assert "is_colab" in env
    assert env["dry_run"] is True


def test_mount_google_drive():
    assert mount_google_drive(dry_run=True) is True
    assert mount_google_drive(dry_run=False) is False


def test_parse_args_with_jupyter_kernel_flags():
    kernel_args = [
        "-f",
        "/root/.local/share/jupyter/runtime/kernel-12345.json",
        "--dry-run",
        "--output",
        "custom_models/detector.onnx",
        "--n-samples",
        "120",
    ]
    parsed = parse_args(kernel_args)
    assert parsed.dry_run is True
    assert parsed.output == "custom_models/detector.onnx"
    assert parsed.n_samples == 120


def test_load_dataset_or_fallback(tmp_path: Path):
    trainer = TabularMLTrainer()
    X_fallback, y_fallback = trainer.load_dataset_or_fallback(
        dataset_path=str(tmp_path / "non_existent.parquet"),
        n_samples=30,
    )
    assert len(X_fallback) == 30

    parquet_file = tmp_path / "offline_features.parquet"
    df = pl.DataFrame(
        {
            "query_velocity": [1.0, 2.0, 25.0],
            "p99_latency_ms": [10.0, 15.0, 250.0],
            "z_score": [0.1, 0.2, 5.0],
            "error_rate": [0.0, 0.01, 0.8],
            "schema_complexity_delta": [0.1, 0.2, 8.0],
            "num_tables": [2, 3, 10],
            "num_tokens": [20.0, 30.0, 400.0],
            "is_anomaly": [0, 0, 1],
        }
    )
    df.write_parquet(parquet_file)

    X_loaded, y_loaded = trainer.load_dataset_or_fallback(
        dataset_path=str(parquet_file)
    )
    assert len(X_loaded) == 3
    assert X_loaded.shape == (3, 7)
    assert list(y_loaded) == [0, 0, 1]


def test_main_cli_execution(tmp_path: Path):
    target_onnx = tmp_path / "cli_detector.onnx"
    exit_code = main(["--dry-run", "--output", str(target_onnx)])
    assert exit_code == 0
    assert target_onnx.exists()
