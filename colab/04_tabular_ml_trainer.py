from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if "/content" not in sys.path:
    sys.path.insert(0, "/content")
if "/content/drive/MyDrive/SchemaSentinel" not in sys.path:
    sys.path.insert(0, "/content/drive/MyDrive/SchemaSentinel")

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier


def ensure_dependencies(auto_install: bool = True) -> Dict[str, bool]:
    results = {}
    for pkg in ["skl2onnx", "onnx", "onnxruntime"]:
        try:
            importlib.import_module(pkg)
            results[pkg] = True
        except ImportError:
            results[pkg] = False
            if auto_install:
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", pkg],
                        check=False,
                    )
                    importlib.import_module(pkg)
                    results[pkg] = True
                except Exception:
                    pass
    return results


def _get_onnx_converter() -> Tuple[Any, Any]:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        return convert_sklearn, FloatTensorType
    except ImportError:
        ensure_dependencies(auto_install=True)
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        return convert_sklearn, FloatTensorType


class DriveMountConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mount_point: str = "/content/drive"
    target_dir: str = "/content/drive/MyDrive/SchemaSentinel"
    models_dir: str = "/content/drive/MyDrive/SchemaSentinel/models"
    datasets_dir: str = "/content/drive/MyDrive/SchemaSentinel/datasets"
    logs_dir: str = "/content/drive/MyDrive/SchemaSentinel/logs"


def mount_google_drive(
    mount_point: str = "/content/drive",
    dry_run: bool = False,
) -> bool:
    if dry_run:
        return True
    if "google.colab" in sys.modules:
        try:
            from google.colab import drive

            drive.mount(mount_point, force_remount=False)
            return True
        except Exception:
            return False
    return False


def detect_environment(dry_run: bool = False) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "is_colab": "google.colab" in sys.modules or os.path.exists("/content"),
        "cpu_count": os.cpu_count() or 1,
        "drive_mounted": os.path.exists("/content/drive/MyDrive"),
        "dry_run": dry_run,
    }


class TabularTrainerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_onnx_path: str = "models/anomaly_detector.onnx"
    dataset_path: Optional[str] = None
    n_samples: int = 500
    n_trials: int = 5
    contamination: float = 0.05
    random_state: int = 42
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    max_model_size_bytes: int = 15 * 1024 * 1024
    use_xgboost: bool = False
    dry_run: bool = False
    save_drive_copy: bool = True
    drive_models_dir: str = "/content/drive/MyDrive/SchemaSentinel/models"
    save_summary_json: bool = True
    summary_json_path: Optional[str] = None


class TabularTrainingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_type: str
    best_params: Dict[str, Any]
    focal_loss: float
    train_accuracy: float
    onnx_output_path: str
    onnx_size_bytes: int
    is_within_size_budget: bool
    drive_copied_path: Optional[str] = None
    summary_json_path: Optional[str] = None
    duration_seconds: float = 0.0


class TabularMLTrainer:
    def __init__(self, config: Optional[TabularTrainerConfig] = None) -> None:
        self.config = config or TabularTrainerConfig()
        np.random.seed(self.config.random_state)

    def generate_synthetic_dataset(
        self,
        n_samples: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.random_state)
        n_anomalies = max(2, int(n_samples * self.config.contamination))
        n_nominal = n_samples - n_anomalies

        nominal_velocity = rng.uniform(0.5, 5.0, size=(n_nominal, 1))
        nominal_latency = rng.normal(15.0, 3.0, size=(n_nominal, 1))
        nominal_z_score = rng.normal(0.0, 0.8, size=(n_nominal, 1))
        nominal_error_rate = rng.uniform(0.0, 0.02, size=(n_nominal, 1))
        nominal_complexity = rng.uniform(0.0, 1.0, size=(n_nominal, 1))
        nominal_tables = rng.integers(1, 5, size=(n_nominal, 1))
        nominal_tokens = rng.uniform(10.0, 50.0, size=(n_nominal, 1))

        X_nominal = np.hstack(
            [
                nominal_velocity,
                nominal_latency,
                nominal_z_score,
                nominal_error_rate,
                nominal_complexity,
                nominal_tables,
                nominal_tokens,
            ]
        )
        y_nominal = np.zeros(n_nominal, dtype=np.int64)

        anomaly_velocity = rng.uniform(10.0, 50.0, size=(n_anomalies, 1))
        anomaly_latency = rng.uniform(80.0, 500.0, size=(n_anomalies, 1))
        anomaly_z_score = rng.uniform(3.5, 12.0, size=(n_anomalies, 1))
        anomaly_error_rate = rng.uniform(0.3, 1.0, size=(n_anomalies, 1))
        anomaly_complexity = rng.uniform(5.0, 15.0, size=(n_anomalies, 1))
        anomaly_tables = rng.integers(6, 20, size=(n_anomalies, 1))
        anomaly_tokens = rng.uniform(150.0, 1000.0, size=(n_anomalies, 1))

        X_anomaly = np.hstack(
            [
                anomaly_velocity,
                anomaly_latency,
                anomaly_z_score,
                anomaly_error_rate,
                anomaly_complexity,
                anomaly_tables,
                anomaly_tokens,
            ]
        )
        y_anomaly = np.ones(n_anomalies, dtype=np.int64)

        X = np.vstack([X_nominal, X_anomaly]).astype(np.float32)
        y = np.concatenate([y_nominal, y_anomaly])

        indices = np.arange(len(y))
        rng.shuffle(indices)
        return X[indices], y[indices]

    def load_dataset_or_fallback(
        self,
        dataset_path: Optional[str] = None,
        n_samples: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_path = dataset_path or self.config.dataset_path
        if target_path and Path(target_path).exists():
            try:
                ext = Path(target_path).suffix.lower()
                if ext == ".parquet":
                    import polars as pl

                    df = pl.read_parquet(target_path)
                    feature_cols = [
                        c
                        for c in [
                            "query_velocity",
                            "p99_latency_ms",
                            "z_score",
                            "error_rate",
                            "schema_complexity_delta",
                            "num_tables",
                            "num_tokens",
                        ]
                        if c in df.columns
                    ]
                    if len(feature_cols) == 7:
                        X = df.select(feature_cols).to_numpy().astype(np.float32)
                        if "is_anomaly" in df.columns:
                            y = df["is_anomaly"].cast(pl.Int64).to_numpy()
                        elif "label" in df.columns:
                            y = df["label"].cast(pl.Int64).to_numpy()
                        else:
                            y = np.zeros(len(X), dtype=np.int64)
                        return X, y
            except Exception:
                pass
        return self.generate_synthetic_dataset(n_samples=n_samples)

    def compute_focal_loss(
        self,
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
    ) -> float:
        pt = np.where(y_true == 1, y_pred_prob, 1.0 - y_pred_prob)
        pt = np.clip(pt, 1e-7, 1.0 - 1e-7)
        alpha_t = np.where(
            y_true == 1,
            self.config.focal_loss_alpha,
            1.0 - self.config.focal_loss_alpha,
        )
        loss = -alpha_t * np.power(1.0 - pt, self.config.focal_loss_gamma) * np.log(pt)
        return float(np.mean(loss))

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 3,
    ) -> Dict[str, Any]:
        candidates = [
            {"n_estimators": 5, "max_depth": 2},
            {"n_estimators": 10, "max_depth": 3},
            {"n_estimators": 15, "max_depth": 4},
        ]

        best_params = candidates[0]
        best_loss = float("inf")

        for params in candidates[:n_trials]:
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=self.config.random_state,
            )
            clf.fit(X, y)
            probs = clf.predict_proba(X)[:, 1]
            loss = self.compute_focal_loss(y, probs)
            if loss < best_loss:
                best_loss = loss
                best_params = params

        return best_params

    def export_onnx(
        self,
        model: Any,
        output_path: str,
        n_features: int = 7,
    ) -> Tuple[str, int]:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        convert_sklearn, FloatTensorType = _get_onnx_converter()
        initial_types = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_types)
        serialized = onnx_model.SerializeToString()
        size_bytes = len(serialized)

        with open(target_path, "wb") as f:
            f.write(serialized)

        return str(target_path), size_bytes

    def train_and_export(
        self,
        dry_run: bool = False,
        dataset_path: Optional[str] = None,
    ) -> TabularTrainingSummary:
        start_time = time.perf_counter()
        effective_dry = dry_run or self.config.dry_run
        n_samples = 50 if effective_dry else self.config.n_samples
        n_trials = 2 if effective_dry else self.config.n_trials

        X, y = self.load_dataset_or_fallback(
            dataset_path=dataset_path,
            n_samples=n_samples,
        )
        best_params = self.optimize_hyperparameters(X, y, n_trials=n_trials)

        clf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=self.config.random_state,
        )
        clf.fit(X, y)

        preds = clf.predict(X)
        probs = clf.predict_proba(X)[:, 1]

        acc = float(np.mean(preds == y))
        loss = self.compute_focal_loss(y, probs)

        out_path = self.config.output_onnx_path
        exported_path, size_bytes = self.export_onnx(clf, out_path)

        is_within_budget = size_bytes <= self.config.max_model_size_bytes

        drive_copied_path = None
        if self.config.save_drive_copy and not effective_dry:
            drive_dir = Path(self.config.drive_models_dir)
            if drive_dir.parent.exists() or Path("/content/drive/MyDrive").exists():
                try:
                    drive_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = drive_dir / Path(exported_path).name
                    shutil.copy2(exported_path, str(dest_file))
                    drive_copied_path = str(dest_file)
                except Exception:
                    pass

        duration = round(time.perf_counter() - start_time, 3)

        summary_json_path = None
        if self.config.save_summary_json:
            target_json = self.config.summary_json_path or str(
                Path(exported_path).parent / "tabular_training_summary.json"
            )
            try:
                Path(target_json).parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model_type": "RandomForest_AnomalyDetector",
                    "best_params": best_params,
                    "focal_loss": round(loss, 6),
                    "train_accuracy": round(acc, 4),
                    "onnx_output_path": exported_path,
                    "onnx_size_bytes": size_bytes,
                    "is_within_size_budget": is_within_budget,
                    "drive_copied_path": drive_copied_path,
                    "duration_seconds": duration,
                    "environment": detect_environment(dry_run=effective_dry),
                }
                with open(target_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                summary_json_path = target_json
            except Exception:
                pass

        return TabularTrainingSummary(
            model_type="RandomForest_AnomalyDetector",
            best_params=best_params,
            focal_loss=round(loss, 6),
            train_accuracy=round(acc, 4),
            onnx_output_path=exported_path,
            onnx_size_bytes=size_bytes,
            is_within_size_budget=is_within_budget,
            drive_copied_path=drive_copied_path,
            summary_json_path=summary_json_path,
            duration_seconds=duration,
        )


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SchemaSentinel Colab Tabular ML Trainer with Focal Loss"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run quick dry-run training with synthetic data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/anomaly_detector.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to input Parquet feature store dataset",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of synthetic samples when data-path is omitted",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of hyperparameter optimization trials",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Anomaly contamination ratio",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Mount Google Drive in Colab before training",
    )
    parser.add_argument(
        "--no-drive-copy",
        action="store_true",
        help="Skip copying ONNX model to Google Drive",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Path to write JSON training summary report",
    )
    parsed, _ = parser.parse_known_args(args)
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.mount_drive:
        mount_google_drive(dry_run=args.dry_run)

    cfg = TabularTrainerConfig(
        output_onnx_path=args.output,
        dataset_path=args.data_path,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        contamination=args.contamination,
        dry_run=args.dry_run,
        save_drive_copy=not args.no_drive_copy,
        summary_json_path=args.summary_json,
    )
    trainer = TabularMLTrainer(cfg)
    summary = trainer.train_and_export(
        dry_run=args.dry_run,
        dataset_path=args.data_path,
    )
    print(f"Tabular Model Training Completed: {summary.model_type}")
    print(f"ONNX Model Exported: {summary.onnx_output_path} ({summary.onnx_size_bytes} bytes)")
    print(f"Train Accuracy: {summary.train_accuracy}, Focal Loss: {summary.focal_loss}")
    if summary.drive_copied_path:
        print(f"Drive Model Persisted: {summary.drive_copied_path}")
    if summary.summary_json_path:
        print(f"Summary JSON Written: {summary.summary_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
