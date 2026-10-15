import importlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import onnxruntime as ort

from src.ml_engine import AnomalyScore, FeatureVector, ScorerConfig


class LocalONNXScorer:
    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self.config = config or ScorerConfig()
        self.calibrated_threshold = float(self.config.calibrated_threshold)
        self.model_path = Path(self.config.model_path)
        self._ensure_model_exists()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.config.intra_op_num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=[self.config.execution_provider],
        )
        self.input_name = self._session.get_inputs()[0].name

    def _ensure_model_exists(self) -> None:
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer_mod = importlib.import_module("colab.04_tabular_ml_trainer")
            trainer_cfg = trainer_mod.TabularTrainerConfig(
                output_onnx_path=str(self.model_path),
                dry_run=True,
            )
            trainer = trainer_mod.TabularMLTrainer(trainer_cfg)
            trainer.train_and_export(dry_run=True)

    def score_features(self, feature_vector: FeatureVector) -> AnomalyScore:
        arr = np.array([feature_vector.to_array()], dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self.input_name: arr})
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        if len(outputs) > 1 and isinstance(outputs[1], list) and outputs[1]:
            prob_dict = outputs[1][0]
            anomaly_score = float(prob_dict.get(1, prob_dict.get(1.0, 0.0)))
        else:
            raw_val = float(outputs[0][0])
            anomaly_score = float(np.clip(raw_val, 0.0, 1.0))

        is_anomaly = anomaly_score >= self.calibrated_threshold

        contributions = {
            k: float(v) for k, v in feature_vector.to_feature_dict().items()
        }

        return AnomalyScore(
            tenant_id=feature_vector.tenant_id,
            timestamp=feature_vector.timestamp,
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anomaly,
            threshold_used=self.calibrated_threshold,
            inference_latency_ms=latency_ms,
            contributing_features=contributions,
        )

    def score_batch(self, feature_vectors: List[FeatureVector]) -> List[AnomalyScore]:
        if not feature_vectors:
            return []

        matrix = np.array([fv.to_array() for fv in feature_vectors], dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self.input_name: matrix})
        total_latency_ms = (time.perf_counter() - t0) * 1000
        per_item_latency = round(total_latency_ms / len(feature_vectors), 3)

        scores: List[AnomalyScore] = []
        has_probabilities = len(outputs) > 1 and isinstance(outputs[1], list)

        for idx, fv in enumerate(feature_vectors):
            if has_probabilities:
                prob_dict = outputs[1][idx]
                score_val = float(prob_dict.get(1, prob_dict.get(1.0, 0.0)))
            else:
                score_val = float(np.clip(outputs[0][idx], 0.0, 1.0))

            is_anomaly = score_val >= self.calibrated_threshold
            scores.append(
                AnomalyScore(
                    tenant_id=fv.tenant_id,
                    timestamp=fv.timestamp,
                    anomaly_score=round(score_val, 4),
                    is_anomaly=is_anomaly,
                    threshold_used=self.calibrated_threshold,
                    inference_latency_ms=per_item_latency,
                    contributing_features=fv.to_feature_dict(),
                )
            )

        return scores

    def calibrate_threshold(
        self,
        validation_features: List[FeatureVector],
        target_fpr: float = 0.05,
    ) -> float:
        if not validation_features:
            return self.calibrated_threshold

        scores = self.score_batch(validation_features)
        raw_vals = [s.anomaly_score for s in scores]
        quantile = float(np.percentile(raw_vals, (1.0 - target_fpr) * 100))
        self.calibrated_threshold = round(quantile, 4)
        return self.calibrated_threshold

    def check_latency_sla(
        self,
        feature_vector: FeatureVector,
        n_warmup: int = 5,
        n_iterations: int = 50,
        p99_sla_ms: float = 10.0,
    ) -> bool:
        for _ in range(n_warmup):
            self.score_features(feature_vector)

        latencies: List[float] = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            self.score_features(feature_vector)
            latencies.append((time.perf_counter() - t0) * 1000)

        p99 = float(np.percentile(latencies, 99))
        return p99 <= p99_sla_ms
