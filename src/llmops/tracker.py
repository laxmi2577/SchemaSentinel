from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field


class TrackerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: str = "SchemaSentinel-Enterprise"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    job_type: str = "train"
    tags: List[str] = Field(
        default_factory=lambda: ["production", "text-to-sql", "qlora"]
    )
    mode: str = "disabled"
    log_code: bool = True
    dataset_name: str = "spider-bird-enterprise"
    dataset_version: str = "v1.0"
    artifact_dir: str = "artifacts"
    dry_run: bool = False


class ModelLineageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    run_name: str
    timestamp: str
    git_commit_hash: str
    base_model: str
    adapter_checkpoint_path: str
    dataset_version: str
    dataset_hash: str
    training_stage: str
    metrics_summary: Dict[str, float] = Field(default_factory=dict)
    artifact_uris: List[str] = Field(default_factory=list)


def get_git_commit_hash(
    default_hash: str = "0000000000000000000000000000000000000000",
) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            if len(commit) == 40:
                return commit
    except Exception:
        pass
    fallback_content = f"{sys.executable}:{os.name}:{default_hash}"
    return hashlib.sha256(fallback_content.encode("utf-8")).hexdigest()[:40]


def compute_dataset_hash(dataset_input: Union[str, bytes]) -> str:
    hasher = hashlib.sha256()
    if isinstance(dataset_input, bytes):
        hasher.update(dataset_input)
    else:
        path = Path(dataset_input)
        if path.is_file():
            hasher.update(path.read_bytes())
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    hasher.update(child.name.encode("utf-8"))
                    hasher.update(child.read_bytes())
        else:
            hasher.update(dataset_input.encode("utf-8"))
    return hasher.hexdigest()


def log_loss_decomposition(
    tracker: WBTelemetryTracker,
    step: int,
    ce_loss: float,
    unlikelihood_loss: float,
    total_loss: float,
    margin: Optional[float] = None,
    grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    payload: Dict[str, float] = {
        "loss/cross_entropy": float(ce_loss),
        "loss/unlikelihood": float(unlikelihood_loss),
        "loss/total": float(total_loss),
    }
    if margin is not None:
        payload["margin/implicit_reward"] = float(margin)
    if grad_norm is not None:
        payload["optim/gradient_norm"] = float(grad_norm)

    tracker.log_metrics(payload, step=step)
    return payload


class WBTelemetryTracker:
    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.client = client
        self.is_active = False
        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = self.config.run_name or f"run-{self.run_id}"
        self.metrics_history: List[Dict[str, Any]] = []
        self.registered_artifacts: List[Dict[str, Any]] = []

        if not self.config.dry_run and self.config.mode != "disabled":
            self.init_run()
        else:
            self.is_active = True

    def init_run(self) -> bool:
        if self.config.dry_run or self.config.mode == "disabled":
            self.is_active = True
            return True

        try:
            import wandb

            self.client = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.run_name,
                job_type=self.config.job_type,
                tags=self.config.tags,
                mode=self.config.mode,
            )
            self.run_id = str(getattr(self.client, "id", self.run_id))
            self.is_active = True
            return True
        except Exception:
            self.is_active = True
            return False

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "step": step,
            "timestamp": time.time(),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        self.metrics_history.append(record)

        if self.client is not None and hasattr(self.client, "log"):
            try:
                payload = dict(metrics)
                if step is not None:
                    self.client.log(payload, step=step)
                else:
                    self.client.log(payload)
            except Exception:
                pass

        return record

    def register_model_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = metadata or {}
        entry: Dict[str, Any] = {
            "name": artifact_name,
            "type": artifact_type,
            "file_path": file_path,
            "metadata": meta,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        self.registered_artifacts.append(entry)

        if self.client is not None and hasattr(self.client, "log_artifact"):
            try:
                import wandb

                art = wandb.Artifact(
                    name=artifact_name,
                    type=artifact_type,
                    metadata=meta,
                )
                if os.path.isfile(file_path):
                    art.add_file(file_path)
                elif os.path.isdir(file_path):
                    art.add_dir(file_path)
                self.client.log_artifact(art)
            except Exception:
                pass

        return entry

    def create_lineage_record(
        self,
        base_model: str,
        adapter_checkpoint_path: str,
        training_stage: str,
        dataset_input: Optional[Union[str, bytes]] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> ModelLineageRecord:
        git_hash = get_git_commit_hash()
        d_hash = (
            compute_dataset_hash(dataset_input)
            if dataset_input is not None
            else "dataset-hash-unavailable"
        )

        summary_metrics: Dict[str, float] = {}
        if custom_metrics:
            summary_metrics.update(custom_metrics)
        elif self.metrics_history:
            last_entry = self.metrics_history[-1].get("metrics", {})
            summary_metrics.update(last_entry)

        uris = [a["file_path"] for a in self.registered_artifacts]

        return ModelLineageRecord(
            run_id=self.run_id,
            run_name=self.run_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            git_commit_hash=git_hash,
            base_model=base_model,
            adapter_checkpoint_path=adapter_checkpoint_path,
            dataset_version=self.config.dataset_version,
            dataset_hash=d_hash,
            training_stage=training_stage,
            metrics_summary=summary_metrics,
            artifact_uris=uris,
        )

    def export_lineage_manifest(
        self,
        record: ModelLineageRecord,
        output_path: str,
    ) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
        return str(out)

    def finish_run(self) -> bool:
        if self.client is not None and hasattr(self.client, "finish"):
            try:
                self.client.finish()
            except Exception:
                pass
        self.is_active = False
        return True
