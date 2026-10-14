import json
from pathlib import Path
import pytest
from src.llmops.tracker import (
    TrackerConfig,
    ModelLineageRecord,
    get_git_commit_hash,
    compute_dataset_hash,
    log_loss_decomposition,
    WBTelemetryTracker,
)


def test_tracker_config_defaults():
    config = TrackerConfig()
    assert config.project == "SchemaSentinel-Enterprise"
    assert config.job_type == "train"
    assert config.mode == "disabled"
    assert "production" in config.tags
    assert config.dataset_name == "spider-bird-enterprise"
    assert config.dataset_version == "v1.0"
    assert config.log_code is True


def test_get_git_commit_hash():
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) == 40
    int(commit, 16)


def test_compute_dataset_hash_bytes_and_file(tmp_path: Path):
    raw_data = b"schema_sentinel_sample_data"
    hash_bytes = compute_dataset_hash(raw_data)
    assert isinstance(hash_bytes, str)
    assert len(hash_bytes) == 64

    temp_file = tmp_path / "dataset.jsonl"
    temp_file.write_bytes(raw_data)
    hash_file = compute_dataset_hash(str(temp_file))
    assert hash_file == hash_bytes


def test_wb_telemetry_tracker_init_and_metrics():
    config = TrackerConfig(dry_run=True, run_name="test-run-1")
    tracker = WBTelemetryTracker(config=config)

    assert tracker.is_active is True
    assert tracker.run_name == "test-run-1"

    rec1 = tracker.log_metrics({"train/loss": 1.45, "train/lr": 0.0002}, step=1)
    rec2 = tracker.log_metrics({"train/loss": 1.12, "train/lr": 0.00018}, step=2)

    assert len(tracker.metrics_history) == 2
    assert rec1["metrics"]["train/loss"] == 1.45
    assert rec2["metrics"]["train/loss"] == 1.12


def test_log_loss_decomposition():
    tracker = WBTelemetryTracker(config=TrackerConfig(dry_run=True))
    payload = log_loss_decomposition(
        tracker=tracker,
        step=10,
        ce_loss=0.85,
        unlikelihood_loss=0.15,
        total_loss=1.00,
        margin=0.45,
        grad_norm=1.23,
    )

    assert payload["loss/cross_entropy"] == 0.85
    assert payload["loss/unlikelihood"] == 0.15
    assert payload["loss/total"] == 1.00
    assert payload["margin/implicit_reward"] == 0.45
    assert payload["optim/gradient_norm"] == 1.23
    assert len(tracker.metrics_history) == 1


def test_register_model_artifact(tmp_path: Path):
    mock_weights = tmp_path / "adapter_model.safetensors"
    mock_weights.write_bytes(b"MOCK_WEIGHTS")

    tracker = WBTelemetryTracker(config=TrackerConfig(dry_run=True))
    entry = tracker.register_model_artifact(
        artifact_name="aligned_adapter",
        artifact_type="model_weights",
        file_path=str(mock_weights),
        metadata={"stage": "dpo_winner", "rank": 32},
    )

    assert entry["name"] == "aligned_adapter"
    assert entry["type"] == "model_weights"
    assert entry["file_path"] == str(mock_weights)
    assert entry["metadata"]["rank"] == 32
    assert len(tracker.registered_artifacts) == 1


def test_create_lineage_record_and_export_manifest(tmp_path: Path):
    tracker = WBTelemetryTracker(config=TrackerConfig(dry_run=True, dataset_version="v2.1"))
    tracker.log_metrics({"eval/accuracy": 0.92, "eval/refusal_precision": 1.0}, step=100)

    record = tracker.create_lineage_record(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_checkpoint_path="checkpoints/aligned_adapter",
        training_stage="dpo_alignment",
        dataset_input=b"train_dataset_blob",
    )

    assert isinstance(record, ModelLineageRecord)
    assert record.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert record.dataset_version == "v2.1"
    assert record.training_stage == "dpo_alignment"
    assert len(record.git_commit_hash) == 40
    assert len(record.dataset_hash) == 64
    assert record.metrics_summary["eval/accuracy"] == 0.92

    manifest_path = tmp_path / "lineage_manifest.json"
    exported_file = tracker.export_lineage_manifest(record, str(manifest_path))

    assert Path(exported_file).exists()
    loaded_data = json.loads(Path(exported_file).read_text(encoding="utf-8"))
    assert loaded_data["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert loaded_data["dataset_version"] == "v2.1"


def test_tracker_with_mock_wandb_client():
    class MockArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.added_files = []

        def add_file(self, p):
            self.added_files.append(p)

    class MockWandbClient:
        def __init__(self):
            self.id = "mock-wandb-id-123"
            self.logged_payloads = []
            self.logged_artifacts = []
            self.is_finished = False

        def log(self, payload, step=None):
            self.logged_payloads.append((payload, step))

        def log_artifact(self, artifact):
            self.logged_artifacts.append(artifact)

        def finish(self):
            self.is_finished = True

    mock_client = MockWandbClient()
    tracker = WBTelemetryTracker(
        config=TrackerConfig(dry_run=False, mode="online"),
        client=mock_client,
    )

    tracker.log_metrics({"step_loss": 0.42}, step=5)
    assert len(mock_client.logged_payloads) == 1
    assert mock_client.logged_payloads[0][1] == 5

    tracker.finish_run()
    assert mock_client.is_finished is True
