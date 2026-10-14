import importlib
import json
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn

align_module = importlib.import_module("colab.02_dpo_orpo_align")

AlignmentMethod = align_module.AlignmentMethod
DriveMountConfig = align_module.DriveMountConfig
HFAuthConfig = align_module.HFAuthConfig
WandbConfig = align_module.WandbConfig
HardwareReport = align_module.HardwareReport
AblationSweepConfig = align_module.AblationSweepConfig
AblationRunMetrics = align_module.AblationRunMetrics
AblationSweepResult = align_module.AblationSweepResult
audit_gpu_hardware = align_module.audit_gpu_hardware
mount_google_drive = align_module.mount_google_drive
setup_huggingface_auth = align_module.setup_huggingface_auth
init_wandb_session = align_module.init_wandb_session
ensure_cloud_directories = align_module.ensure_cloud_directories
bootstrap_alignment_environment = align_module.bootstrap_alignment_environment
build_synthetic_alignment_batches = align_module.build_synthetic_alignment_batches
evaluate_refusal_precision = align_module.evaluate_refusal_precision
save_alignment_checkpoint = align_module.save_alignment_checkpoint
run_dpo_ablation = align_module.run_dpo_ablation
run_orpo_ablation = align_module.run_orpo_ablation
compare_and_select_winner = align_module.compare_and_select_winner
run_ablation_sweep = align_module.run_ablation_sweep
parse_args = align_module.parse_args
MockLinearModel = align_module.MockLinearModel
AlignmentManifest = align_module.AlignmentManifest
serialize_winning_adapter_checkpoint = align_module.serialize_winning_adapter_checkpoint
verify_aligned_adapter_checkpoint = align_module.verify_aligned_adapter_checkpoint


def test_ablation_sweep_config_defaults():
    config = AblationSweepConfig()
    assert config.base_model_name == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.method == AlignmentMethod.SWEEP
    assert config.dpo_beta == 0.1
    assert config.orpo_lambda == 0.1
    assert config.learning_rate == 5e-5
    assert config.batch_size == 2
    assert config.dry_run is False


def test_audit_gpu_hardware():
    report = audit_gpu_hardware()
    assert isinstance(report, HardwareReport)
    assert isinstance(report.cuda_available, bool)


def test_mount_google_drive_dry_run():
    assert mount_google_drive(dry_run=True) is True


def test_setup_huggingface_auth(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert setup_huggingface_auth(dry_run=True) is True
    assert setup_huggingface_auth(explicit_token="mock_token", dry_run=False) is True


def test_ensure_cloud_directories(tmp_path):
    target = str(tmp_path / "SchemaSentinel")
    checkpoints = str(tmp_path / "SchemaSentinel" / "checkpoints" / "aligned_adapter")
    drive_cfg = DriveMountConfig(target_dir=target, checkpoints_dir=checkpoints)
    created = ensure_cloud_directories(drive_cfg, dry_run=False)
    assert len(created) > 0
    assert os.path.isdir(checkpoints)


def test_build_synthetic_alignment_batches():
    train_batches, refusal_batch = build_synthetic_alignment_batches(
        vocab_size=32,
        seq_len=8,
        batch_size=2,
        num_batches=2,
    )
    assert len(train_batches) == 2
    assert "chosen_input_ids" in train_batches[0]
    assert "rejected_input_ids" in train_batches[0]
    assert "chosen_input_ids" in refusal_batch
    assert refusal_batch["chosen_input_ids"].shape == (2, 8)


def test_dpo_and_orpo_ablation_runs(tmp_path):
    output_dir = str(tmp_path / "checkpoints")
    train_batches, refusal_batch = build_synthetic_alignment_batches(
        vocab_size=32,
        seq_len=8,
        batch_size=2,
        num_batches=1,
    )

    from src.training.dpo_engine import DPOTrainingArguments
    from src.training.orpo_engine import ORPOTrainingArguments

    model_dpo = MockLinearModel(vocab_size=32, hidden_dim=8)
    dpo_args = DPOTrainingArguments(beta=0.1, learning_rate=1e-3)
    dpo_metrics = run_dpo_ablation(
        model=model_dpo,
        args=dpo_args,
        train_batches=train_batches,
        refusal_batch=refusal_batch,
        output_dir=output_dir,
    )
    assert isinstance(dpo_metrics, AblationRunMetrics)
    assert dpo_metrics.method == "dpo"
    assert os.path.isdir(dpo_metrics.checkpoint_dir)

    model_orpo = MockLinearModel(vocab_size=32, hidden_dim=8)
    orpo_args = ORPOTrainingArguments(lambda_or=0.1, learning_rate=1e-3)
    orpo_metrics = run_orpo_ablation(
        model=model_orpo,
        args=orpo_args,
        train_batches=train_batches,
        refusal_batch=refusal_batch,
        output_dir=output_dir,
    )
    assert isinstance(orpo_metrics, AblationRunMetrics)
    assert orpo_metrics.method == "orpo"
    assert os.path.isdir(orpo_metrics.checkpoint_dir)

    sweep_res = compare_and_select_winner(dpo_metrics, orpo_metrics, output_dir)
    assert isinstance(sweep_res, AblationSweepResult)
    assert sweep_res.winning_method in ("dpo", "orpo")
    assert os.path.isdir(sweep_res.winning_checkpoint)
    summary_file = Path(sweep_res.winning_checkpoint) / "ablation_summary.json"
    assert summary_file.is_file()


def test_run_ablation_sweep_dry_run(tmp_path):
    checkpoints_dir = str(tmp_path / "sweep_ckpts")
    cfg = AblationSweepConfig(
        dry_run=True,
        drive=DriveMountConfig(checkpoints_dir=checkpoints_dir),
    )
    result = run_ablation_sweep(config=cfg)
    assert isinstance(result, AblationSweepResult)
    assert result.winning_method in ("dpo", "orpo")
    assert result.dpo_metrics is not None
    assert result.orpo_metrics is not None


def test_cli_argument_parsing():
    parsed = parse_args(
        [
            "--method",
            "orpo",
            "--dry-run",
            "--dpo-beta",
            "0.15",
            "--orpo-lambda",
            "0.2",
            "--learning-rate",
            "3e-5",
            "--num-epochs",
            "3",
        ]
    )
    assert parsed.method == "orpo"
    assert parsed.dry_run is True
    assert parsed.dpo_beta == 0.15
    assert parsed.orpo_lambda == 0.2
    assert parsed.learning_rate == 3e-5
    assert parsed.num_epochs == 3


def test_serialize_and_verify_winning_adapter_checkpoint(tmp_path):
    output_dir = str(tmp_path / "aligned_adapter")
    metrics = AblationRunMetrics(
        method="orpo",
        loss=0.45,
        sft_loss=0.35,
        preference_loss=0.10,
        reward_margin=0.88,
        refusal_precision=1.0,
        accuracy=0.95,
        checkpoint_dir=output_dir,
        steps=50,
    )
    summary = {
        "winning_method": "orpo",
        "selection_reason": "superior_refusal_precision",
    }
    saved_path = serialize_winning_adapter_checkpoint(
        winner_method="orpo",
        output_dir=output_dir,
        metrics=metrics,
        summary=summary,
    )
    assert os.path.isdir(saved_path)
    assert (Path(saved_path) / "adapter_config.json").is_file()
    assert (Path(saved_path) / "alignment_manifest.json").is_file()
    assert (Path(saved_path) / "ablation_summary.json").is_file()

    report = verify_aligned_adapter_checkpoint(saved_path)
    assert report["valid"] is True
    assert report["manifest"]["winning_method"] == "orpo"
    assert report["manifest"]["refusal_precision"] == 1.0


def test_verify_aligned_adapter_checkpoint_missing_directory():
    report = verify_aligned_adapter_checkpoint("non_existent_dir_999")
    assert report["valid"] is False
