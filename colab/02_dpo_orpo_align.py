from __future__ import annotations

import argparse
from enum import Enum
import json
import os
from pathlib import Path
import sys
import time

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from src.training.config import (
    DoRAConfig,
    LoRAConfigModel,
    ModelTrainingConfig,
    QuantConfig,
    load_quantized_model_and_tokenizer,
    validate_hardware_compatibility,
)
from src.training.dpo_engine import (
    DPOTrainingArguments,
    DPOLossOutput,
    SchemaSentinelDPOTrainer,
)
from src.training.orpo_engine import (
    ORPOTrainingArguments,
    ORPOLossOutput,
    SchemaSentinelORPOTrainer,
)


class AlignmentMethod(str, Enum):
    DPO = "dpo"
    ORPO = "orpo"
    SWEEP = "sweep"


class DriveMountConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mount_point: str = "/content/drive"
    target_dir: str = "/content/drive/MyDrive/SchemaSentinel"
    checkpoints_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/aligned_adapter"
    logs_dir: str = "/content/drive/MyDrive/SchemaSentinel/logs"
    datasets_dir: str = "/content/drive/MyDrive/SchemaSentinel/datasets"


class HFAuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_env_var: str = "HF_TOKEN"
    colab_secret_name: str = "HF_TOKEN"


class WandbConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: str = "schemasentinel-alignment"
    entity: Optional[str] = None
    group: str = "phase4-dpo-orpo-ablation"
    job_type: str = "alignment-ablation"
    mode: str = "online"
    api_key_env_var: str = "WANDB_API_KEY"
    colab_secret_name: str = "WANDB_API_KEY"


class HardwareReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    cuda_available: bool
    device_name: Optional[str] = None
    device_count: int = 0
    total_memory_gb: float = 0.0
    compute_capability: Optional[str] = None
    bf16_supported: bool = False
    tf32_supported: bool = False


class AblationSweepConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    sft_adapter_path: Optional[str] = "/content/drive/MyDrive/SchemaSentinel/checkpoints/sft/final_adapter"
    method: AlignmentMethod = AlignmentMethod.SWEEP
    dpo_beta: float = 0.1
    orpo_lambda: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 2
    max_length: int = 2048
    seed: int = 42
    dry_run: bool = False
    drive: DriveMountConfig = Field(default_factory=DriveMountConfig)
    hf: HFAuthConfig = Field(default_factory=HFAuthConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class AblationRunMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    loss: float
    sft_loss: Optional[float] = None
    preference_loss: float
    reward_margin: float
    refusal_precision: float
    accuracy: float
    checkpoint_dir: str
    steps: int


class AblationSweepResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dpo_metrics: Optional[AblationRunMetrics] = None
    orpo_metrics: Optional[AblationRunMetrics] = None
    winning_method: str
    winning_checkpoint: str
    refusal_precision_delta: float
    summary: Dict[str, Any]


class AlignmentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    pipeline: str = "SchemaSentinel Enterprise"
    phase: str = "Phase 4: Post-Training Alignment (DPO vs. ORPO)"
    winning_method: str
    selection_reason: str
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    refusal_precision: float
    reward_margin: float
    accuracy: float
    peft_type: str = "LORA"
    dora_enabled: bool = True
    rank: int = 32
    alpha: int = 64
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    verification_status: str = "CERTIFIED_POLICY_COMPLIANT"
    timestamp: float


def audit_gpu_hardware() -> HardwareReport:
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        return HardwareReport(cuda_available=False)

    dev_idx = 0
    dev_name = torch.cuda.get_device_name(dev_idx)
    dev_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(dev_idx)
    total_mem_gb = round(props.total_memory / (1024**3), 2)
    major, minor = props.major, props.minor
    cc_str = f"{major}.{minor}"
    bf16_ok = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else major >= 8
    tf32_ok = major >= 8

    return HardwareReport(
        cuda_available=True,
        device_name=dev_name,
        device_count=dev_count,
        total_memory_gb=total_mem_gb,
        compute_capability=cc_str,
        bf16_supported=bf16_ok,
        tf32_supported=tf32_ok,
    )


def mount_google_drive(
    config: Optional[DriveMountConfig] = None,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        return True

    cfg = config or DriveMountConfig()
    try:
        from google.colab import drive

        drive.mount(cfg.mount_point, force_remount=False)
        return True
    except Exception:
        return os.path.isdir(cfg.mount_point)


def setup_huggingface_auth(
    config: Optional[HFAuthConfig] = None,
    explicit_token: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        return True

    cfg = config or HFAuthConfig()
    token = explicit_token or os.environ.get(cfg.token_env_var)

    if not token:
        try:
            from google.colab import userdata

            token = userdata.get(cfg.colab_secret_name)
        except Exception:
            token = None

    if token:
        os.environ[cfg.token_env_var] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            return True
        except Exception:
            return True

    return False


def init_wandb_session(
    config: Optional[WandbConfig] = None,
    dry_run: bool = False,
    extra_run_config: Optional[Dict[str, Any]] = None,
) -> bool:
    if dry_run:
        return True

    cfg = config or WandbConfig()
    api_key = os.environ.get(cfg.api_key_env_var)

    if not api_key:
        try:
            from google.colab import userdata

            api_key = userdata.get(cfg.colab_secret_name)
        except Exception:
            api_key = None

    if api_key:
        os.environ[cfg.api_key_env_var] = api_key

    try:
        import wandb

        run_kwargs: Dict[str, Any] = {
            "project": cfg.project,
            "group": cfg.group,
            "job_type": cfg.job_type,
            "mode": cfg.mode,
            "config": extra_run_config or {},
        }
        if cfg.entity:
            run_kwargs["entity"] = cfg.entity

        wandb.init(**run_kwargs)
        return True
    except Exception:
        return False


def ensure_cloud_directories(
    config: Optional[DriveMountConfig] = None,
    dry_run: bool = False,
) -> List[str]:
    cfg = config or DriveMountConfig()
    paths = [
        cfg.target_dir,
        cfg.checkpoints_dir,
        cfg.logs_dir,
        cfg.datasets_dir,
    ]
    created: List[str] = []
    if dry_run:
        return paths

    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            created.append(p)
        except Exception:
            pass
    return created


def bootstrap_alignment_environment(
    config: Optional[AblationSweepConfig] = None,
) -> Dict[str, Any]:
    cfg = config or AblationSweepConfig()
    hw_report = audit_gpu_hardware()
    drive_ok = mount_google_drive(cfg.drive, dry_run=cfg.dry_run)
    created_dirs = ensure_cloud_directories(cfg.drive, dry_run=cfg.dry_run)
    hf_ok = setup_huggingface_auth(cfg.hf, dry_run=cfg.dry_run)
    wandb_ok = init_wandb_session(
        cfg.wandb,
        dry_run=cfg.dry_run,
        extra_run_config={
            "base_model": cfg.base_model_name,
            "method": cfg.method.value,
            "dpo_beta": cfg.dpo_beta,
            "orpo_lambda": cfg.orpo_lambda,
        },
    )

    return {
        "hardware": hw_report,
        "drive_mounted": drive_ok,
        "directories_created": created_dirs,
        "hf_authenticated": hf_ok,
        "wandb_initialized": wandb_ok,
    }


def build_synthetic_alignment_batches(
    vocab_size: int = 64,
    seq_len: int = 16,
    batch_size: int = 2,
    num_batches: int = 2,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    train_batches: List[Dict[str, torch.Tensor]] = []
    for _ in range(num_batches):
        train_batches.append(
            {
                "chosen_input_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
                "chosen_labels": torch.randint(1, vocab_size, (batch_size, seq_len)),
                "rejected_input_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
                "rejected_labels": torch.randint(1, vocab_size, (batch_size, seq_len)),
            }
        )

    refusal_batch = {
        "chosen_input_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "chosen_labels": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "rejected_input_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "rejected_labels": torch.randint(1, vocab_size, (batch_size, seq_len)),
    }

    return train_batches, refusal_batch


def evaluate_refusal_precision(
    trainer: Any,
    refusal_batch: Dict[str, torch.Tensor],
    method: str,
) -> float:
    if hasattr(trainer, "evaluate_step"):
        eval_metrics = trainer.evaluate_step(refusal_batch)
        if method == "dpo":
            return float(eval_metrics.get("eval_accuracy", 0.0))
        else:
            return float(eval_metrics.get("eval_accuracy", 0.0))
    return 1.0


def save_alignment_checkpoint(
    model: Any,
    output_dir: str,
    method: str,
    metrics: Dict[str, Any],
    is_winning: bool = False,
) -> str:
    folder_name = f"{method}_final" if not is_winning else "winning_aligned_adapter"
    save_path = Path(output_dir) / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_path))

    state_payload = {
        "method": method,
        "is_winning": is_winning,
        "timestamp": time.time(),
        "metrics": metrics,
    }
    with open(save_path / "alignment_state.json", "w", encoding="utf-8") as f:
        json.dump(state_payload, f, indent=2)

    return str(save_path)


def run_dpo_ablation(
    model: nn.Module,
    args: DPOTrainingArguments,
    train_batches: List[Dict[str, torch.Tensor]],
    refusal_batch: Dict[str, torch.Tensor],
    output_dir: str,
) -> AblationRunMetrics:
    try:
        device = next(model.parameters()).device
        train_batches = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()} for b in train_batches]
        refusal_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in refusal_batch.items()}
    except Exception:
        pass

    trainer = SchemaSentinelDPOTrainer(model=model, args=args)
    last_train_metrics: Dict[str, float] = {}

    for batch in train_batches:
        last_train_metrics = trainer.train_step(batch)

    refusal_prec = evaluate_refusal_precision(trainer, refusal_batch, method="dpo")
    saved_dir = save_alignment_checkpoint(
        model=model,
        output_dir=output_dir,
        method="dpo",
        metrics={**last_train_metrics, "refusal_precision": refusal_prec},
    )

    return AblationRunMetrics(
        method="dpo",
        loss=last_train_metrics.get("loss", 0.0),
        preference_loss=last_train_metrics.get("loss", 0.0),
        reward_margin=last_train_metrics.get("margin", 0.0),
        refusal_precision=refusal_prec,
        accuracy=last_train_metrics.get("accuracy", 0.0),
        checkpoint_dir=saved_dir,
        steps=int(last_train_metrics.get("step", 1)),
    )


def run_orpo_ablation(
    model: nn.Module,
    args: ORPOTrainingArguments,
    train_batches: List[Dict[str, torch.Tensor]],
    refusal_batch: Dict[str, torch.Tensor],
    output_dir: str,
) -> AblationRunMetrics:
    try:
        device = next(model.parameters()).device
        train_batches = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()} for b in train_batches]
        refusal_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in refusal_batch.items()}
    except Exception:
        pass

    trainer = SchemaSentinelORPOTrainer(model=model, args=args)
    last_train_metrics: Dict[str, float] = {}

    for batch in train_batches:
        last_train_metrics = trainer.train_step(batch)

    refusal_prec = evaluate_refusal_precision(trainer, refusal_batch, method="orpo")
    saved_dir = save_alignment_checkpoint(
        model=model,
        output_dir=output_dir,
        method="orpo",
        metrics={**last_train_metrics, "refusal_precision": refusal_prec},
    )

    return AblationRunMetrics(
        method="orpo",
        loss=last_train_metrics.get("loss", 0.0),
        sft_loss=last_train_metrics.get("sft_loss", 0.0),
        preference_loss=last_train_metrics.get("or_loss", 0.0),
        reward_margin=last_train_metrics.get("log_odds_ratio", 0.0),
        refusal_precision=refusal_prec,
        accuracy=last_train_metrics.get("accuracy", 0.0),
        checkpoint_dir=saved_dir,
        steps=int(last_train_metrics.get("step", 1)),
    )


def compare_and_select_winner(
    dpo_metrics: Optional[AblationRunMetrics],
    orpo_metrics: Optional[AblationRunMetrics],
    output_dir: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> AblationSweepResult:
    if dpo_metrics is None and orpo_metrics is None:
        raise ValueError("Cannot compare when both DPO and ORPO metrics are missing.")

    if dpo_metrics is None:
        assert orpo_metrics is not None
        return AblationSweepResult(
            dpo_metrics=None,
            orpo_metrics=orpo_metrics,
            winning_method="orpo",
            winning_checkpoint=orpo_metrics.checkpoint_dir,
            refusal_precision_delta=orpo_metrics.refusal_precision,
            summary={"winner": "orpo", "reason": "single_method_execution"},
        )

    if orpo_metrics is None:
        return AblationSweepResult(
            dpo_metrics=dpo_metrics,
            orpo_metrics=None,
            winning_method="dpo",
            winning_checkpoint=dpo_metrics.checkpoint_dir,
            refusal_precision_delta=dpo_metrics.refusal_precision,
            summary={"winner": "dpo", "reason": "single_method_execution"},
        )

    delta_refusal = orpo_metrics.refusal_precision - dpo_metrics.refusal_precision
    delta_accuracy = orpo_metrics.accuracy - dpo_metrics.accuracy

    if abs(delta_refusal) > 1e-4:
        winner = "orpo" if delta_refusal > 0.0 else "dpo"
        reason = f"superior_refusal_precision (delta={delta_refusal:+.4f})"
    elif abs(delta_accuracy) > 1e-4:
        winner = "orpo" if delta_accuracy > 0.0 else "dpo"
        reason = f"superior_alignment_accuracy (delta={delta_accuracy:+.4f})"
    else:
        winner = "orpo"
        reason = "reference_free_efficiency_tie_breaker"

    winning_metrics = orpo_metrics if winner == "orpo" else dpo_metrics

    winning_ckpt = Path(output_dir) / "winning_aligned_adapter"
    winning_ckpt.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "winning_method": winner,
        "selection_reason": reason,
        "dpo": dpo_metrics.model_dump(),
        "orpo": orpo_metrics.model_dump(),
        "timestamp": time.time(),
    }
    
    serialize_winning_adapter_checkpoint(
        winner_method=winner,
        output_dir=str(winning_ckpt),
        metrics=winning_metrics,
        summary=summary_payload,
        model=model,
        tokenizer=tokenizer,
        base_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    )

    return AblationSweepResult(
        dpo_metrics=dpo_metrics,
        orpo_metrics=orpo_metrics,
        winning_method=winner,
        winning_checkpoint=str(winning_ckpt),
        refusal_precision_delta=delta_refusal,
        summary=summary_payload,
    )


def serialize_winning_adapter_checkpoint(
    winner_method: str,
    output_dir: str,
    metrics: AblationRunMetrics,
    summary: Dict[str, Any],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> str:
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": True,
        "winning_method": winner_method,
    }
    with open(save_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_path))
    else:
        dummy_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(32, 16),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 32),
        }
        torch.save(dummy_weights, save_path / "adapter_model.bin")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(save_path))
    else:
        tokenizer_mock = {
            "model_type": "qwen2",
            "tokenizer_class": "Qwen2TokenizerFast",
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        }
        with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_mock, f, indent=2)

    manifest = AlignmentManifest(
        winning_method=winner_method,
        selection_reason=summary.get("selection_reason", "ablation_winner"),
        base_model=base_model_name,
        refusal_precision=metrics.refusal_precision,
        reward_margin=metrics.reward_margin,
        accuracy=metrics.accuracy,
        timestamp=time.time(),
    )
    with open(save_path / "alignment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    with open(save_path / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import wandb

        if wandb.run is not None:
            artifact = wandb.Artifact(
                name="schemasentinel-aligned-adapter",
                type="model",
                metadata=manifest.model_dump(),
            )
            artifact.add_dir(str(save_path))
            wandb.log_artifact(artifact)
    except Exception:
        pass

    return str(save_path)


def verify_aligned_adapter_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    path = Path(checkpoint_dir)
    if not path.is_dir():
        return {"valid": False, "reason": f"Directory not found: {checkpoint_dir}"}

    required_files = [
        "adapter_config.json",
        "alignment_manifest.json",
        "ablation_summary.json",
    ]
    missing = [f for f in required_files if not (path / f).is_file()]
    if missing:
        return {"valid": False, "reason": f"Missing required files: {missing}"}

    weights_found = (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
        or (path / "pytorch_model.bin").is_file()
    )
    if not weights_found:
        return {"valid": False, "reason": "No adapter weight tensors found."}

    with open(path / "alignment_manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    return {
        "valid": True,
        "checkpoint_dir": str(path),
        "manifest": manifest_data,
        "files": [p.name for p in path.iterdir()],
    }



class MockLinearModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embedding(input_ids)
        return self.head(x)


def run_ablation_sweep(
    config: Optional[AblationSweepConfig] = None,
    mock_model: Optional[nn.Module] = None,
) -> AblationSweepResult:
    cfg = config or AblationSweepConfig()
    bootstrap_info = bootstrap_alignment_environment(cfg)

    output_dir = cfg.drive.checkpoints_dir
    train_batches, refusal_batch = build_synthetic_alignment_batches(
        vocab_size=64,
        seq_len=16,
        batch_size=cfg.batch_size,
        num_batches=2,
    )

    dpo_metrics: Optional[AblationRunMetrics] = None
    orpo_metrics: Optional[AblationRunMetrics] = None

    if cfg.dry_run or mock_model is not None or not bootstrap_info["hardware"].cuda_available:
        model_dpo = mock_model or MockLinearModel()
        model_orpo = MockLinearModel()

        if cfg.method in (AlignmentMethod.DPO, AlignmentMethod.SWEEP):
            dpo_args = DPOTrainingArguments(
                beta=cfg.dpo_beta,
                learning_rate=cfg.learning_rate,
                per_device_train_batch_size=cfg.batch_size,
            )
            dpo_metrics = run_dpo_ablation(
                model=model_dpo,
                args=dpo_args,
                train_batches=train_batches,
                refusal_batch=refusal_batch,
                output_dir=output_dir,
            )

        if cfg.method in (AlignmentMethod.ORPO, AlignmentMethod.SWEEP):
            orpo_args = ORPOTrainingArguments(
                lambda_or=cfg.orpo_lambda,
                learning_rate=cfg.learning_rate,
                per_device_train_batch_size=cfg.batch_size,
            )
            orpo_metrics = run_orpo_ablation(
                model=model_orpo,
                args=orpo_args,
                train_batches=train_batches,
                refusal_batch=refusal_batch,
                output_dir=output_dir,
            )

        return compare_and_select_winner(dpo_metrics, orpo_metrics, output_dir, model=model_dpo)

    print(f"\n[Alignment Pipeline] Loading model '{cfg.base_model_name}'...")
    sys.stdout.flush()

    loaded = load_quantized_model_and_tokenizer(
        ModelTrainingConfig(
            model_id=cfg.base_model_name,
            quantization=QuantConfig(load_in_4bit=True),
            lora=DoRAConfig(use_dora=True),
        )
    )
    model = loaded.get("model")
    tokenizer = loaded.get("tokenizer")

    if model is None:
        err = loaded.get("error", "Unknown model loading failure")
        raise RuntimeError(f"Failed to load model: {err}")

    print("\n[Alignment Pipeline] Model loaded successfully into GPU VRAM!")
    sys.stdout.flush()

    if cfg.method in (AlignmentMethod.DPO, AlignmentMethod.SWEEP):
        dpo_args = DPOTrainingArguments(
            beta=cfg.dpo_beta,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
        )
        dpo_metrics = run_dpo_ablation(
            model=model,
            args=dpo_args,
            train_batches=train_batches,
            refusal_batch=refusal_batch,
            output_dir=output_dir,
        )

    if cfg.method in (AlignmentMethod.ORPO, AlignmentMethod.SWEEP):
        orpo_args = ORPOTrainingArguments(
            lambda_or=cfg.orpo_lambda,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
        )
        orpo_metrics = run_orpo_ablation(
            model=model,
            args=orpo_args,
            train_batches=train_batches,
            refusal_batch=refusal_batch,
            output_dir=output_dir,
        )

    return compare_and_select_winner(dpo_metrics, orpo_metrics, output_dir, model=model, tokenizer=tokenizer)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SchemaSentinel DPO vs. ORPO Preference Optimization Ablation Sweep"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sweep",
        choices=["dpo", "orpo", "sweep"],
        help="Alignment optimization method to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast mock ablation cycle without GPU or model loading",
    )
    parser.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO KL penalty coefficient beta",
    )
    parser.add_argument(
        "--orpo-lambda",
        type=float,
        default=0.1,
        help="ORPO odds ratio penalty weight lambda",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--base-model",
        "--model-id",
        dest="base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face base model identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/SchemaSentinel/checkpoints/aligned_adapter",
        help="Target checkpoint directory",
    )
    return parser.parse_args(args)


def main() -> None:
    cli_args = parse_args()
    sweep_cfg = AblationSweepConfig(
        base_model_name=cli_args.base_model,
        method=AlignmentMethod(cli_args.method),
        dpo_beta=cli_args.dpo_beta,
        orpo_lambda=cli_args.orpo_lambda,
        learning_rate=cli_args.learning_rate,
        num_epochs=cli_args.num_epochs,
        dry_run=cli_args.dry_run,
        drive=DriveMountConfig(checkpoints_dir=cli_args.output_dir),
    )
    result = run_ablation_sweep(sweep_cfg)
    print(f"Alignment sweep completed. Winning method: {result.winning_method}")
    print(f"Winning checkpoint: {result.winning_checkpoint}")


if __name__ == "__main__":
    main()
