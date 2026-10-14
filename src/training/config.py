from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


class QuantConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class LoRAConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False


class DoRAConfig(LoRAConfigModel):
    model_config = ConfigDict(frozen=True)

    use_dora: bool = True


class TrainingHyperparameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"


class ModelTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    device_map: str = "auto"
    quantization: QuantConfig = Field(default_factory=QuantConfig)
    lora: DoRAConfig = Field(default_factory=DoRAConfig)
    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)


def resolve_compute_dtype(dtype_name: str) -> Any:
    try:
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_name, torch.bfloat16)
    except ImportError:
        return dtype_name


def build_bnb_config(quant_config: Optional[QuantConfig] = None) -> Any:
    cfg = quant_config or QuantConfig()
    compute_dtype = resolve_compute_dtype(cfg.bnb_4bit_compute_dtype)

    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except ImportError:
        return {
            "load_in_4bit": cfg.load_in_4bit,
            "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": cfg.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": str(compute_dtype),
        }


def build_peft_config(lora_config: Optional[Union[LoRAConfigModel, DoRAConfig]] = None) -> Any:
    cfg = lora_config or DoRAConfig()

    try:
        from peft import LoraConfig, TaskType

        task = TaskType.CAUSAL_LM if cfg.task_type == "CAUSAL_LM" else cfg.task_type
        return LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            bias=cfg.bias,
            task_type=task,
            use_dora=cfg.use_dora,
        )
    except ImportError:
        return {
            "r": cfg.r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": list(cfg.target_modules),
            "bias": cfg.bias,
            "task_type": cfg.task_type,
            "use_dora": cfg.use_dora,
        }


def validate_hardware_compatibility(
    config: ModelTrainingConfig,
    cuda_available: bool = False,
    total_vram_gb: float = 0.0,
) -> Dict[str, Any]:
    estimated_4bit_weights_gb = 5.5
    estimated_overhead_gb = 3.5
    min_required_vram_gb = estimated_4bit_weights_gb + estimated_overhead_gb

    if not cuda_available:
        return {
            "compatible": False,
            "reason": "CUDA GPU required for BitsAndBytes 4-bit NormalFloat training.",
            "min_required_vram_gb": min_required_vram_gb,
            "total_vram_gb": 0.0,
            "plane": "Local Control Plane (CPU only)",
        }

    if total_vram_gb < min_required_vram_gb:
        return {
            "compatible": False,
            "reason": f"Insufficient VRAM: detected {total_vram_gb} GB, required at least {min_required_vram_gb} GB.",
            "min_required_vram_gb": min_required_vram_gb,
            "total_vram_gb": total_vram_gb,
            "plane": "Insufficient Cloud GPU",
        }

    return {
        "compatible": True,
        "reason": f"Hardware approved: {total_vram_gb} GB VRAM exceeds minimum {min_required_vram_gb} GB requirement.",
        "min_required_vram_gb": min_required_vram_gb,
        "total_vram_gb": total_vram_gb,
        "plane": "Cloud Compute Plane (Colab T4/A100)",
    }


def load_quantized_model_and_tokenizer(
    config: Optional[ModelTrainingConfig] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cfg = config or ModelTrainingConfig()
    bnb_cfg = build_bnb_config(cfg.quantization)
    peft_cfg = build_peft_config(cfg.lora)

    if dry_run or os.environ.get("SCHEMASENTINEL_CONTROL_PLANE") == "1":
        return {
            "status": "dry_run_success",
            "model_id": cfg.model_id,
            "quantization_config": bnb_cfg,
            "peft_config": peft_cfg,
            "device_map": cfg.device_map,
            "model": None,
            "tokenizer": None,
        }

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=bnb_cfg,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation=cfg.attn_implementation,
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
        )
        model = get_peft_model(model, peft_cfg)

        return {
            "status": "loaded",
            "model_id": cfg.model_id,
            "quantization_config": bnb_cfg,
            "peft_config": peft_cfg,
            "device_map": cfg.device_map,
            "model": model,
            "tokenizer": tokenizer,
        }
    except Exception as exc:
        return {
            "status": "error",
            "model_id": cfg.model_id,
            "quantization_config": bnb_cfg,
            "peft_config": peft_cfg,
            "device_map": cfg.device_map,
            "error": str(exc),
            "model": None,
            "tokenizer": None,
        }
