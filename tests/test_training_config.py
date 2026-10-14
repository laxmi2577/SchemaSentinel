import pytest
from src.training.config import (
    QuantConfig,
    LoRAConfigModel,
    DoRAConfig,
    TrainingHyperparameters,
    ModelTrainingConfig,
    resolve_compute_dtype,
    build_bnb_config,
    build_peft_config,
    validate_hardware_compatibility,
    load_quantized_model_and_tokenizer,
)


def test_quant_config_defaults():
    cfg = QuantConfig()
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == "bfloat16"


def test_lora_and_dora_config_modules():
    lora = LoRAConfigModel()
    assert lora.r == 32
    assert lora.lora_alpha == 64
    assert lora.use_dora is False
    assert "q_proj" in lora.target_modules
    assert "k_proj" in lora.target_modules
    assert "v_proj" in lora.target_modules
    assert "o_proj" in lora.target_modules
    assert "gate_proj" in lora.target_modules
    assert "up_proj" in lora.target_modules
    assert "down_proj" in lora.target_modules

    dora = DoRAConfig()
    assert dora.r == 32
    assert dora.lora_alpha == 64
    assert dora.use_dora is True


def test_training_hyperparameters():
    hparams = TrainingHyperparameters()
    assert hparams.learning_rate == 2e-4
    assert hparams.optimizer == "paged_adamw_8bit"
    assert hparams.lr_scheduler_type == "cosine"
    assert hparams.gradient_accumulation_steps == 8
    assert hparams.max_seq_length == 2048


def test_model_training_config_structure():
    config = ModelTrainingConfig()
    assert config.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.torch_dtype == "bfloat16"
    assert config.quantization.load_in_4bit is True
    assert config.lora.use_dora is True
    assert config.hyperparameters.optimizer == "paged_adamw_8bit"


def test_resolve_compute_dtype():
    import torch

    assert resolve_compute_dtype("bfloat16") == torch.bfloat16
    assert resolve_compute_dtype("float16") == torch.float16
    assert resolve_compute_dtype("float32") == torch.float32


def test_build_bnb_config_structure():
    bnb_cfg = build_bnb_config()
    assert bnb_cfg is not None


def test_build_peft_config_structure():
    peft_cfg = build_peft_config()
    assert peft_cfg is not None


def test_validate_hardware_compatibility():
    config = ModelTrainingConfig()

    cpu_res = validate_hardware_compatibility(config, cuda_available=False, total_vram_gb=0.0)
    assert cpu_res["compatible"] is False
    assert "Control Plane" in cpu_res["plane"]

    low_vram_res = validate_hardware_compatibility(config, cuda_available=True, total_vram_gb=4.0)
    assert low_vram_res["compatible"] is False
    assert "Insufficient" in low_vram_res["reason"]

    t4_res = validate_hardware_compatibility(config, cuda_available=True, total_vram_gb=15.0)
    assert t4_res["compatible"] is True
    assert "Cloud Compute Plane" in t4_res["plane"]

    a100_res = validate_hardware_compatibility(config, cuda_available=True, total_vram_gb=40.0)
    assert a100_res["compatible"] is True


def test_load_quantized_model_dry_run():
    config = ModelTrainingConfig()
    res = load_quantized_model_and_tokenizer(config, dry_run=True)
    assert res["status"] == "dry_run_success"
    assert res["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert res["quantization_config"] is not None
    assert res["peft_config"] is not None
    assert res["model"] is None
