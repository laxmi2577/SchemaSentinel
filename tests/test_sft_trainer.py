import pytest
import torch
import torch.nn as nn
from src.training.sft_trainer import (
    AdapterType,
    SFTTrainingArguments,
    SchemaSentinelSFTTrainer,
    build_trainer,
    get_trainable_parameters_summary,
)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return logits


def test_sft_training_arguments_defaults():
    args = SFTTrainingArguments()
    assert args.adapter_type == AdapterType.DORA
    assert args.learning_rate == 2e-4
    assert args.optimizer == "paged_adamw_8bit"
    assert args.lambda_ul == 0.5
    assert args.use_unlikelihood_loss is True
    assert args.max_seq_length == 2048


def test_get_trainable_parameters_summary():
    model = TinyLanguageModel(vocab_size=10, hidden_dim=4)
    for p in model.embedding.parameters():
        p.requires_grad = False

    summary = get_trainable_parameters_summary(model)
    assert summary["all_params"] > 0
    assert summary["trainable_params"] < summary["all_params"]
    assert 0.0 < summary["trainable_percentage"] < 100.0


def test_sft_trainer_initialization():
    model = TinyLanguageModel()
    args = SFTTrainingArguments(learning_rate=1e-3)
    trainer = SchemaSentinelSFTTrainer(
        model=model,
        args=args,
        negative_token_ids=[2, 3],
    )
    assert trainer.global_step == 0
    assert trainer.optimizer is not None


def test_sft_trainer_train_step():
    model = TinyLanguageModel(vocab_size=16, hidden_dim=8)
    args = SFTTrainingArguments(learning_rate=1e-3)
    trainer = SchemaSentinelSFTTrainer(
        model=model,
        args=args,
        negative_token_ids=[4, 5],
    )

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1]]),
        "labels": torch.tensor([[-100, 2, 3, 4], [-100, 3, 4, 1]]),
    }

    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert "ce_loss" in metrics
    assert "ul_loss" in metrics
    assert metrics["step"] == 1.0
    assert trainer.global_step == 1


def test_sft_trainer_evaluate_step():
    model = TinyLanguageModel(vocab_size=16, hidden_dim=8)
    trainer = SchemaSentinelSFTTrainer(
        model=model,
        negative_token_ids=[4, 5],
    )

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, 2, 3, 4]]),
    }

    eval_metrics = trainer.evaluate_step(batch)
    assert "eval_loss" in eval_metrics
    assert "eval_ce_loss" in eval_metrics
    assert "eval_ul_loss" in eval_metrics


def test_sft_trainer_without_unlikelihood_loss():
    model = TinyLanguageModel(vocab_size=16, hidden_dim=8)
    args = SFTTrainingArguments(use_unlikelihood_loss=False)
    trainer = SchemaSentinelSFTTrainer(
        model=model,
        args=args,
        negative_token_ids=[4, 5],
    )

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[-100, 2, 3]]),
    }

    metrics = trainer.train_step(batch)
    assert metrics["ul_loss"] == 0.0
    assert abs(metrics["loss"] - metrics["ce_loss"]) < 1e-6


def test_build_trainer_helper():
    model = TinyLanguageModel()
    trainer = build_trainer(model=model)
    assert isinstance(trainer, SchemaSentinelSFTTrainer)
    summary = trainer.get_summary()
    assert summary["adapter_type"] == "dora"
