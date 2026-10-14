import pytest
import torch
import torch.nn as nn
from src.training.orpo_engine import (
    ORPOTrainingArguments,
    ORPOLossOutput,
    compute_sequence_logps_and_lengths,
    compute_log_odds,
    odds_ratio_loss,
    SchemaSentinelORPOTrainer,
)


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embedding(input_ids)
        return self.head(x)


def test_orpo_training_arguments_defaults():
    args = ORPOTrainingArguments()
    assert args.lambda_or == 0.1
    assert args.learning_rate == 5e-5
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 4
    assert args.optimizer == "paged_adamw_8bit"
    assert args.max_length == 2048


def test_compute_sequence_logps_and_lengths():
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([[-100, 1, 2, 3, -100], [-100, -100, 4, 5, 6]])

    sum_logps, lengths = compute_sequence_logps_and_lengths(logits, labels, shift=True)
    assert sum_logps.shape == (batch_size,)
    assert lengths.shape == (batch_size,)
    assert (sum_logps < 0.0).all()
    assert lengths[0].item() == 3.0
    assert lengths[1].item() == 3.0


def test_compute_log_odds():
    sum_logps = torch.tensor([-2.0, -10.0])
    lengths = torch.tensor([2.0, 2.0])

    log_odds = compute_log_odds(sum_logps, lengths)
    assert log_odds.shape == (2,)
    assert log_odds[0] > log_odds[1]


def test_odds_ratio_loss_mechanics():
    batch_size = 2
    seq_len = 4
    vocab_size = 8

    chosen_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    chosen_labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    rejected_labels = torch.tensor([[5, 6, 7, 1], [3, 2, 1, 0]])

    output = odds_ratio_loss(
        policy_chosen_logits=chosen_logits,
        policy_chosen_labels=chosen_labels,
        policy_rejected_logits=rejected_logits,
        policy_rejected_labels=rejected_labels,
        lambda_or=0.1,
    )

    assert isinstance(output, ORPOLossOutput)
    assert output.loss.ndim == 0
    assert output.sft_loss.ndim == 0
    assert output.or_loss.ndim == 0
    assert output.log_odds_ratio.shape == (batch_size,)

    output.loss.backward()
    assert chosen_logits.grad is not None
    assert rejected_logits.grad is not None


def test_orpo_trainer_train_and_eval_step():
    torch.manual_seed(42)
    vocab_size = 16
    seq_len = 6
    model = TinyModel(vocab_size=vocab_size, hidden_dim=8)

    args = ORPOTrainingArguments(learning_rate=1e-3)
    trainer = SchemaSentinelORPOTrainer(model=model, args=args)

    batch = {
        "chosen_input_ids": torch.randint(0, vocab_size, (2, seq_len)),
        "chosen_labels": torch.randint(0, vocab_size, (2, seq_len)),
        "rejected_input_ids": torch.randint(0, vocab_size, (2, seq_len)),
        "rejected_labels": torch.randint(0, vocab_size, (2, seq_len)),
    }

    summary = trainer.get_summary()
    assert summary["reference_free"] is True
    assert summary["lambda_or"] == 0.1
    assert summary["global_step"] == 0

    eval_metrics = trainer.evaluate_step(batch)
    assert "eval_loss" in eval_metrics
    assert "eval_sft_loss" in eval_metrics
    assert "eval_or_loss" in eval_metrics
    assert "eval_log_odds_ratio" in eval_metrics
    assert "eval_accuracy" in eval_metrics

    train_metrics = trainer.train_step(batch)
    assert train_metrics["step"] == 1.0
    assert "loss" in train_metrics
    assert "sft_loss" in train_metrics
    assert "or_loss" in train_metrics
    assert "log_odds_ratio" in train_metrics
    assert "accuracy" in train_metrics
