import pytest
import torch
import torch.nn as nn
from src.training.dpo_engine import (
    DPOTrainingArguments,
    DPOLossOutput,
    compute_log_probs,
    compute_dpo_loss,
    compute_margins,
    SchemaSentinelDPOTrainer,
)


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embedding(input_ids)
        return self.head(x)


def test_dpo_training_arguments_defaults():
    args = DPOTrainingArguments()
    assert args.beta == 0.1
    assert args.learning_rate == 5e-5
    assert args.loss_type == "sigmoid"
    assert args.optimizer == "paged_adamw_8bit"
    assert args.max_length == 2048


def test_compute_log_probs():
    batch_size = 2
    seq_len = 4
    vocab_size = 8

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])

    logps = compute_log_probs(logits, labels, shift=True)
    assert logps.shape == (batch_size,)
    assert (logps < 0.0).all()


def test_compute_dpo_loss_mechanics():
    pi_chosen = torch.tensor([-1.0, -1.2], requires_grad=True)
    pi_rejected = torch.tensor([-4.0, -5.0], requires_grad=True)
    ref_chosen = torch.tensor([-2.0, -2.0])
    ref_rejected = torch.tensor([-2.5, -2.5])

    out_favors_chosen = compute_dpo_loss(
        policy_chosen_logps=pi_chosen,
        policy_rejected_logps=pi_rejected,
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=0.1,
    )
    assert isinstance(out_favors_chosen, DPOLossOutput)
    assert out_favors_chosen.accuracy.item() == 1.0
    assert out_favors_chosen.reward_margin.mean().item() > 0.0

    out_favors_chosen.loss.backward()
    assert pi_chosen.grad is not None
    assert pi_rejected.grad is not None


def test_compute_margins():
    chosen_r = torch.tensor([0.5, 0.8, 1.2])
    rejected_r = torch.tensor([0.1, 0.2, 0.3])

    stats = compute_margins(chosen_r, rejected_r)
    assert stats["chosen_reward_mean"] > stats["rejected_reward_mean"]
    assert stats["reward_margin_mean"] > 0.0
    assert stats["accuracy"] == 1.0


def test_dpo_trainer_train_and_eval_steps():
    model = TinyModel()
    ref_model = TinyModel()
    args = DPOTrainingArguments(beta=0.1, learning_rate=1e-3)
    trainer = SchemaSentinelDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
    )

    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
        "chosen_labels": torch.tensor([[-100, 2, 3, 4]]),
        "rejected_input_ids": torch.tensor([[1, 5, 6, 7]]),
        "rejected_labels": torch.tensor([[-100, 5, 6, 7]]),
    }

    train_res = trainer.train_step(batch)
    assert "loss" in train_res
    assert "reward_margin" in train_res
    assert "accuracy" in train_res
    assert train_res["step"] == 1.0
    assert trainer.global_step == 1

    eval_res = trainer.evaluate_step(batch)
    assert "eval_loss" in eval_res
    assert "eval_reward_margin" in eval_res
    assert "eval_accuracy" in eval_res
