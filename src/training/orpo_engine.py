from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field


class ORPOTrainingArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/orpo"
    lambda_or: float = 0.1
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2
    max_prompt_length: int = 1024
    max_length: int = 2048
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    seed: int = 42


@dataclass
class ORPOLossOutput:
    loss: torch.Tensor
    sft_loss: torch.Tensor
    or_loss: torch.Tensor
    log_odds_ratio: torch.Tensor
    accuracy: torch.Tensor


def compute_sequence_logps_and_lengths(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    shift: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if shift:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != ignore_index

    clamped_labels = shift_labels.clone()
    clamped_labels[~mask] = 0

    per_token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=clamped_labels.unsqueeze(-1),
    ).squeeze(-1)

    sum_logps = (per_token_logps * mask.float()).sum(dim=-1)
    lengths = mask.float().sum(dim=-1).clamp(min=1.0)

    return sum_logps, lengths


def compute_log_odds(
    sum_logps: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    avg_logps = sum_logps / lengths
    avg_logps = avg_logps.clamp(max=-eps)
    p = torch.exp(avg_logps).clamp(min=eps, max=1.0 - eps)
    one_minus_p = (1.0 - p).clamp(min=eps)
    return avg_logps - torch.log(one_minus_p)


def odds_ratio_loss(
    policy_chosen_logits: torch.Tensor,
    policy_chosen_labels: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    policy_rejected_labels: torch.Tensor,
    lambda_or: float = 0.1,
    ignore_index: int = -100,
) -> ORPOLossOutput:
    chosen_sum_logps, chosen_lengths = compute_sequence_logps_and_lengths(
        policy_chosen_logits,
        policy_chosen_labels,
        ignore_index=ignore_index,
    )
    rejected_sum_logps, rejected_lengths = compute_sequence_logps_and_lengths(
        policy_rejected_logits,
        policy_rejected_labels,
        ignore_index=ignore_index,
    )

    sft_loss = - (chosen_sum_logps / chosen_lengths).mean()

    chosen_log_odds = compute_log_odds(chosen_sum_logps, chosen_lengths)
    rejected_log_odds = compute_log_odds(rejected_sum_logps, rejected_lengths)

    log_odds_ratio = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    total_loss = sft_loss + (lambda_or * or_loss)
    accuracy = (log_odds_ratio > 0.0).float().mean()

    return ORPOLossOutput(
        loss=total_loss,
        sft_loss=sft_loss,
        or_loss=or_loss,
        log_odds_ratio=log_odds_ratio,
        accuracy=accuracy,
    )


class SchemaSentinelORPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: Optional[ORPOTrainingArguments] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.args = args or ORPOTrainingArguments()
        self.tokenizer = tokenizer

        self.optimizer = self._build_optimizer()
        self.global_step = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            params = list(self.model.parameters())

        lr = self.args.learning_rate
        try:
            if self.args.optimizer == "paged_adamw_8bit" and torch.cuda.is_available():
                import bitsandbytes as bnb

                return bnb.optim.PagedAdamW8bit(params, lr=lr)
        except Exception:
            pass

        return torch.optim.AdamW(params, lr=lr)

    def _forward_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            device = next(self.model.parameters()).device
            if input_ids.device != device:
                input_ids = input_ids.to(device)
            if attention_mask is not None and attention_mask.device != device:
                attention_mask = attention_mask.to(device)
        except Exception:
            pass
        kwargs: Dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.model(input_ids, **kwargs)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def compute_batch_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> ORPOLossOutput:
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch.get("chosen_labels", chosen_input_ids)
        chosen_mask = batch.get("chosen_attention_mask")

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch.get("rejected_labels", rejected_input_ids)
        rejected_mask = batch.get("rejected_attention_mask")

        chosen_logits = self._forward_model(chosen_input_ids, chosen_mask)
        rejected_logits = self._forward_model(rejected_input_ids, rejected_mask)

        chosen_labels = chosen_labels.to(chosen_logits.device)
        rejected_labels = rejected_labels.to(rejected_logits.device)

        return odds_ratio_loss(
            policy_chosen_logits=chosen_logits,
            policy_chosen_labels=chosen_labels,
            policy_rejected_logits=rejected_logits,
            policy_rejected_labels=rejected_labels,
            lambda_or=self.args.lambda_or,
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        output = self.compute_batch_loss(batch)
        output.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": float(output.loss.item()),
            "sft_loss": float(output.sft_loss.item()),
            "or_loss": float(output.or_loss.item()),
            "log_odds_ratio": float(output.log_odds_ratio.mean().item()),
            "accuracy": float(output.accuracy.item()),
            "step": float(self.global_step),
        }

    def evaluate_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            output = self.compute_batch_loss(batch)

        return {
            "eval_loss": float(output.loss.item()),
            "eval_sft_loss": float(output.sft_loss.item()),
            "eval_or_loss": float(output.or_loss.item()),
            "eval_log_odds_ratio": float(output.log_odds_ratio.mean().item()),
            "eval_accuracy": float(output.accuracy.item()),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "lambda_or": self.args.lambda_or,
            "learning_rate": self.args.learning_rate,
            "global_step": self.global_step,
            "loss_type": "odds_ratio_preference_optimization",
            "reference_free": True,
        }
