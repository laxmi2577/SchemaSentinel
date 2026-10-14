from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field


class DPOTrainingArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/dpo"
    beta: float = 0.1
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2
    max_prompt_length: int = 1024
    max_length: int = 2048
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    seed: int = 42


@dataclass
class DPOLossOutput:
    loss: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    reward_margin: torch.Tensor
    accuracy: torch.Tensor


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    shift: bool = True,
) -> torch.Tensor:
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

    return (per_token_logps * mask.float()).sum(dim=-1)


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios
    reward_margin = beta * logits

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    if label_smoothing == 0.0:
        losses = -F.logsigmoid(reward_margin)
    else:
        losses = (
            -F.logsigmoid(reward_margin) * (1.0 - label_smoothing)
            - F.logsigmoid(-reward_margin) * label_smoothing
        )

    loss = losses.mean()
    accuracy = (reward_margin > 0.0).float().mean()

    return DPOLossOutput(
        loss=loss,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margin=reward_margin,
        accuracy=accuracy,
    )


def compute_margins(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> Dict[str, float]:
    margin = chosen_rewards - rejected_rewards
    acc = (margin > 0.0).float().mean()
    return {
        "chosen_reward_mean": float(chosen_rewards.mean().item()),
        "rejected_reward_mean": float(rejected_rewards.mean().item()),
        "reward_margin_mean": float(margin.mean().item()),
        "accuracy": float(acc.item()),
    }


class SchemaSentinelDPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        args: Optional[DPOTrainingArguments] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.args = args or DPOTrainingArguments()
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
        target_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            device = next(target_model.parameters()).device
            if input_ids.device != device:
                input_ids = input_ids.to(device)
            if attention_mask is not None and attention_mask.device != device:
                attention_mask = attention_mask.to(device)
        except Exception:
            pass
        kwargs: Dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = target_model(input_ids, **kwargs)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def compute_batch_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> DPOLossOutput:
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch.get("chosen_labels", chosen_input_ids)
        chosen_attention_mask = batch.get("chosen_attention_mask")

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch.get("rejected_labels", rejected_input_ids)
        rejected_attention_mask = batch.get("rejected_attention_mask")

        policy_chosen_logits = self._forward_model(self.model, chosen_input_ids, chosen_attention_mask)
        policy_rejected_logits = self._forward_model(self.model, rejected_input_ids, rejected_attention_mask)

        chosen_labels = chosen_labels.to(policy_chosen_logits.device)
        rejected_labels = rejected_labels.to(policy_rejected_logits.device)

        policy_chosen_logps = compute_log_probs(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = compute_log_probs(policy_rejected_logits, rejected_labels)

        with torch.no_grad():
            if self.ref_model is not None:
                ref_chosen_logits = self._forward_model(self.ref_model, chosen_input_ids, chosen_attention_mask)
                ref_rejected_logits = self._forward_model(self.ref_model, rejected_input_ids, rejected_attention_mask)
            elif hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    ref_chosen_logits = self._forward_model(self.model, chosen_input_ids, chosen_attention_mask)
                    ref_rejected_logits = self._forward_model(self.model, rejected_input_ids, rejected_attention_mask)
            else:
                ref_chosen_logits = policy_chosen_logits.detach()
                ref_rejected_logits = policy_rejected_logits.detach()

            ref_chosen_logps = compute_log_probs(ref_chosen_logits, chosen_labels)
            ref_rejected_logps = compute_log_probs(ref_rejected_logits, rejected_labels)

        return compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            beta=self.args.beta,
            label_smoothing=self.args.label_smoothing,
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

        margin_stats = compute_margins(output.chosen_rewards, output.rejected_rewards)
        return {
            "loss": float(output.loss.item()),
            "reward_margin": margin_stats["reward_margin_mean"],
            "accuracy": margin_stats["accuracy"],
            "step": float(self.global_step),
        }

    def evaluate_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            output = self.compute_batch_loss(batch)

        margin_stats = compute_margins(output.chosen_rewards, output.rejected_rewards)
        return {
            "eval_loss": float(output.loss.item()),
            "eval_reward_margin": margin_stats["reward_margin_mean"],
            "eval_accuracy": margin_stats["accuracy"],
        }
