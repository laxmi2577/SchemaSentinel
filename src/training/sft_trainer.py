from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from src.training.config import (
    DoRAConfig,
    LoRAConfigModel,
    ModelTrainingConfig,
    QuantConfig,
    TrainingHyperparameters,
    build_peft_config,
)
from src.training.losses import (
    FORBIDDEN_SQL_KEYWORDS,
    HybridUnlikelihoodLoss,
    LossBreakdown,
    NegativeKeywordRegistry,
)


class AdapterType(str, Enum):
    QLORA = "qlora"
    DORA = "dora"


class SFTTrainingArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/sft"
    adapter_type: AdapterType = AdapterType.DORA
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    lambda_ul: float = 0.5
    use_unlikelihood_loss: bool = True
    seed: int = 42
    bf16: bool = True
    fp16: bool = False


def get_trainable_parameters_summary(model: Any) -> Dict[str, Any]:
    if model is None:
        return {
            "trainable_params": 0,
            "all_params": 0,
            "trainable_percentage": 0.0,
        }

    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_elem = param.numel()
        all_params += num_elem
        if param.requires_grad:
            trainable_params += num_elem

    pct = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": round(pct, 4),
    }


class SchemaSentinelSFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: Optional[SFTTrainingArguments] = None,
        tokenizer: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        negative_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.model = model
        self.args = args or SFTTrainingArguments()
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.loss_fn = HybridUnlikelihoodLoss(
            negative_token_ids=negative_token_ids,
            lambda_ul=self.args.lambda_ul,
            shift_labels=True,
        )

        self.optimizer = self._build_optimizer()
        self.global_step = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            params = list(self.model.parameters())

        lr = self.args.learning_rate
        wd = self.args.weight_decay

        try:
            if self.args.optimizer == "paged_adamw_8bit" and torch.cuda.is_available():
                import bitsandbytes as bnb

                return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=wd)
        except Exception:
            pass

        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def compute_loss(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> LossBreakdown:
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", input_ids)
        token_weights = inputs.get("token_weights")

        try:
            device = next(self.model.parameters()).device
            if input_ids.device != device:
                input_ids = input_ids.to(device)
            if labels.device != device:
                labels = labels.to(device)
            if attention_mask is not None and attention_mask.device != device:
                attention_mask = attention_mask.to(device)
            if token_weights is not None and token_weights.device != device:
                token_weights = token_weights.to(device)
        except Exception:
            pass

        kwargs: Dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = self.model(input_ids, **kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if self.args.use_unlikelihood_loss:
            return self.loss_fn(logits, labels, token_weights=token_weights)
        else:
            ce_loss = self.loss_fn.ce_loss_module(logits, labels, token_weights=token_weights)
            zero_ul = torch.tensor(0.0, device=logits.device)
            return LossBreakdown(total_loss=ce_loss, ce_loss=ce_loss, ul_loss=zero_ul)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        breakdown = self.compute_loss(batch)
        breakdown.total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": float(breakdown.total_loss.item()),
            "ce_loss": float(breakdown.ce_loss.item()),
            "ul_loss": float(breakdown.ul_loss.item()),
            "step": float(self.global_step),
        }

    def evaluate_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            breakdown = self.compute_loss(batch)

        return {
            "eval_loss": float(breakdown.total_loss.item()),
            "eval_ce_loss": float(breakdown.ce_loss.item()),
            "eval_ul_loss": float(breakdown.ul_loss.item()),
        }

    def get_summary(self) -> Dict[str, Any]:
        params_info = get_trainable_parameters_summary(self.model)
        return {
            "adapter_type": self.args.adapter_type.value,
            "lambda_ul": self.args.lambda_ul,
            "use_unlikelihood_loss": self.args.use_unlikelihood_loss,
            "global_step": self.global_step,
            "params": params_info,
        }


def build_trainer(
    model: nn.Module,
    args: Optional[SFTTrainingArguments] = None,
    tokenizer: Optional[Any] = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    negative_token_ids: Optional[Sequence[int]] = None,
) -> SchemaSentinelSFTTrainer:
    return SchemaSentinelSFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        negative_token_ids=negative_token_ids,
    )
