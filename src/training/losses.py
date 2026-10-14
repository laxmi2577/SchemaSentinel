from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set
import torch
import torch.nn as nn
import torch.nn.functional as F


FORBIDDEN_SQL_KEYWORDS: List[str] = [
    "DROP",
    "DELETE",
    "TRUNCATE",
    "ALTER",
    "UPDATE",
    "INSERT",
    "GRANT",
    "REVOKE",
]


@dataclass
class LossBreakdown:
    total_loss: torch.Tensor
    ce_loss: torch.Tensor
    ul_loss: torch.Tensor


class NegativeKeywordRegistry:
    def __init__(self, keywords: Optional[List[str]] = None) -> None:
        self.keywords = keywords or list(FORBIDDEN_SQL_KEYWORDS)

    def extract_token_ids(self, tokenizer: Any) -> List[int]:
        if tokenizer is None:
            return []

        token_ids: Set[int] = set()
        for kw in self.keywords:
            variants = [kw, kw.lower(), kw.capitalize(), f" {kw}", f" {kw.lower()}", f"\n{kw}"]
            for v in variants:
                ids = tokenizer.encode(v, add_special_tokens=False)
                for t in ids:
                    token_ids.add(t)

        return sorted(list(token_ids))


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        shift_labels: bool = True,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.shift_labels = shift_labels

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.shift_labels:
            if logits.size(1) <= 1:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = token_weights[..., 1:].contiguous() if token_weights is not None else None
        else:
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()
            shift_weights = token_weights.contiguous() if token_weights is not None else None

        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        active_mask = flat_labels != self.ignore_index
        active_count = active_mask.sum().float()

        if active_count == 0.0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if shift_weights is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")
            per_token_loss = loss_fn(flat_logits, flat_labels)
            flat_weights = shift_weights.view(-1)
            weighted_loss = (per_token_loss * flat_weights).sum()
            return weighted_loss / active_count
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
            return loss_fn(flat_logits, flat_labels)


class KeywordUnlikelihoodLoss(nn.Module):
    def __init__(
        self,
        negative_token_ids: Optional[Sequence[int]] = None,
        ignore_index: int = -100,
        shift_labels: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.shift_labels = shift_labels
        self.eps = eps
        token_list = list(negative_token_ids) if negative_token_ids is not None else []
        self.register_buffer(
            "neg_token_ids",
            torch.tensor(token_list, dtype=torch.long) if token_list else torch.empty(0, dtype=torch.long),
        )

    def set_negative_token_ids(self, token_ids: Sequence[int], device: Optional[torch.device] = None) -> None:
        tensor_ids = torch.tensor(list(token_ids), dtype=torch.long, device=device)
        self.register_buffer("neg_token_ids", tensor_ids)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.neg_token_ids.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if self.shift_labels:
            if logits.size(1) <= 1:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()

        active_mask = shift_labels != self.ignore_index
        active_count = active_mask.sum().float()

        if active_count == 0.0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = F.softmax(shift_logits, dim=-1)

        active_probs = probs[active_mask]
        valid_neg_ids = self.neg_token_ids[self.neg_token_ids < active_probs.size(-1)]

        if valid_neg_ids.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        neg_probs = active_probs[:, valid_neg_ids]
        clamp_probs = torch.clamp(1.0 - neg_probs, min=self.eps, max=1.0)
        neg_log_probs = -torch.log(clamp_probs)
        total_ul_penalty = neg_log_probs.sum()

        return total_ul_penalty / active_count


class HybridUnlikelihoodLoss(nn.Module):
    def __init__(
        self,
        negative_token_ids: Optional[Sequence[int]] = None,
        lambda_ul: float = 0.5,
        ignore_index: int = -100,
        shift_labels: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_ul = lambda_ul
        self.ce_loss_module = MaskedCrossEntropyLoss(
            ignore_index=ignore_index,
            shift_labels=shift_labels,
        )
        self.ul_loss_module = KeywordUnlikelihoodLoss(
            negative_token_ids=negative_token_ids,
            ignore_index=ignore_index,
            shift_labels=shift_labels,
        )

    def set_negative_token_ids(self, token_ids: Sequence[int], device: Optional[torch.device] = None) -> None:
        self.ul_loss_module.set_negative_token_ids(token_ids, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
    ) -> LossBreakdown:
        ce_loss = self.ce_loss_module(logits, labels, token_weights=token_weights)
        ul_loss = self.ul_loss_module(logits, labels)
        total_loss = ce_loss + (self.lambda_ul * ul_loss)
        return LossBreakdown(
            total_loss=total_loss,
            ce_loss=ce_loss,
            ul_loss=ul_loss,
        )
