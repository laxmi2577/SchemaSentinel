from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CollatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pad_token_id: int = 151643
    ignore_index: int = -100
    response_template_ids: List[int] = Field(default_factory=list)
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8


class CompletionOnlyCollator:
    def __init__(self, config: Optional[CollatorConfig] = None) -> None:
        self.config = config or CollatorConfig()

    def find_subsequence(self, sequence: List[int], target: List[int]) -> int:
        if not target or len(target) > len(sequence):
            return -1

        target_len = len(target)
        for i in range(len(sequence) - target_len + 1):
            if sequence[i : i + target_len] == target:
                return i
        return -1

    def mask_prompt_labels(
        self,
        input_ids: List[int],
        response_template_ids: Optional[List[int]] = None,
        ignore_index: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> List[int]:
        target_template = response_template_ids or self.config.response_template_ids
        ign_idx = ignore_index if ignore_index is not None else self.config.ignore_index
        pad_idx = pad_token_id if pad_token_id is not None else self.config.pad_token_id

        if not target_template:
            return [ign_idx if tok == pad_idx else tok for tok in input_ids]

        match_idx = self.find_subsequence(input_ids, target_template)
        if match_idx == -1:
            return [ign_idx] * len(input_ids)

        completion_start_idx = match_idx + len(target_template)
        labels: List[int] = []

        for i, tok in enumerate(input_ids):
            if i < completion_start_idx or tok == pad_idx:
                labels.append(ign_idx)
            else:
                labels.append(tok)

        return labels

    def pad_sequences(
        self,
        sequences: List[List[int]],
        pad_value: int,
        target_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> List[List[int]]:
        max_batch_len = max(len(seq) for seq in sequences) if sequences else 0
        final_len = target_length or max_batch_len

        if pad_to_multiple_of and pad_to_multiple_of > 0:
            remainder = final_len % pad_to_multiple_of
            if remainder != 0:
                final_len += pad_to_multiple_of - remainder

        padded_results: List[List[int]] = []
        for seq in sequences:
            if len(seq) > final_len:
                padded = seq[:final_len]
            else:
                padded = seq + [pad_value] * (final_len - len(seq))
            padded_results.append(padded)

        return padded_results

    def collate_batch(
        self,
        batch: List[Dict[str, Any]],
        return_tensors: bool = False,
    ) -> Dict[str, Any]:
        raw_input_ids: List[List[int]] = []
        raw_labels: List[List[int]] = []
        raw_attention_masks: List[List[int]] = []

        for item in batch:
            input_ids = list(item.get("input_ids", []))
            if "labels" in item:
                labels = list(item["labels"])
            else:
                labels = self.mask_prompt_labels(input_ids)

            raw_input_ids.append(input_ids)
            raw_labels.append(labels)
            raw_attention_masks.append([1] * len(input_ids))

        padded_inputs = self.pad_sequences(
            raw_input_ids,
            pad_value=self.config.pad_token_id,
            target_length=self.config.max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        padded_labels = self.pad_sequences(
            raw_labels,
            pad_value=self.config.ignore_index,
            target_length=self.config.max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        padded_masks = self.pad_sequences(
            raw_attention_masks,
            pad_value=0,
            target_length=self.config.max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        output: Dict[str, Any] = {
            "input_ids": padded_inputs,
            "attention_mask": padded_masks,
            "labels": padded_labels,
        }

        if return_tensors and HAS_TORCH:
            output["input_ids"] = torch.tensor(padded_inputs, dtype=torch.long)
            output["attention_mask"] = torch.tensor(padded_masks, dtype=torch.long)
            output["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return output
