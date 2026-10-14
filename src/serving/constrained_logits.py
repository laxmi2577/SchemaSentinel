from __future__ import annotations

from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from src.serving.grammar import (
    FORBIDDEN_MUTATION_KEYWORDS,
    SQL_EBNF_GRAMMAR,
    validate_sql_syntax,
)


class GenerationPhase(str, Enum):
    START = "START"
    DECISION = "DECISION"
    SQL_FENCE_START = "SQL_FENCE_START"
    SQL_BODY = "SQL_BODY"
    SQL_FENCE_END = "SQL_FENCE_END"
    EVIDENCE = "EVIDENCE"
    DONE = "DONE"


class ConstrainedLogitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mask_value: float = -1e9
    enforce_read_only: bool = True
    forbid_mutations: bool = True
    temperature: float = 1.0
    eos_token_id: int = 151645
    pad_token_id: int = 151643


def mask_invalid_logits(
    logits: torch.Tensor,
    allowed_token_ids: Sequence[int],
    mask_value: float = -1e9,
) -> torch.Tensor:
    masked_logits = logits.clone()
    allowed_set = set(allowed_token_ids)
    vocab_size = logits.shape[-1]

    mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
    if allowed_set:
        allowed_tensor = torch.tensor(list(allowed_set), dtype=torch.long, device=logits.device)
        mask[allowed_tensor] = False

    masked_logits[..., mask] = mask_value
    return masked_logits


def mask_forbidden_keyword_tokens(
    logits: torch.Tensor,
    forbidden_token_ids: Sequence[int],
    mask_value: float = -1e9,
) -> torch.Tensor:
    if not forbidden_token_ids:
        return logits

    masked_logits = logits.clone()
    forbidden_tensor = torch.tensor(list(forbidden_token_ids), dtype=torch.long, device=logits.device)
    masked_logits[..., forbidden_tensor] = mask_value
    return masked_logits


class SimpleVocabularyTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        if vocab is not None:
            self.vocab = vocab
        else:
            default_words = [
                "<|im_start|>",
                "<|im_end|>",
                "assistant",
                "user",
                "system",
                "ALLOW",
                "REFUSE",
                "\n",
                "```sql\n",
                "```",
                "SELECT",
                "FROM",
                "WHERE",
                "JOIN",
                "ON",
                "GROUP",
                "BY",
                "HAVING",
                "ORDER",
                "LIMIT",
                "AND",
                "OR",
                "NOT",
                "IN",
                "BETWEEN",
                "LIKE",
                "IS",
                "NULL",
                "COUNT",
                "SUM",
                "AVG",
                "MIN",
                "MAX",
                "*",
                ",",
                ";",
                "=",
                ">",
                "<",
                ">=",
                "<=",
                "!=",
                "users",
                "orders",
                "sales",
                "id",
                "total",
                "region",
                "revenue",
                "tenant_id",
                "1",
                "10",
                "100",
                "'t1'",
                "EVIDENCE:",
                "Query",
                "executed",
                "successfully.",
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "TRUNCATE",
                "ALTER",
            ]
            self.vocab = {word: idx for idx, word in enumerate(default_words)}

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def decode(self, token_ids: Sequence[int]) -> str:
        words = [self.inv_vocab.get(int(i), "") for i in token_ids]
        return " ".join(w for w in words if w)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def find_token_ids(self, words: Sequence[str]) -> List[int]:
        target_words = {w.lower() for w in words}
        matching_ids: List[int] = []
        for word, idx in self.vocab.items():
            if word.lower().strip() in target_words:
                matching_ids.append(idx)
        return matching_ids


class GrammarLogitProcessor:
    def __init__(
        self,
        tokenizer: Any,
        config: Optional[ConstrainedLogitsConfig] = None,
        forbidden_keywords: Optional[Set[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or ConstrainedLogitsConfig()
        keywords = forbidden_keywords or FORBIDDEN_MUTATION_KEYWORDS
        self.forbidden_keywords = {k.lower() for k in keywords}

        self.forbidden_token_ids = self._identify_forbidden_token_ids()

    def _identify_forbidden_token_ids(self) -> Set[int]:
        ids: Set[int] = set()
        if hasattr(self.tokenizer, "find_token_ids"):
            found = self.tokenizer.find_token_ids(list(self.forbidden_keywords))
            ids.update(found)
        elif hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
            for word, idx in vocab.items():
                cleaned = re.sub(r"^[^\w]+", "", word.lower().strip())
                if cleaned in self.forbidden_keywords:
                    ids.add(idx)
        elif hasattr(self.tokenizer, "vocab"):
            for word, idx in self.tokenizer.vocab.items():
                cleaned = re.sub(r"^[^\w]+", "", word.lower().strip())
                if cleaned in self.forbidden_keywords:
                    ids.add(idx)
        return ids

    def detect_generation_phase(self, decoded_text: str) -> GenerationPhase:
        text = decoded_text.strip()
        if not text:
            return GenerationPhase.START

        if "```sql" in text and "```" in text.split("```sql", 1)[1]:
            if "EVIDENCE:" in text:
                return GenerationPhase.DONE
            return GenerationPhase.EVIDENCE

        if "```sql" in text:
            return GenerationPhase.SQL_BODY

        if text.startswith("ALLOW") or text.startswith("REFUSE"):
            return GenerationPhase.SQL_FENCE_START

        return GenerationPhase.DECISION

    def extract_sql_candidate(self, text: str) -> str:
        if "```sql" in text:
            parts = text.split("```sql", 1)[1]
            if "```" in parts:
                return parts.split("```", 1)[0].strip()
            return parts.strip()
        return ""

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        modified_scores = scores.clone()

        if self.config.forbid_mutations and self.forbidden_token_ids:
            for forbidden_id in self.forbidden_token_ids:
                if forbidden_id < modified_scores.shape[-1]:
                    modified_scores[..., forbidden_id] = self.config.mask_value

        for b in range(batch_size):
            seq = input_ids[b].tolist() if hasattr(input_ids[b], "tolist") else list(input_ids[b])
            decoded = self.tokenizer.decode(seq)
            phase = self.detect_generation_phase(decoded)

            if phase == GenerationPhase.SQL_BODY and self.config.enforce_read_only:
                current_sql = self.extract_sql_candidate(decoded)
                for forbidden_id in self.forbidden_token_ids:
                    if forbidden_id < modified_scores.shape[-1]:
                        modified_scores[b, forbidden_id] = self.config.mask_value

        return modified_scores

    def get_summary(self) -> Dict[str, Any]:
        return {
            "forbidden_tokens_count": len(self.forbidden_token_ids),
            "enforce_read_only": self.config.enforce_read_only,
            "forbid_mutations": self.config.forbid_mutations,
            "mask_value": self.config.mask_value,
        }
