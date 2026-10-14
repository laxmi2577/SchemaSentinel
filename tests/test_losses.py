import pytest
import torch
from src.training.losses import (
    FORBIDDEN_SQL_KEYWORDS,
    LossBreakdown,
    NegativeKeywordRegistry,
    MaskedCrossEntropyLoss,
    KeywordUnlikelihoodLoss,
    HybridUnlikelihoodLoss,
)


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "DROP": 101,
            "drop": 102,
            "Drop": 103,
            " DROP": 104,
            " drop": 105,
            "\nDROP": 106,
            "DELETE": 107,
            "delete": 108,
            "Delete": 109,
            " DELETE": 110,
            " delete": 111,
            "\nDELETE": 112,
            "SELECT": 201,
            "select": 202,
            "from": 203,
        }

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self.vocab:
            return [self.vocab[text]]
        return [999]


def test_negative_keyword_registry_defaults():
    registry = NegativeKeywordRegistry()
    assert "DROP" in registry.keywords
    assert "DELETE" in registry.keywords
    assert "TRUNCATE" in registry.keywords
    assert "ALTER" in registry.keywords
    assert "UPDATE" in registry.keywords
    assert "INSERT" in registry.keywords
    assert "GRANT" in registry.keywords
    assert "REVOKE" in registry.keywords

    tokenizer = MockTokenizer()
    extracted_ids = registry.extract_token_ids(tokenizer)
    assert len(extracted_ids) > 0
    assert 101 in extracted_ids
    assert 102 in extracted_ids


def test_masked_cross_entropy_basic():
    torch.manual_seed(42)
    loss_fn = MaskedCrossEntropyLoss(ignore_index=-100, shift_labels=True)

    batch_size = 2
    seq_len = 6
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.tensor([
        [-100, -100, 2, 4, 1, 0],
        [-100, 3, 5, 2, -100, -100],
    ])

    loss = loss_fn(logits, labels)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0


def test_masked_cross_entropy_all_masked():
    loss_fn = MaskedCrossEntropyLoss(ignore_index=-100, shift_labels=True)
    logits = torch.randn(2, 4, 10, requires_grad=True)
    labels = torch.full((2, 4), -100, dtype=torch.long)

    loss = loss_fn(logits, labels)
    assert loss.item() == 0.0


def test_masked_cross_entropy_weighted_tokens():
    loss_fn = MaskedCrossEntropyLoss(ignore_index=-100, shift_labels=True)

    logits = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3]])

    unweighted_loss = loss_fn(logits, labels)

    weights = torch.tensor([[1.0, 3.0, 3.0, 3.0]])
    weighted_loss = loss_fn(logits, labels, token_weights=weights)

    assert weighted_loss.item() > 0.0
    assert abs(weighted_loss.item() - 3.0 * unweighted_loss.item()) < 1e-4


def test_keyword_unlikelihood_loss_penalizes_forbidden_tokens():
    loss_fn = KeywordUnlikelihoodLoss(negative_token_ids=[7, 8], shift_labels=True)

    batch_size = 1
    seq_len = 3
    vocab_size = 10

    safe_logits = torch.zeros(batch_size, seq_len, vocab_size, requires_grad=True)
    with torch.no_grad():
        safe_logits[:, :, 1] = 10.0
    labels = torch.tensor([[-100, 1, 1]])

    safe_loss = loss_fn(safe_logits, labels)

    dangerous_logits = torch.zeros(batch_size, seq_len, vocab_size, requires_grad=True)
    with torch.no_grad():
        dangerous_logits[:, :, 7] = 10.0

    dangerous_loss = loss_fn(dangerous_logits, labels)

    assert dangerous_loss.item() > safe_loss.item()
    assert dangerous_loss.item() > 1.0


def test_keyword_unlikelihood_loss_empty_or_masked():
    empty_fn = KeywordUnlikelihoodLoss(negative_token_ids=[], shift_labels=True)
    logits = torch.randn(2, 4, 10, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 1, 2, 3]])
    assert empty_fn(logits, labels).item() == 0.0

    normal_fn = KeywordUnlikelihoodLoss(negative_token_ids=[2], shift_labels=True)
    all_masked_labels = torch.full((2, 4), -100, dtype=torch.long)
    assert normal_fn(logits, all_masked_labels).item() == 0.0


def test_hybrid_unlikelihood_loss_breakdown():
    hybrid_fn = HybridUnlikelihoodLoss(
        negative_token_ids=[4, 5],
        lambda_ul=0.5,
        shift_labels=True,
    )

    logits = torch.randn(2, 5, 12, requires_grad=True)
    labels = torch.tensor([
        [-100, -100, 1, 2, 3],
        [-100, 2, 4, 1, -100],
    ])

    breakdown = hybrid_fn(logits, labels)
    assert isinstance(breakdown, LossBreakdown)
    assert breakdown.total_loss.dim() == 0
    assert breakdown.ce_loss.dim() == 0
    assert breakdown.ul_loss.dim() == 0

    expected_total = breakdown.ce_loss + 0.5 * breakdown.ul_loss
    assert abs(breakdown.total_loss.item() - expected_total.item()) < 1e-5

    breakdown.total_loss.backward()
    assert logits.grad is not None
