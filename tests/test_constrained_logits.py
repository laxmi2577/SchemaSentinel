import pytest
import torch
from src.serving.constrained_logits import (
    GenerationPhase,
    ConstrainedLogitsConfig,
    mask_invalid_logits,
    mask_forbidden_keyword_tokens,
    SimpleVocabularyTokenizer,
    GrammarLogitProcessor,
)


def test_constrained_logits_config_defaults():
    config = ConstrainedLogitsConfig()
    assert config.mask_value == -1e9
    assert config.enforce_read_only is True
    assert config.forbid_mutations is True
    assert config.temperature == 1.0


def test_mask_invalid_logits_mechanism():
    vocab_size = 8
    logits = torch.zeros(1, vocab_size)

    allowed = [1, 3, 5]
    masked = mask_invalid_logits(logits, allowed_token_ids=allowed, mask_value=-1e9)

    for idx in range(vocab_size):
        if idx in allowed:
            assert masked[0, idx].item() == 0.0
        else:
            assert masked[0, idx].item() == -1e9


def test_mask_forbidden_keyword_tokens():
    vocab_size = 6
    logits = torch.ones(2, vocab_size) * 5.0

    forbidden = [0, 4]
    masked = mask_forbidden_keyword_tokens(logits, forbidden_token_ids=forbidden, mask_value=-1e9)

    for b in range(2):
        assert masked[b, 0].item() == -1e9
        assert masked[b, 4].item() == -1e9
        assert masked[b, 1].item() == 5.0
        assert masked[b, 2].item() == 5.0


def test_simple_vocabulary_tokenizer():
    tokenizer = SimpleVocabularyTokenizer()
    assert tokenizer.get_vocab_size() > 20

    encoded = tokenizer.encode("SELECT id FROM users")
    assert len(encoded) == 4

    decoded = tokenizer.decode(encoded)
    assert "SELECT" in decoded
    assert "users" in decoded

    forbidden_ids = tokenizer.find_token_ids(["drop", "delete", "truncate"])
    assert len(forbidden_ids) == 3


def test_grammar_logit_processor_detect_phase():
    tokenizer = SimpleVocabularyTokenizer()
    processor = GrammarLogitProcessor(tokenizer=tokenizer)

    assert processor.detect_generation_phase("") == GenerationPhase.START
    assert processor.detect_generation_phase("ALLOW") == GenerationPhase.SQL_FENCE_START
    assert processor.detect_generation_phase("ALLOW\n```sql\nSELECT id FROM users") == GenerationPhase.SQL_BODY
    assert processor.detect_generation_phase("ALLOW\n```sql\nSELECT id FROM users;\n```\n") == GenerationPhase.EVIDENCE
    assert processor.detect_generation_phase("ALLOW\n```sql\nSELECT id FROM users;\n```\nEVIDENCE: valid") == GenerationPhase.DONE


def test_grammar_logit_processor_extract_sql_candidate():
    tokenizer = SimpleVocabularyTokenizer()
    processor = GrammarLogitProcessor(tokenizer=tokenizer)

    extracted_partial = processor.extract_sql_candidate("ALLOW\n```sql\nSELECT id FROM users")
    assert extracted_partial == "SELECT id FROM users"

    extracted_closed = processor.extract_sql_candidate("ALLOW\n```sql\nSELECT id FROM users;\n```\nEVIDENCE: ok")
    assert extracted_closed == "SELECT id FROM users;"


def test_grammar_logit_processor_masks_forbidden_mutations():
    tokenizer = SimpleVocabularyTokenizer()
    processor = GrammarLogitProcessor(tokenizer=tokenizer)

    summary = processor.get_summary()
    assert summary["forbidden_tokens_count"] >= 5
    assert summary["forbid_mutations"] is True

    vocab_size = tokenizer.get_vocab_size()
    batch_size = 1
    scores = torch.ones(batch_size, vocab_size) * 2.0

    prompt = "ALLOW\n```sql\nSELECT id FROM users WHERE"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    processed_scores = processor(input_ids, scores)

    forbidden_ids = tokenizer.find_token_ids(["drop", "delete", "truncate", "alter", "update"])
    for f_id in forbidden_ids:
        assert processed_scores[0, f_id].item() == -1e9

    allowed_id = tokenizer.vocab["tenant_id"]
    assert processed_scores[0, allowed_id].item() == 2.0
