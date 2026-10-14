import pytest
from src.data_engine.collator import (
    CollatorConfig,
    CompletionOnlyCollator,
)


def test_find_subsequence():
    collator = CompletionOnlyCollator()
    seq = [10, 20, 30, 40, 50, 60]
    target = [30, 40]

    assert collator.find_subsequence(seq, target) == 2
    assert collator.find_subsequence(seq, [10, 20]) == 0
    assert collator.find_subsequence(seq, [50, 60]) == 4
    assert collator.find_subsequence(seq, [99, 100]) == -1
    assert collator.find_subsequence(seq, [10, 20, 30, 40, 50, 60, 70]) == -1


def test_mask_prompt_labels_exact_boundary():
    response_template = [151644, 77091, 198]
    config = CollatorConfig(
        response_template_ids=response_template,
        ignore_index=-100,
        pad_token_id=151643,
    )
    collator = CompletionOnlyCollator(config)

    prompt_tokens = [151644, 8948, 198, 200, 300, 151645, 198]
    completion_tokens = [1001, 1002, 1003, 151645]
    full_input_ids = prompt_tokens + response_template + completion_tokens

    labels = collator.mask_prompt_labels(full_input_ids)

    prompt_and_template_len = len(prompt_tokens) + len(response_template)
    assert len(labels) == len(full_input_ids)

    for i in range(prompt_and_template_len):
        assert labels[i] == -100

    for i in range(len(completion_tokens)):
        assert labels[prompt_and_template_len + i] == completion_tokens[i]


def test_mask_prompt_labels_missing_template():
    config = CollatorConfig(
        response_template_ids=[999, 998],
        ignore_index=-100,
    )
    collator = CompletionOnlyCollator(config)
    input_ids = [1, 2, 3, 4, 5]

    labels = collator.mask_prompt_labels(input_ids)
    assert labels == [-100, -100, -100, -100, -100]


def test_pad_sequences_multiple_of_8():
    collator = CompletionOnlyCollator(CollatorConfig(pad_to_multiple_of=8))
    seqs = [
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]

    padded = collator.pad_sequences(seqs, pad_value=0, pad_to_multiple_of=8)

    assert len(padded[0]) == 16
    assert len(padded[1]) == 16
    assert padded[0][:3] == [1, 2, 3]
    assert padded[0][3:] == [0] * 13
    assert padded[1][:10] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert padded[1][10:] == [0] * 6


def test_collate_batch_structure():
    response_template = [500]
    config = CollatorConfig(
        response_template_ids=response_template,
        pad_token_id=0,
        ignore_index=-100,
        pad_to_multiple_of=4,
    )
    collator = CompletionOnlyCollator(config)

    batch = [
        {"input_ids": [10, 20, 500, 100, 101]},
        {"input_ids": [10, 500, 200]},
    ]

    collated = collator.collate_batch(batch, return_tensors=False)

    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "labels" in collated

    assert len(collated["input_ids"]) == 2
    assert len(collated["input_ids"][0]) == 8
    assert len(collated["input_ids"][1]) == 8

    assert collated["attention_mask"][0] == [1, 1, 1, 1, 1, 0, 0, 0]
    assert collated["attention_mask"][1] == [1, 1, 1, 0, 0, 0, 0, 0]

    assert collated["labels"][0][:3] == [-100, -100, -100]
    assert collated["labels"][0][3:5] == [100, 101]
    assert collated["labels"][0][5:] == [-100, -100, -100]
