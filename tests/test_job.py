"""GPU-free tests for job settings, tokenization, batch order, and atomic writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dolly_lora.job import (
    IGNORE_INDEX,
    JobConfig,
    batch_indices,
    read_examples,
    tokenize_example,
    write_atomically,
)


class CharTokenizer:
    """Render chats as tagged text and tokenize one character per token."""

    pad_token_id = 0

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in conversation)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 250 + 2 for character in text]


def _chat(question: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def test_only_the_final_assistant_reply_is_supervised() -> None:
    tokenizer = CharTokenizer()
    example = tokenize_example(tokenizer, _chat("Hi?", "Yes."), max_tokens=512)

    assert example is not None
    prompt_length = len("<user>Hi?\n<assistant>")
    assert example.labels[:prompt_length] == (IGNORE_INDEX,) * prompt_length
    supervised = [label for label in example.labels if label != IGNORE_INDEX]
    assert supervised == tokenizer.encode("Yes.\n", add_special_tokens=False)


def test_truncation_that_removes_every_reply_token_drops_the_example() -> None:
    assert tokenize_example(CharTokenizer(), _chat("A long question?", "No."), max_tokens=5) is None


def test_batch_order_depends_only_on_step_and_covers_each_epoch() -> None:
    first_epoch = [
        index
        for step in range(3)
        for index in batch_indices(step, batch_size=4, example_count=12, seed=7)
    ]

    assert sorted(first_epoch) == list(range(12))
    assert batch_indices(5, batch_size=4, example_count=12, seed=7) == batch_indices(
        5, batch_size=4, example_count=12, seed=7
    )
    assert batch_indices(3, batch_size=4, example_count=12, seed=7) != batch_indices(
        0, batch_size=4, example_count=12, seed=7
    )


def test_fingerprint_changes_with_settings_or_data() -> None:
    config = JobConfig(base_model="m", model_revision="r", total_steps=10)

    assert config.fingerprint("a") == JobConfig(
        base_model="m", model_revision="r", total_steps=10
    ).fingerprint("a")
    assert config.fingerprint("a") != config.fingerprint("b")
    assert config.fingerprint("a") != JobConfig(
        base_model="m", model_revision="r", total_steps=11
    ).fingerprint("a")


def test_invalid_settings_and_examples_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="total_steps must be positive"):
        JobConfig(base_model="m", model_revision="r", total_steps=0)
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read_examples(path)


def test_atomic_write_never_exposes_a_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "state.pt"
    target.write_text("complete old state")

    def fail_midway(path: Path) -> None:
        path.write_text("partial")
        raise OSError("disk went away")

    with pytest.raises(OSError):
        write_atomically(target, fail_midway)

    assert target.read_text() == "complete old state"
    write_atomically(target, lambda path: path.write_text("complete new state"))
    assert target.read_text() == "complete new state"
