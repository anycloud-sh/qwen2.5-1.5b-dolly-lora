"""Job settings, data preparation, batch order, and atomic checkpoint files.

Everything here is deterministic and GPU-free, so a resumed container rebuilds exactly the same
examples and batch sequence as the container it replaces.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

STATE_FILE = "state.pt"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class JobConfig:
    """Frozen training settings read from ``job.json``."""

    base_model: str
    model_revision: str
    total_steps: int
    batch_size: int = 4
    checkpoint_every_steps: int = 25
    learning_rate: float = 1e-4
    lora_rank: int = 16
    max_tokens: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a bounded, resumable run."""
        for name in ("total_steps", "batch_size", "checkpoint_every_steps", "lora_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")
        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def load(cls, path: Path) -> JobConfig:
        """Read and validate one ``job.json`` document."""
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("job.json must contain one JSON object")
        return cls(**document)

    def fingerprint(self, dataset_sha256: str) -> str:
        """Identify these settings plus the exact dataset bytes, for resume safety."""
        payload = json.dumps({**asdict(self), "dataset_sha256": dataset_sha256}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class ChatTokenizer(Protocol):
    """The tokenizer surface used to render chat examples."""

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        """Render one conversation as model-specific text."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Return the token IDs for ``text``."""
        ...


@dataclass(frozen=True)
class TokenizedExample:
    """One example with the loss applied only to its final assistant message."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def read_examples(path: Path) -> tuple[list[list[dict[str, str]]], str]:
    """Read chat examples from JSON Lines and return them with the file's SHA-256.

    Each line is ``{"messages": [{"role": ..., "content": ...}, ...]}`` ending with an
    assistant message.
    """
    raw = path.read_bytes()
    examples: list[list[dict[str, str]]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        messages = json.loads(line).get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(m, dict)
                and isinstance(m.get("role"), str)
                and isinstance(m.get("content"), str)
                for m in messages
            )
            or messages[-1]["role"] != "assistant"
        ):
            raise ValueError(f"examples.jsonl line {number} is not a chat ending with assistant")
        examples.append([{"role": m["role"], "content": m["content"]} for m in messages])
    if not examples:
        raise ValueError("examples.jsonl contains no examples")
    return examples, hashlib.sha256(raw).hexdigest()


def tokenize_example(
    tokenizer: ChatTokenizer, messages: list[dict[str, str]], max_tokens: int
) -> TokenizedExample | None:
    """Tokenize one chat, masking everything except the final assistant reply.

    Returns ``None`` when truncation to ``max_tokens`` would leave no supervised token.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    shared = 0
    for prompt_id, full_id in zip(prompt_ids, full_ids, strict=False):
        if prompt_id != full_id:
            break
        shared += 1
    input_ids = full_ids[:max_tokens]
    labels = [IGNORE_INDEX] * min(shared, len(input_ids)) + input_ids[shared:]
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return TokenizedExample(input_ids=tuple(input_ids), labels=tuple(labels))


def batch_indices(step: int, *, batch_size: int, example_count: int, seed: int) -> list[int]:
    """Return the example indices for one 0-based step.

    The order is a fresh seeded shuffle per epoch and depends only on the step number, so a
    resumed run continues the exact sequence the interrupted run would have used.
    """
    indices = []
    for position in range(step * batch_size, (step + 1) * batch_size):
        epoch, offset = divmod(position, example_count)
        order = list(range(example_count))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        indices.append(order[offset])
    return indices


class WriteFile(Protocol):
    """Callable that writes one complete file at the given path."""

    def __call__(self, path: Path) -> None:
        """Write the file."""
        ...


def write_atomically(path: Path, write: WriteFile) -> None:
    """Write ``path`` through a fsynced temporary file and an atomic rename.

    AnyCloud copies the checkpoint directory to its bucket on a timer, so a reader may see the
    directory at any moment. The rename guarantees ``path`` is always a complete file.
    """
    temporary = path.with_name(f"{path.name}.tmp")
    write(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def losses_summary(losses: Sequence[float]) -> dict[str, float]:
    """Summarize recorded step losses for ``result.json``."""
    head = list(losses[: min(10, len(losses))])
    tail = list(losses[-min(10, len(losses)) :])
    return {
        "first_10_mean": sum(head) / len(head),
        "last_10_mean": sum(tail) / len(tail),
    }
