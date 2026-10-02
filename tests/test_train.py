"""End-to-end CPU tests: train, interrupt after a checkpoint, resume, export, and reload."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dolly_lora import train
from dolly_lora.job import JobConfig
from tests.test_job import CharTokenizer


def _tiny_model(config: JobConfig, device: torch.device) -> tuple[torch.nn.Module, object]:
    del config
    torch.manual_seed(1234)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
        )
    )
    return model.to(device), CharTokenizer()


def _paths(root: Path) -> train.JobPaths:
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "job.json").write_text(
        json.dumps(
            {
                "base_model": "tiny-llama",
                "model_revision": "test",
                "total_steps": 6,
                "batch_size": 2,
                "checkpoint_every_steps": 2,
                "learning_rate": 0.01,
                "lora_rank": 4,
                "max_tokens": 64,
            }
        )
    )
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Q{i}?"},
                {"role": "assistant", "content": f"A{i}."},
            ]
        }
        for i in range(5)
    ]
    (input_dir / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return train.JobPaths(input_dir, root / "output", root / "checkpoint")


class Preempted(BaseException):
    """Stands in for the container being stopped by a spot interruption."""


def test_resume_after_interruption_matches_an_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uninterrupted = train.run(_paths(tmp_path / "a"), load_model=_tiny_model)

    paths = _paths(tmp_path / "b")
    real_save = train._save

    def save_then_preempt(path, model, optimizer, fingerprint, step, losses):  # noqa: ANN001, ANN202
        real_save(path, model, optimizer, fingerprint, step, losses)
        if step == 4:
            raise Preempted

    monkeypatch.setattr(train, "_save", save_then_preempt)
    with pytest.raises(Preempted):
        train.run(paths, load_model=_tiny_model)
    assert not (paths.output_dir / "result.json").exists()
    monkeypatch.setattr(train, "_save", real_save)

    resumed = train.run(paths, load_model=_tiny_model)

    assert resumed["resumed_from_step"] == 4
    assert resumed["steps_trained_in_this_container"] == 2
    assert uninterrupted["resumed_from_step"] == 0
    assert len(resumed["step_losses"]) == 6
    assert resumed["step_losses"] == pytest.approx(uninterrupted["step_losses"])
    assert resumed["probe_loss_trained"] == pytest.approx(uninterrupted["probe_loss_trained"])
    assert resumed["adapter_reload_verified"] is True
    assert json.loads((paths.output_dir / "result.json").read_text()) == resumed
    assert (paths.output_dir / "adapter.tar.gz").stat().st_size > 0


def test_a_checkpoint_from_other_settings_is_never_resumed(tmp_path: Path) -> None:
    first = _paths(tmp_path / "a")
    train.run(first, load_model=_tiny_model)
    second = _paths(tmp_path / "b")
    shutil.copytree(first.checkpoint_dir, second.checkpoint_dir)
    job = json.loads((second.input_dir / "job.json").read_text())
    (second.input_dir / "job.json").write_text(json.dumps({**job, "learning_rate": 0.02}))

    with pytest.raises(RuntimeError, match="different job settings or data"):
        train.run(second, load_model=_tiny_model)
