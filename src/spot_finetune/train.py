"""Run one LoRA fine-tune as an AnyCloud spot Job, resuming from ``/mnt/checkpoint``.

AnyCloud copies ``/mnt/checkpoint`` to the Job's checkpoint bucket about every 60 seconds and
restores it before a replacement container starts after a spot interruption. This entrypoint only
has to keep one complete ``state.pt`` there and resume from it when it exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import signal
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Protocol

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft import set_peft_model_state_dict as set_adapter_state
from transformers import AutoModelForCausalLM, AutoTokenizer

from spot_finetune.job import (
    IGNORE_INDEX,
    STATE_FILE,
    JobConfig,
    TokenizedExample,
    batch_indices,
    losses_summary,
    read_examples,
    tokenize_example,
    write_atomically,
)

_log = logging.getLogger("spot_finetune")


class CausalModel(Protocol):
    """The subset of a Hugging Face causal LM the loop uses."""

    def __call__(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> object:
        """Return an output whose ``loss`` attribute is the mean token loss."""
        ...


ModelLoader = Callable[[JobConfig, torch.device], tuple[torch.nn.Module, object]]


@dataclass(frozen=True)
class JobPaths:
    """Directories AnyCloud mounts for a Job with input, output, and checkpoint buckets."""

    input_dir: Path
    output_dir: Path
    checkpoint_dir: Path


def load_hugging_face(config: JobConfig, device: torch.device) -> tuple[torch.nn.Module, object]:
    """Load the pinned base model and tokenizer from the Hugging Face Hub."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, revision=config.model_revision, dtype=dtype
    ).to(device)
    return model, tokenizer


def run(paths: JobPaths, *, load_model: ModelLoader = load_hugging_face) -> dict[str, object]:
    """Train to ``total_steps``, checkpointing and resuming through ``paths.checkpoint_dir``.

    Returns:
        The ``result.json`` document, also written to ``paths.output_dir``.
    """
    config = JobConfig.load(paths.input_dir / "job.json")
    conversations, dataset_sha256 = read_examples(paths.input_dir / "examples.jsonl")
    fingerprint = config.fingerprint(dataset_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, tokenizer = load_model(config, device)
    examples = [
        example
        for messages in conversations
        if (example := tokenize_example(tokenizer, messages, config.max_tokens)) is not None
    ]
    if not examples:
        raise ValueError("no example has a supervised token after truncation")
    model = get_peft_model(
        base,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    pad_id = _pad_token_id(tokenizer)

    state_path = paths.checkpoint_dir / STATE_FILE
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_step, losses = 0, []
    if state_path.exists():
        start_step, losses = _restore(state_path, model, optimizer, fingerprint, device)
        _log.info("resumed from checkpoint at step %d of %d", start_step, config.total_steps)
    else:
        torch.manual_seed(config.seed)
        _log.info("no checkpoint found; starting at step 0 of %d", config.total_steps)
    resumed_from_step = start_step

    model.train()
    started = time.monotonic()
    for step in range(start_step, config.total_steps):
        batch = [
            examples[index]
            for index in batch_indices(
                step,
                batch_size=config.batch_size,
                example_count=len(examples),
                seed=config.seed,
            )
        ]
        loss = _loss(model, batch, pad_id, device)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"step {step + 1} produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        completed = step + 1
        if completed % 10 == 0 or completed == config.total_steps:
            _log.info("step %d/%d loss %.4f", completed, config.total_steps, losses[-1])
        if completed % config.checkpoint_every_steps == 0 or completed == config.total_steps:
            _save(state_path, model, optimizer, fingerprint, completed, losses)
            _log.info("checkpoint saved at step %d", completed)

    result = _export_and_verify(
        paths, config, model, examples[: config.batch_size], pad_id, device, load_model
    )
    result.update(
        {
            "base_model": config.base_model,
            "model_revision": config.model_revision,
            "total_steps": config.total_steps,
            "resumed_from_step": resumed_from_step,
            "steps_trained_in_this_container": config.total_steps - resumed_from_step,
            "train_seconds_in_this_container": round(time.monotonic() - started, 1),
            "example_count": len(examples),
            "dataset_sha256": dataset_sha256,
            "step_losses": losses,
            "loss_summary": losses_summary(losses),
        }
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_atomically(
        paths.output_dir / "result.json",
        lambda path: path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"),
    )
    _log.info("training complete; result written")
    return result


def _pad_token_id(tokenizer: object) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_id, int):
        raise ValueError("the tokenizer has no pad token")
    return pad_id


def _loss(
    model: CausalModel, batch: Sequence[TokenizedExample], pad_id: int, device: torch.device
) -> torch.Tensor:
    width = max(len(example.input_ids) for example in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, example in enumerate(batch):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids)
        labels[row, :length] = torch.tensor(example.labels)
        attention[row, :length] = 1
    output = model(
        input_ids=input_ids.to(device),
        attention_mask=attention.to(device),
        labels=labels.to(device),
    )
    loss = getattr(output, "loss", None)
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError("the model returned no loss")
    return loss


def _save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fingerprint: str,
    step: int,
    losses: list[float],
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "step": step,
        "losses": list(losses),
        "adapter_state": get_peft_model_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    write_atomically(path, lambda temporary: torch.save(payload, temporary))


def _restore(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fingerprint: str,
    device: torch.device,
) -> tuple[int, list[float]]:
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "the checkpoint was written for different job settings or data; refusing to resume"
        )
    set_adapter_state(model, state["adapter_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng_state_all"]])
    return int(state["step"]), [float(loss) for loss in state["losses"]]


def _export_and_verify(
    paths: JobPaths,
    config: JobConfig,
    model: PeftModel,
    probe: Sequence[TokenizedExample],
    pad_id: int,
    device: torch.device,
    load_model: ModelLoader,
) -> dict[str, object]:
    """Export the adapter, then reload it into a fresh base model and compare a probe loss."""
    model.eval()
    with torch.no_grad():
        trained_loss = _loss(model, probe, pad_id, device).item()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    archive = paths.output_dir / "adapter.tar.gz"
    with tempfile.TemporaryDirectory(prefix="spot-finetune-") as directory:
        adapter_dir = Path(directory) / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)

        def write_archive(path: Path) -> None:
            # The fastest gzip level: adapter weights barely compress, and level 9 takes minutes.
            with tarfile.open(path, "w:gz", compresslevel=1) as tar:
                tar.add(adapter_dir, arcname="adapter")

        write_atomically(archive, write_archive)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fresh_base, _ = load_model(config, device)
        reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir).to(device)
        reloaded.eval()
        with torch.no_grad():
            reloaded_loss = _loss(reloaded, probe, pad_id, device).item()
    if not math.isclose(trained_loss, reloaded_loss, rel_tol=1e-2, abs_tol=1e-3):
        raise RuntimeError(
            f"reloaded adapter loss {reloaded_loss:.6f} differs from trained {trained_loss:.6f}"
        )
    return {
        "adapter_archive": archive.name,
        "adapter_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "adapter_size_bytes": archive.stat().st_size,
        "probe_loss_trained": trained_loss,
        "probe_loss_reloaded": reloaded_loss,
        "adapter_reload_verified": True,
    }


def _keep_training_on_sigterm(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    _log.warning("received SIGTERM; continuing until the container is stopped")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse mount paths, then train. Exit status 0 means the adapter was exported and verified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/output"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/mnt/checkpoint"))
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # A spot interruption stops the VM. Exiting 0 early would mark the Job completed, so a
    # SIGTERM is logged and training continues until the platform stops the container.
    signal.signal(signal.SIGTERM, _keep_training_on_sigterm)
    run(JobPaths(arguments.input_dir, arguments.output_dir, arguments.checkpoint_dir))


if __name__ == "__main__":
    main()
