"""Write the validation job's input files from a pinned slice of Databricks Dolly 15k.

Dolly 15k is licensed CC BY-SA 3.0 by Databricks. The slice is converted to chat examples whose
user turn is the instruction (plus context, when present) and whose assistant turn is the
human-written response.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DATASET_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DATASET_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
    f"{DATASET_REVISION}/databricks-dolly-15k.jsonl"
)
JOB = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "total_steps": 600,
    "batch_size": 4,
    "checkpoint_every_steps": 25,
    "learning_rate": 0.0001,
    "lora_rank": 16,
    "max_tokens": 512,
    "seed": 0,
}


def main() -> None:
    """Download the pinned dataset and write ``job.json`` and ``examples.jsonl``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--examples", type=int, default=2000)
    arguments = parser.parse_args()
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response:  # noqa: S310
        rows = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows[: arguments.examples]:
            prompt = row["instruction"].strip()
            if row["context"].strip():
                prompt = f"{prompt}\n\n{row['context'].strip()}"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["response"].strip()},
            ]
            handle.write(json.dumps({"messages": messages}) + "\n")
    (arguments.output_dir / "job.json").write_text(json.dumps(JOB, indent=2) + "\n")


if __name__ == "__main__":
    main()
