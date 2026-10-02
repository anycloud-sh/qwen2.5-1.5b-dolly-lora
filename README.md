# Qwen2.5-1.5B Dolly LoRA

AnyCloud-powered LoRA fine-tune of
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on
[Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), run as one
AnyCloud Job that resumes after interruptions. It runs on on-demand or spot GPUs. On spot, if the
cloud takes the VM back, AnyCloud starts a replacement, restores the latest checkpoint, and the same
command picks up where it stopped.

## How it works

The training code does two things:

- it writes one complete `state.pt` to `/mnt/checkpoint` every `checkpoint_every_steps` steps,
  holding the step counter with the adapter weights, optimizer state, and random-number state;
- on start, it resumes from `/mnt/checkpoint/state.pt` when the file exists.

AnyCloud does the rest:

- while the Job runs, it copies `/mnt/checkpoint` to the Job's checkpoint bucket about every 60
  seconds;
- when a spot VM is interrupted, it provisions a replacement under the same Job ID, restores the
  checkpoint bucket into `/mnt/checkpoint`, and runs the same command again;
- `/mnt/input` and `/mnt/output` are copied from and to the input and output buckets.

An interruption costs the steps since the last checkpoint that reached the bucket. AnyCloud does
not warn the container before the VM stops, so the job never relies on saving at the last moment.
Each checkpoint is written to a temporary file and renamed, so the bucket only ever holds complete
checkpoints. The checkpoint records a fingerprint of the job settings and the training data, and
the job refuses to resume from a checkpoint written for anything else.

The batch order depends only on the step number, so a resumed run trains on exactly the batches the
interrupted run would have used. At the end, the job exports the PEFT adapter to
`/mnt/output/adapter.tar.gz`, reloads it into a fresh copy of the base model, checks that it gives
the same loss on a probe batch, and writes `/mnt/output/result.json`. The process exits 0 only
after that check passes.

## Run it

Prepare the input files, upload them to an input bucket, and submit the Job. The input and
checkpoint buckets must already exist; AnyCloud creates the output bucket when needed. Use three
different bucket names.

```bash
python validation/prepare_dolly.py ./input
aws s3 cp --recursive ./input s3://YOUR-INPUT-BUCKET/

anycloud job ghcr.io/anycloud-sh/qwen2.5-1.5b-dolly-lora@sha256:DIGEST \
  --spot --credentials YOUR-AWS-CREDENTIALS --vm-type g5.xlarge --gpus all --disk-size 100 \
  --input-bucket YOUR-INPUT-BUCKET \
  --output-bucket YOUR-OUTPUT-BUCKET \
  --checkpoint-bucket YOUR-CHECKPOINT-BUCKET
```

Without `--checkpoint-bucket`, a spot Job still gets a checkpoint bucket that AnyCloud creates and
deletes with the Job. Naming one keeps the checkpoints after the Job ends.

`job.json` sets the base model and its exact revision, the step count, batch size, checkpoint
interval, learning rate, LoRA rank, maximum tokens per example, and seed. `examples.jsonl` holds one
chat per line, `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. Only the final
assistant message is trained on.

The validation input is the first 2,000 rows of
[Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA
3.0) at a pinned revision, trained into
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0).

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/pytest -q
```

The tests run on CPU with a tiny randomly initialized model. One interrupts a run right after a
checkpoint, reruns the same command, and checks that it resumes at that step and produces the same
per-step losses as an uninterrupted run.

Licensed under Apache-2.0.
