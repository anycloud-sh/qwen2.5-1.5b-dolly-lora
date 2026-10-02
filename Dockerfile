FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

LABEL org.opencontainers.image.source="https://github.com/anycloud-sh/anycloud-spot-finetune"
LABEL org.opencontainers.image.description="LoRA fine-tuning that survives spot interruptions on AnyCloud"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install . && mkdir -p /cache/huggingface

CMD ["python", "-m", "spot_finetune.train"]
