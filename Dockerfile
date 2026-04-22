# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv python3-pip \
        git build-essential wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so system pip stays clean
RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"

# torch 2.7.0 + CUDA 12.8 (matches our Targon setup)
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel for torch 2.7 / cu12 / cp312 / cxx11abi=TRUE
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN wget -q "${FA_URL}" -O /tmp/flash_attn.whl \
 && pip install /tmp/flash_attn.whl \
 && rm /tmp/flash_attn.whl

# Source + install
WORKDIR /opt/reliquary
COPY . /opt/reliquary
RUN pip install -e .

# bittensor 10.2.0 ships with async-substrate-interface 2.0 which conflicts
# with its own scalecodec import path — roll back to the 1.x line that matches.
RUN pip uninstall -y cyscale \
 && pip install 'async-substrate-interface<2.0.0' \
 && pip install --force-reinstall --no-deps scalecodec==1.2.12

# boto3 for R2 (weight-only mode + trainer archive uploads)
RUN pip install boto3

# Runtime
ENV GRAIL_ATTN_IMPL=flash_attention_2
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/opt/entrypoint.sh"]
