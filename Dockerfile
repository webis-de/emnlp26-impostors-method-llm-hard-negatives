# docker build -t registry.webis.de/code-teaching/theses/artificial-authorship-verification .
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Copy only dependency-related files
WORKDIR /src
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install poetry packaging setuptools \
    && python3 -m poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock README.md ./
COPY genai_detection/__init__.py ./genai_detection/

RUN sed -i.bak 's/+cpu", source = "torch-cpu"/+cu128", source = "torch-cu128"/g' pyproject.toml
RUN poetry lock
RUN --mount=type=cache,target=/root/.cache \
    poetry install

COPY . .
