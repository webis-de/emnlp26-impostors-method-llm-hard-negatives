# docker build -t registry.webis.de/code-research/theses/artificial-authorship-verification .
FROM python:3.11-slim

# Install dependencies before copying actual source files to make image updates faster.
# Install flash-attn separately, as it cannot be installed with build isolation and thus Poetry right now.
COPY pyproject.toml poetry.lock README.md /opt/artificial-authorship-verification/
WORKDIR /opt/artificial-authorship-verification

# Install Poetry and dependencies
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

RUN pip install .

RUN --mount=type=cache,target=/root/.cache set -x && \
    python3 -m poetry install

