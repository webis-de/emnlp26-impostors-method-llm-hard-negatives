# docker build -t registry.webis.de/code-research/theses/artificial-authorship-verification .
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Copy only dependency-related files
COPY pyproject.toml poetry.lock /src/
WORKDIR /src

# Install Poetry and project dependencies (without source code)
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip 
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# RUN pip install poetry 
# RUN poetry config virtualenvs.create false 
# RUN poetry install --no-root
RUN pip install .
