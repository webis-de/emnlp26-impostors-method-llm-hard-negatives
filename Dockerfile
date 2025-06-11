# docker build -t registry.webis.de/code-research/theses/artificial-authorship-verification .
FROM python:3.11-slim

# Copy only dependency-related files
COPY pyproject.toml poetry.lock /src/
WORKDIR /src

# Install Poetry and project dependencies (without source code)
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root
