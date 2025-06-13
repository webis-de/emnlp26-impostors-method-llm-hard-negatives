# docker build -t registry.webis.de/code-research/theses/artificial-authorship-verification .
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Copy only dependency-related files
WORKDIR /src
COPY setup.cfg setup.py ./

RUN --mount=type=cache,target=/root/.cache \
    pip install ./
