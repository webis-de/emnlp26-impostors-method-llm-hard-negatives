#!/bin/bash
#SBATCH --job-name=chunks
#SBATCH --mem=128g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%j-chunks.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/experiments/paraphrase_chunks.py --task evaluate