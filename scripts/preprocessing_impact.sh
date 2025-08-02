#!/bin/bash
#SBATCH --job-name=preprocessing_impact
#SBATCH --mem=128g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-preprocessing-impact-%j.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/experiments/impact_preprocessing.py
