#!/bin/bash
#SBATCH --job-name=imposter_dataset_creation
#SBATCH --mem=128g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-dataset-creation-%j.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/opt/artificial-authorship-verification
python3 genai_detection/dataset_util.py
