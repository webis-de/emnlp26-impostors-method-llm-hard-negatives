#!/bin/bash
#SBATCH --job-name=evaluate_paraphrasers
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-comp-paraphrasers-%j.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/experiments/run_evaluation.py 
