#!/bin/bash
#SBATCH --job-name=paraphrase_eval
#SBATCH --time=2-2:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-paraphrase-eval-%j.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 run_evaluation.py
