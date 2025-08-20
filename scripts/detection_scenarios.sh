#!/bin/bash
#SBATCH --job-name=detection_scenarios
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-detection_scenarios.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src 
python3 genai_detection/experiments/eval_llm_detection.py 
