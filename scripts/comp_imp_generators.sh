#!/bin/bash
#SBATCH --job-name=content
#SBATCH --mem=256g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-imp-gen-comp-content.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src 
python3 genai_detection/experiments/comp_imp_generators.py 
