#!/bin/bash
#SBATCH --job-name=eval_min_comp
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-min-comparison-prec-rec.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/experiments/minimal_comp_prec_recall_modularized.py --rounds 100 --top_n 100000 --impostor_technique "fixed" --fig2reproduce 5
