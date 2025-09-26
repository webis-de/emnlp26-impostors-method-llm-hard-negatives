#!/bin/bash
#SBATCH --job-name=eval_llm_impostors_prec_rec
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-eval-llm-impostors-prec-rec.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/experiments/evaluate_llm_impostors_baselines_prec_recall.py --rounds 100 --top_n 100000 --impostor_technique "fixed" 
