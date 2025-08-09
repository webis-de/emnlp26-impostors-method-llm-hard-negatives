#!/bin/bash
#SBATCH --job-name=syn_acc
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-syn_p_acc.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src 
python3 genai_detection/experiments/impact_similarity_paraphrases_imp_score.py 
