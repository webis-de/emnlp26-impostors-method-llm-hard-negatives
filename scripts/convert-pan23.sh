#!/bin/bash
#SBATCH --job-name=imposter_pan23
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-data-gen-%j.log
#SBATCH --container-image=registry.webis.de/code-research/arguana/args/args-acquisition:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/args-acquisition/:/src
python3 genai_detection/dataset_util.py --path "/mnt/ceph/storage/data-in-progress/data-research/authorship/pan23-authorship-verification/" --out "/mnt/ceph/storage/data-tmp/2024/kgutekunst/dev/artificial-authorship-verification/" 
