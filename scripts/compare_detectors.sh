#!/bin/bash
#SBATCH --job-name=detectors
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-repr-koppel-%j.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/vis_detectors.py --rounds 100 --top_n 100000 --impostor_technique "fixed" --fig2reproduce 2
