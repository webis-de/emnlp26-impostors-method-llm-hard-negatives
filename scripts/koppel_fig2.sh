#!/bin/bash
#SBATCH --job-name=fig_2_content
#SBATCH --mem=256g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-fig2-content.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/vis_detectors.py --rounds 100 --top_n 100000 --impostor_technique "content" --fig2reproduce 2
