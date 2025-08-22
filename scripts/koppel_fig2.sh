#!/bin/bash
#SBATCH --job-name=fig_2_text_len
#SBATCH --mem=128g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j-fig2-text_len.log
#SBATCH --container-image=registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/artificial-authorship-verification/:/src
python3 genai_detection/vis_detectors.py --rounds 100 --top_n 100000 --impostor_technique "text_len" --fig2reproduce 2
