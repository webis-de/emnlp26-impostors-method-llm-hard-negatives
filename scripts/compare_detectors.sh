#!/bin/bash
#SBATCH --job-name=imposter_pan23
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/log-comp-detectors-%j.log
#SBATCH --container-image=registry.webis.de/code-research/arguana/args/args-acquisition:latest
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kgutekunst/dev/args-acquisition/:/src
python3 genai_detection/vis_detectors.py --rounds 100 --top_n 100000 --imposter_technique "fixed" 
