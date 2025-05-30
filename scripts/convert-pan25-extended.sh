#!/usr/bin/env bash

# Excludes german, high-temperature, alpaca, gpt2
genai-detection dataset convert \
    -h data/datasets/dataset-extended-2025-part/human \
    -m data/datasets/dataset-extended-2025-part/machines \
    --test-ids data/datasets/dataset-extended-2025-part/ids-test.txt \
    --recursive \
    --model-name-parent 1 \
    --val-split-size 2000 \
    --output data/datasets/dataset-extended-2025-part-converted
