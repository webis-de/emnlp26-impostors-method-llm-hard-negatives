#!/usr/bin/env bash

genai-detection dataset convert \
    -h data/datasets/dataset-extended-2025-part/human \
    -m data/datasets/dataset-extended-2025-part/machines \
    --test-ids data/datasets/dataset-extended-2025-part/ids-test.txt \
    --model-name-parent 1 \
    --recursive \
    --val-split-size 0.1 \
    --output data/datasets/pan25-o1-case-study-converted
