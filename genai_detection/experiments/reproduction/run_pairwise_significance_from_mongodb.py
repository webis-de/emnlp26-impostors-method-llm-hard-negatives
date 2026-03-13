"""
Thin runner for pairwise significance testing on stored impostor outputs.

Execution steps:
1. Configure logging for experiment-style console output.
2. Delegate argument parsing and execution to
   `pairwise_significance_from_mongodb.main`.
3. Keep this file parallel to other `run_*` scripts for consistent usage.
"""

import logging

from genai_detection.experiments.reproduction.pairwise_significance_from_mongodb import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
