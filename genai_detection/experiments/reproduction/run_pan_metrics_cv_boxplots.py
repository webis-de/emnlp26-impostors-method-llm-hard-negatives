import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
from pathlib import Path

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_visualization import (
    plot_pan_metrics_boxplots,
)

logger = logging.getLogger(__name__)

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "reproduction" / "boxplots"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

metrics = [
    "precision",
    "recall",
    # "f1",
    # "accuracy",
    # "c_at_1",
    # "auroc",
    "auroc_c_at_1",
]

plot_pan_metrics_boxplots(
    dataset_name=CONFIG.BLOG,
    save_path=LOCAL_SAVE_PATH,
    metrics=metrics,
)
logger.info("Done with blog.")

plot_pan_metrics_boxplots(
    dataset_name=CONFIG.STUDENT_ESSAYS,
    save_path=LOCAL_SAVE_PATH,
    metrics=metrics,
)
logger.info("Done with student essays.")