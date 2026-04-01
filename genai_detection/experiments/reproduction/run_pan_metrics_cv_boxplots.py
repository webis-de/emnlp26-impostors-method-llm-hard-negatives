from pathlib import Path

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics import plot_pan_metrics_boxplots

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "reproduction" / "boxplots"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

plot_pan_metrics_boxplots(
    dataset_name=CONFIG.BLOG,
    save_path=LOCAL_SAVE_PATH,
)

# plot_pan_metrics_boxplots(
#     dataset_name=CONFIG.STUDENT_ESSAYS,
#     save_path=LOCAL_SAVE_PATH,
# )
