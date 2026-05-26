# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
from pathlib import Path

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_visualization import (
    plot_pan_metrics_boxplots,
    plot_pan_metrics_heatmap_per_split,
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
    # "auroc_c_at_1",
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

plot_pan_metrics_heatmap_per_split(dataset_names=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS],
    save_path=None,
    metrics=metrics,
    methods=None
)

print("Done with heatmaps.")