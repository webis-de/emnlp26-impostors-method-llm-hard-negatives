import argparse
import logging
from pathlib import Path

import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.impostor_metrics import LABEL_TRANSLATIONS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def csv_to_latex_table(
    csv_path: Path,
    caption: str,
    label: str,
    float_format: str = "%.3f",
) -> str:
    """
    Convert a CSV heatmap (n_selected × n_potential) to a LaTeX table.
    """
    df = pd.read_csv(csv_path, index_col=0)

    latex = df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        bold_rows=True,
    )

    return latex


def generate_latex_tables_from_dir(
    csv_dir: Path,
    output_tex: Path,
):
    """
    Generate a LaTeX file containing tables for all CSVs in a directory.
    """
    tables = []

    if output_tex.is_dir():
        output_tex = output_tex / "out.tex"

    logger.info(f"Generating LaTeX tables for {csv_dir}.\nOutput file: {output_tex}")

    for csv_file in sorted(csv_dir.glob("*.csv")):
        name = csv_file.stem

        technique = name.split("_")[1]
        metric = name.split("_")[2]

        caption = (
            f"Maximum {metric.upper()} scores for "
            f"{LABEL_TRANSLATIONS.get(technique, technique)}"
        )
        label = f"tab:{name}"

        tables.append(
            csv_to_latex_table(
                csv_file,
                caption=caption,
                label=label,
            )
        )


    output_tex.write_text("\n\n".join(tables))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=False,
        help="Path to the directory containing the CSV files.",
        default=Path(__file__).resolve().parents[3]
            / CONFIG.SAVE_PATH
            / "reproduction"
            / "n_imp_selection_config"/"in_domain",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        help="Path to the output LaTeX file.",
        default=Path(__file__).resolve().parents[3] / CONFIG.SAVE_PATH,
    )
    args = parser.parse_args()
    generate_latex_tables_from_dir(
        args.csv_path,
        args.output_dir,
    )
