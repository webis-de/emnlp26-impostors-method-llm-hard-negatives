from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB


class ParaphraseScoresVisualizer():
    def __init__(self):
        self.mongoDB = ParaphraseMongoDB()
        self.original_collection = self.mongoDB.original_collection
        self.paraphrase_collection = self.mongoDB.paraphrase_collection
        self.paraphrase_score_collection = self.mongoDB.paraphrase_score_collection
        self.save_dir_path = (Path(__file__).parent.parent.parent / CONFIG.SAVE_PATH / "openai_paraphrases" /
                           "visualization")
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

    def scores_per_original_document(self, original_document_id:str):
        """
        Fetch paraphrase documents for a given original document ID,
        convert them to a DataFrame, and display distributions of numeric features.
        :param original_document_id: ID of the original document in the original collection.
        """
        # get scores for paraphrases for original document ID
        cursor = self.mongoDB.find_document_by_non_id_field(collection=self.paraphrase_score_collection,
                                                            document_field_name="text_id",
                                                            document_value=original_document_id)
        # Convert cursor to list of dicts
        paraphrase_docs = list(cursor)

        if not paraphrase_docs:
            print(
                f"No paraphrases found for original document ID: {original_document_id}"
            )
            return None  # or return empty DataFrame

        # Convert to DataFrame
        df = pd.DataFrame(paraphrase_docs)

        # Optionally remove non-numeric columns before describing
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            print("No numeric features found in the documents.")
            return df

        # Display distribution of numeric features
        print(
            f"Distribution of numeric features for original document ID {original_document_id}:"
        )
        print(numeric_df.describe())

        # Plot each numeric feature
        for column in numeric_df.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(numeric_df[column], bins=20, color="skyblue", edgecolor="black")
            plt.title(
                f"{column} distribution for original document {original_document_id}"
            )
            plt.xlabel(column)
            plt.ylabel("Frequency")

            # Create speaking filename
            safe_column_name = column.replace(" ", "_").lower()
            filename = (
                self.save_dir_path / f"{original_document_id}_{safe_column_name}.svg"
            )

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()  # Close figure to free memory
        if "sem_sim_avg" in numeric_df.columns and "syn_sim_avg" in numeric_df.columns:
            fig = plt.figure(figsize=(12, 6))
            main_ax = fig.add_axes(
                [0.1, 0.1, 0.6, 0.85]
            )  # [left, bottom, width, height]
            main_ax.scatter(
                numeric_df["sem_sim_avg"],
                numeric_df["syn_sim_avg"],
                c="blue",
                alpha=0.6,
                edgecolor="k",
            )
            main_ax.set_title(
                f"Semantic vs. Syntactic Similarity for original document {original_document_id}"
            )
            main_ax.set_xlabel("Average Semantic similarity")
            main_ax.set_ylabel("Average Syntactic similarity")

            # Inset with full range
            inset_size = 0.25
            inset_ax = fig.add_axes([0.73, 0.1, inset_size, inset_size])

            # Scatter points in small axes
            inset_ax.scatter(
                numeric_df["sem_sim_avg"],
                numeric_df["syn_sim_avg"],
                c="red",
                alpha=0.6,
                edgecolor="k",
                s=10,  # smaller points for inset
            )
            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            inset_ax.set_title("Full range")
            inset_ax.grid(True)
            inset_ax.set_xticks([0, 0.5, 1])
            inset_ax.set_yticks([0, 0.5, 1])
            inset_ax.tick_params(axis="both", which="major", labelsize=8)

            # Optional: style inset border
            for spine in inset_ax.spines.values():
                spine.set_edgecolor("gray")

            # Create speaking filename
            filename = (
                self.save_dir_path / f"{original_document_id}_sem_vs_syn_scatter.svg"
            )

            plt.savefig(filename)
            plt.close()

        print(f"Plots saved to {self.save_dir_path}")

        return df  # return full DataFrame for further use


if __name__ == "__main__":
    visualizer = ParaphraseScoresVisualizer()
    visualizer.scores_per_original_document(original_document_id="68f50029edacdf3d5c0279e8")
