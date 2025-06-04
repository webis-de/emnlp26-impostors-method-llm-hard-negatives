from abc import ABC, abstractmethod
import os, sys
from typing import Union
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk
sys.path.append(os.path.abspath(".."))
from config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector

from datasets import Dataset, DatasetDict, ClassLabel, Features, Value


class BaseDatasetVisualization(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load_dataset(self) -> Union[Dataset, DatasetDict]:
        pass

    def _flatten_list(self, nested_list):
        """
        Flattens a nested list into a single list.
        
        Args:
            nested_list (list): A list that may contain other lists.
        
        Returns:
            list: A flattened list containing all elements.
        """
        return [s for item in nested_list for s in item.flatten().tolist() if isinstance(item, np.ndarray)]


    def plot_text_length_histogram(self, text_list, bins=10, preprocessed:bool = False, verbose:bool = False):
        """
        Plots a histogram of text lengths (in characters) for a list of strings.

        Args:
            text_list (list): List of strings or ngrams.
            bins (int): Number of bins in the histogram.
            preprocessed (bool): If True, indicates that the text has been stemmed.
            unit (str): Unit of measurement for text length, either 'characters' or 'ngrams'.
        """
        if isinstance(text_list[0], np.ndarray):
            text_list = self._flatten_list(text_list)
        lengths = [len(text) for text in text_list]
        unit = 'characters' if type(lengths[0]) is str else 'ngrams'
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=bins, color='skyblue', edgecolor='black')
        title = 'Histogram of Text Lengths (Stemmed)' if preprocessed else 'Histogram of Text Lengths'
        plt.title(title)
        plt.xlabel(f'Text Length ({unit})')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        if not verbose:
            print(f"Average length (characters) per text: {np.mean(lengths):.2f}")

   

class Pan23Visualization(BaseDatasetVisualization):
    def __init__(self):
        super().__init__("pan23")

    def load_dataset(self) -> DatasetDict:
        ds_pan = load_from_disk(os.path.join(os.path.abspath(".."), CONFIG.PATH2PAN23))['train'].to_pandas()
        return ds_pan
    
    def plot_avg_text_length_per_author(self, df, unit='characters'):
        """
        Plots a bar chart of average text length per author.

        Args:
            df (pd.DataFrame): DataFrame with 'author' and 'text' columns.
        """
        if unit == 'ngrams':
            impostor_det = ImpostorDetector(df, top_n=250, portion_delete=0.5, shared_vocab_only=True, strict=True)
            df['text_length'] = df['text'].apply(lambda text: len(impostor_det.tokenize_char_ngrams(text, n=4, normalize_ws=True, space_free=True)))
        else:
            # Default to characters
            df['text_length'] = df['text'].str.len()
        avg_len = df.groupby('author')['text_length'].mean().sort_values()

        # Plot
        plt.figure(figsize=(10, 8))
        avg_len.plot(kind='barh', color='mediumseagreen', edgecolor='black')
        plt.xlabel(f'Average Text Length ({unit})')
        plt.title('Average Text Length per Author')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def boxplots_text_len_ngrams(self, df):
        """
        Plot two boxplots side by side in the same plot:
        1) Average text length per author
        2) All text lengths
        Outliers on average lengths are annotated.

        Args:
            df (pd.DataFrame): DataFrame with 'author' and 'text' columns.
        """

        impostor_det = ImpostorDetector(df, top_n=250, portion_delete=0.5, shared_vocab_only=True, strict=True)
        df['n_ngram'] = df['text'].apply(lambda text: len(impostor_det.tokenize_char_ngrams(text, n=4, normalize_ws=True, space_free=True)))
        df['text_length'] = df['text'].str.len()
        
        avg_text_len = df.groupby('author')['text_length'].mean()
        avg_n_gram = df.groupby('author')['n_ngram'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # boxplots vertically stacked (y=2 for text length, y=1 for ngram)
        data = [avg_text_len, avg_n_gram]
        positions = [2, 1]
        
        box = ax.boxplot(data, positions=positions, vert=False, widths=0.6, patch_artist=True)
        
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # Y-axis labels
        ax.set_yticks(positions)
        ax.set_yticklabels(['Avg Text Length per Author', 'Avg ngram Count per Author'])
        ax.set_xlabel('Value')
        ax.set_title('Boxplots of Average Text Length and ngram Count per Author')
        
        # Function to find and annotate outliers
        def annotate_outliers(variable, pos, color):
            Q1 = variable.quantile(0.25)
            Q3 = variable.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = variable[(variable < lower) | (variable > upper)]
            for author, val in outliers.items():
                ax.plot(val, pos, 'o', color=color)
                ax.text(val, pos + 0.1, author, rotation=45, ha='center', va='bottom', fontsize=8)
        
        # Annotate outliers in each boxplot
        annotate_outliers(avg_text_len, 2, 'blue')
        annotate_outliers(avg_n_gram, 1, 'green')
        
        # Legend for outliers
        ax.plot([], [], 'o', color='blue', label='Outliers in Avg Text Length')
        ax.plot([], [], 'o', color='green', label='Outliers in Avg ngram')
        ax.legend(loc='upper right')
        
        ax.set_ylim(0.5, 2.5)
        plt.tight_layout()
        plt.show()

    

