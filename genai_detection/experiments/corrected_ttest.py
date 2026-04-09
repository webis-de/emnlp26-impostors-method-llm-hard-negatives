"""
This implementation is adapted from the `correctipy` project:
https://github.com/hendersontrent/correctipy (08.04.2026)

Original code is licensed under the MIT License.

Modifications:
- Fixed an error in the variance / standard error computation
  (removed an extra square root in the denominator).

This file therefore contains a modified version of the original
implementation and is not a verbatim copy.
"""

import numpy as np
import pandas as pd
# Change code to fix import error
from scipy.stats import t


def repkfold_ttest(data, n1:int, n2:int, k:int, r:int):
    """
    Compute correlated t-statistic and p-value for repeated k-fold cross-validated results.
    Args:
        data (dataframe): dataframe of values for model A and model B over repeated k-fold cross-validation
        n1 (int): train set size
        n2 (int): test set size
        k (int): number of folds used in k-fold
        r (int): number of repeats per fold

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """

    # Arg checks
    if "model" not in data.columns:
        raise ValueError(
            "data should contain at least four columns called 'model', 'values', 'k', and 'r'."
        )
    if "values" not in data.columns:
        raise ValueError(
            "data should contain at least four columns called 'model', 'values', 'k', and 'r'."
        )
    if "k" not in data.columns:
        raise ValueError(
            "data should contain at least four columns called 'model', 'values', 'k', and 'r'."
        )
    if "r" not in data.columns:
        raise ValueError(
            "data should contain at least four columns called 'model', 'values', 'k', and 'r'."
        )

    if not (
        np.issubdtype(data["values"].dtype, np.number)
        and np.issubdtype(data["k"].dtype, np.number)
        and np.issubdtype(data["r"].dtype, np.number)
    ):
        raise ValueError(
            "data should be a data.frame with only numerical values in columns 'values', 'k', and 'r'."
        )

    if not (
        np.isscalar(n1)
        and np.isscalar(n2)
        and np.isscalar(k)
        and np.isscalar(r)
        and np.isreal(n1)
        and np.isreal(n2)
        and np.isreal(k)
        and np.isreal(r)
    ):
        raise ValueError("n1, n2, k, and r should all be integer scalars.")

    if len(data["model"].unique()) != 2:
        raise ValueError(
            "Column 'model' in data should only have two unique labels (one for each model to compare)."
        )

    d = []

    for i in range(1, k + 1):
        for j in range(1, r + 1):
            x = data[(data["k"] == i) & (data["r"] == j)]
            model_values = x.groupby(by="model").agg({"values": "mean"})
            d.append(model_values.values[0, 0] - model_values.values[1, 0])

    # altered code from https://github.com/hendersontrent/correctipy/blob/main/correctipy/statistics.py#L126C31-L126C53
    # to fix obsolete sqrt (cf. Section 3.3. in https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)
    statistic = np.mean(d) / (
        np.sqrt(np.var(d, ddof=1, keepdims=True) * ((1 / (k * r)) + (n2 / n1)))
    )  # Calculate t-statistic

    if statistic < 0:
        p_value = t.cdf(statistic, (k * r) - 1)  # p-value for left tail
    else:
        p_value = t.sf(statistic, (k * r) - 1)  # p-value for right tail

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df
