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

import logging

import numpy as np
import pandas as pd
# Change code to fix import error
from scipy.stats import t

logger = logging.getLogger(__name__)

# renamed variables to avoid confusion
def repkfold_ttest(data, train_set_size:int, test_set_size:int, k:int, r:int):
    """
    Compute correlated t-statistic and p-value for repeated k-fold cross-validated results.
    Args:
        data (dataframe): dataframe of values for model A and model B over repeated k-fold cross-validation
        train_set_size (int): train set size
        test_set_size (int): test set size
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
        np.isscalar(train_set_size)
        and np.isscalar(test_set_size)
        and np.isscalar(k)
        and np.isscalar(r)
        and np.isreal(train_set_size)
        and np.isreal(test_set_size)
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

    d = np.asarray(d, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        logger.debug(
            "repkfold_ttest: insufficient or non-finite diffs; len=%s, non_finite=%s",
            len(d),
            int(np.size(d) - np.isfinite(d).sum()),
        )
        stat_df = pd.DataFrame({"statistic": [np.nan], "p_value": [np.nan]})
        return stat_df

    var_d = np.var(d, ddof=1, keepdims=True)
    if np.all(var_d == 0):
        logger.debug(
            "repkfold_ttest: zero variance in diffs; len=%s, mean=%s",
            len(d),
            float(np.mean(d)),
        )
        stat_df = pd.DataFrame({"statistic": [0.0], "p_value": [1.0]})
        return stat_df

    # altered code from https://github.com/hendersontrent/correctipy/blob/main/correctipy/statistics.py#L126C31-L126C53
    # to fix obsolete sqrt (cf. Section 3.3. in https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)
    statistic = np.mean(d) / (
        np.sqrt(var_d * ((1 / (k * r)) + (test_set_size / train_set_size)))
    )  # Calculate t-statistic

    if statistic < 0:
        p_value = t.cdf(statistic, (k * r) - 1)  # p-value for left tail
    else:
        p_value = t.sf(statistic, (k * r) - 1)  # p-value for right tail

    stat_df = pd.DataFrame({"statistic": [statistic], "p_value": [p_value]})
    return stat_df
