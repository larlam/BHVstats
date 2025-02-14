"""
This module contains functions for performing t-tests.
"""

from typing import Optional
import numpy as np
from scipy.stats import t
from numpy import ndarray


def onesample_t_test(
    sample: ndarray, pop_mean: Optional[float] = 0.0, side="right"
) -> ndarray:
    """
    Performs a one-sided t test.

    Parameters
    ----------
    sample : ndarray
        [TODO:description]
    pop_mean : float
        The population mean under the null-hypothesis.

    Returns
    -------
    ndarray
        The p-value.
    """

    size = sample.shape[0]
    variance = np.var(sample, axis=0, ddof=1)
    delta = np.mean(sample, axis=0) - pop_mean
    t_statistic = size**0.5 * delta / variance**0.5
    t_dist = t(size - 1)
    if side == "right":
        p_value = 1 - t_dist.cdf(t_statistic)
    elif side == "both":
        # quants = np.vstack((t_dist.cdf(t_statistic), \
        #    1 - t_dist.cdf(t_statistic)))
        # p_value = 2 * np.min(quants, axis=0)
        p_value = 2 * t_dist.cdf(-np.abs(t_statistic))
    return p_value


def welch_test(first_sample: ndarray, second_sample: ndarray, side="both") -> ndarray:
    """
    Performs a Welch test for two given samples.

    Args:
        first_sample (ndarray): _description_
        second_sample (ndarray): _description_
        side (str, optional): _description_. Defaults to "both".

    Returns:
        ndarray: _description_
    """
    size_1 = first_sample.shape[0]
    size_2 = second_sample.shape[0]
    variance_1 = np.var(first_sample, axis=0, ddof=1)
    variance_2 = np.var(second_sample, axis=0, ddof=1)
    delta = np.mean(first_sample, axis=0) - np.mean(second_sample, axis=0)
    t_statistic = delta / (variance_1 / size_1 + variance_2 / size_2) ** 0.5
    degree = (variance_1 / size_1 + variance_2 / size_2) ** 2 / (
        (variance_1 / size_1) ** 2 + (variance_2 / size_2) ** 2
    )
    t_dist = t(degree)
    if side == "both":
        p_value = 2 * t_dist.cdf(-np.abs(t_statistic))
    return p_value
