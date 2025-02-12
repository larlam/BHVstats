import numpy as np
from numpy import ndarray


def holm_correction(p_vals: ndarray) -> float:
    """
    For a given set of p-values, compute the Bonferroni-Holm correction.

    Parameters
    ----------
    p_vals : list[float]
        A list of p-values.

    Return
    -------
    p_holm : float
        The corrected p-value.
    """
    num_p = len(p_vals)
    p_holm = np.sort(p_vals)
    p_holm = p_holm * (num_p - np.arange(0, num_p))
    p_holm = min(p_holm)
    return p_holm


def simes_correction(p_vals: ndarray) -> float:
    """
    For a given set of p_values, compute the Simes correction.

    Parameters
    ----------
    p_vals : list[float]
        A list of p-values.

    Return
    -------
    p_simes : float
        The corrected p-value.
    """

    num_p = len(p_vals)
    p_simes = np.sort(p_vals)
    p_simes = p_simes * num_p / np.arange(1, num_p + 1)
    p_simes = min(p_simes)
    return p_simes


# TODO watch out if input is actually a list
def bonferroni_correction(p_vals: ndarray) -> float:
    """
    For a given set of p_values, compute the Bonferroni correction.

    Parameters
    ----------
    p_vals : list[float]
        A list of p-values.

    Return
    -------
    p_simes : float
        The corrected p-value.
    """
    num_p = len(p_vals)
    p_bon = p_vals * num_p
    p_bon = min(p_bon)
    return p_bon
