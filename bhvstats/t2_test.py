import numpy as np
import scipy.linalg
from scipy.stats import chi2, f
from numpy import ndarray


def paired_t2_test(first_sample : ndarray,
                   second_sample : ndarray,
                   test_dist = "f"
                  ) -> float:
    """
    Calculates a paired Hotelling's t-squared-test

    Parameters
    ----------
    first_sample, second_sample : ndarray
        The samples. Both samples must be of the same dimension.
    testdist : {"f", "chi2"}, optional
        Deceides whether the critical value is supposed to be computed
        using a chi-squared- or an F-distribution. The default is "f".

    Returns
    -------
    p_value : float
        The p-value for the given data.
    """
    size, dim = first_sample.shape
    delta =   np.mean(first_sample - second_sample, axis = 0)
    cov = np.cov(first_sample - second_sample, rowvar = False)
    t_squared = size * compute_t2(cov, delta)

    if test_dist == "f":
        statistic = t_squared * (size-dim) / (dim*(size - 1))
        f_dist = f(dim, size - dim)
        p_value = 1 - f_dist.cdf(statistic)

    elif test_dist == "chi2":
        statistic = t_squared
        chi2_dist = chi2(dim)
        p_value = 1 - chi2_dist.cdf(statistic)

    else:
        raise Exception("Use either chi2 or f as testdist.")

    return p_value

def twosamplet2test_equal(first_sample : ndarray,
                          second_sample : ndarray,
                          test_dist = "f"
                          ) -> float:
    """
    Calculates the Hotelling's t-squared-test for two independet samples.
    It is assumed that both samples have the same covariance.

    Parameters
    ----------
    first_sample, second_sample : ndarray
        The samples. Both samples must be of the same dimension.
    testdist : {"f", "chi2"}, optional
        Deceides whether the critical value is supposed to be computed
        using a chi-squared- or an F-distribution. The default is "f".

    Returns
    -------
    p_value : float
        The p-value for the given data.
    """
    size_1, dim = first_sample.shape
    size_2, _ = second_sample.shape
    delta = np.mean(first_sample, axis = 0) - np.mean(second_sample, axis = 0)
    cov_1 = np.cov(first_sample, rowvar = False)
    cov_2 = np.cov(second_sample, rowvar = False)
    cov_pooled = ((size_1-1) * cov_1 + (size_2-1) * cov_2) / (size_1+size_2-2)
    t_squared = (size_1*size_2)/(size_1+size_2) * compute_t2(cov_pooled, delta)

    if test_dist == "f":
        statistic = t_squared * (size_1+size_2-dim-1) / (dim*(size_1+size_2-2))
        f_dist = f(dim, size_2 + size_2 - dim-1)
        p_value = 1 - f_dist.cdf(statistic)

    elif test_dist == "chi2":
        statistic = t_squared
        chi2_dist = chi2(dim)
        p_value = 1 - chi2_dist.cdf(statistic)

    else:
        raise Exception("Use either chi2 or f as testdist.")

    return p_value


def twosamplet2test_unequal(first_sample : ndarray,
                          second_sample : ndarray,
                          test_dist = "f"
                          ) -> float:
    """
    Calculates the Hotelling's t-squared-test for two independet samples that
    do not have the same covariance.


    Parameters
    ----------
    first_sample, second_sample : ndarray
        The samples. Both samples must be of the same dimension.
    testdist : {"f", "chi2"}, optional
        Deceides whether the critical value is supposed to be computed
        using a chi-squared- or an F-distribution. The default is "f".

    Returns
    -------
    p_value : float
        The p-value for the given data.

    References
    ----------
    ..  [1] Ying Yao "An Approximate Degrees of Freedom Solution to the
        Multivariate Behrens Fisher Problem", Biometrika, vol. 52,
        no. 1/2, pp. 139-147, 1965

    """
    size_1, dim = first_sample.shape
    size_2, _ = second_sample.shape
    delta = np.mean(first_sample, axis = 0) - np.mean(second_sample, axis = 0)
    cov_1 = np.cov(first_sample, rowvar = False)
    cov_2 = np.cov(second_sample, rowvar = False)
    cov_pooled = cov_1 / size_1 + cov_2 / size_2
    t_squared = compute_t2(cov_pooled, delta)

    if test_dist == "f":
        dof = compute_dof(cov_1, cov_2, delta, t_squared, size_1, size_2)
        size_tot = size_1 + size_2 - 1
        statistic = t_squared * (size_tot - dim) / ((size_tot - 1) * dim)
        f_dist = f(dim, dof)
        p_value = 1 - f_dist.cdf(statistic)

    elif test_dist == "chi2":
        statistic = t_squared
        chi2_dist = chi2(dim)
        p_value = 1 - chi2_dist.cdf(statistic)

    else:
        raise Exception("Use either chi2 or f as testdist.")
    return p_value


def compute_dof(cov_1 : ndarray,
              cov_2 : ndarray,
              delta : ndarray,
              t_squared : float,
              size_1 : int,
              size_2 : int) -> float:
    """
    Computes the approximate degree of freedom.

    Parameters
    ----------
    cov_1,cov_2 : ndarray
        Empirical covariance matrices of the samples.
    delta : ndarray
        Array containing the differences of the two sample means.
    t_squared : float
        Value of the t-squared statistic.
    size_1,size_2 : int
        Sample sizes.

    Returns
    -------
    dof : float
        The approximate degree of freedom.

    """
    # compute it without iverting it using cholesky decompostion
    cov_pooled = cov_1 / size_1 + cov_2 / size_2
    c_fact, lower = scipy.linalg.cho_factor(cov_pooled)
    sol = scipy.linalg.cho_solve((c_fact, lower), delta)

    dof = 1 / (size_1 - 1) * ((sol.T @ cov_1/size_1 @ sol) / t_squared) ** 2
    dof += 1 / (size_2 - 1) * ((sol.T @ cov_2/size_1 @ sol) / t_squared) ** 2
    dof = 1 / dof

    return dof


def compute_t2(cov_pooled : ndarray, delta : ndarray) -> float:
    """
    Computes the inner product with respect to the inverse of the pooled
    covariance.


    Parameters
    ----------
    cov_pooled : ndarray
        Pooled covariance.
    delta : ndarray
        Difference of sample means.

    Returns
    -------
    float
        Inner product.
    """
    lower = scipy.linalg.cholesky(cov_pooled)
    sol = scipy.linalg.solve_triangular(lower.T, delta, lower = True)
    t_squared = np.linalg.norm(sol) ** 2
    return t_squared
