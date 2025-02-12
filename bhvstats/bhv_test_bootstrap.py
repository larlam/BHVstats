from typing import Union
import random
import numpy as np
from numpy import ndarray
from copy import copy
import matplotlib.pyplot as plt
from bhvstats.bhv_test import directions_in_data, directions_single_split
from bhvstats.phylo_tree import PhyloTree


def bhv_test(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    directions: Union[str, ndarray] = "single_split",
    bootstrap_samples: int = 49998,
    plot_pdf: bool = False,
) -> float:
    """
    Peforms a Kolmogorov-Smirnov test over a finite selection of the
    directional derivatives of the Frechet function. The quantiles can be
    either determined through permutations or bootsraps.

    Parameters
    ----------
    first_sample : list[PhyloTree]
        First sample of phylogenetic trees.
    second_sample : list[PhyloTree]
        Second sample of phylogenetic trees.
    directions :
        [TODO:description]
    bootstrap_samples
        The number of iterations for the bootstrap.
    plot_pdf
        [TODO:description]

    Returns
    -------
    float
        [TODO:description]

    Raises
    ------
    Exception:
        [TODO:description]
    """

    size_1 = len(first_sample)
    size_2 = len(second_sample)

    if isinstance(directions, str):
        if directions == "single_split":
            cosines = directions_single_split(first_sample + second_sample)
        elif directions == "from_data":
            cosines = directions_in_data(first_sample + second_sample)
    elif isinstance(directions, ndarray):
        if directions.shape[0] == size_1 + size_2:
            cosines = directions
        else:
            raise IndexError(
                "The dimensiosn of the given matrix do not match \
                             the sample size."
            )
    else:
        raise TypeError("Use supported way of picking directions")

    tree_norms = np.array([tree.norm() for tree in (first_sample + second_sample)])[
        :, None
    ]
    dir_derivs = -cosines * tree_norms

    statistic = compute_statistic(
        dir_derivs, list(range(0, size_1)), list(range(size_1, size_1 + size_2))
    )
    p_value = 0
    pdf_bs = []
    for _ in range(bootstrap_samples):
        index_1 = list(np.random.choice(range(size_1 + size_2), size_1))
        index_2 = list(np.random.choice(range(size_1 + size_2), size_2))
        stat_bs = compute_statistic(dir_derivs, index_1, index_2)
        if statistic <= stat_bs:
            p_value += 1
        pdf_bs.append(stat_bs)

    p_value = p_value / bootstrap_samples

    if plot_pdf:
        hist = plt.hist(
            pdf_bs, bins="auto", color="lightsteelblue", label="bootstrap statistics"
        )
        plt.vlines(statistic, 0, max(hist[0]), color="salmon", label="test statistic")
        plt.title("bootstrap test, p={}".format(p_value))
        plt.legend()
        plt.show()

    return p_value


def bhv_test_permutation(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    directions: Union[str, ndarray] = "single_split",
    permutation_samples: int = 49998,
    plot_pdf: bool = False,
) -> float:
    size_1 = len(first_sample)
    size_2 = len(second_sample)

    if isinstance(directions, str) and directions == "single_split":
        cosines = directions_single_split(first_sample + second_sample)
    elif isinstance(directions, str) and directions == "from_data":
        cosines = directions_in_data(first_sample + second_sample)
    elif isinstance(directions, str) and directions == "both":
        cosines = directions_in_data(first_sample + second_sample)
        cosines_2 = directions_single_split(first_sample + second_sample)
        cosines = np.hstack((cosines, cosines_2))
    elif isinstance(directions, ndarray) and directions.shape[0] == size_1 + size_2:
        cosines = directions
    else:
        raise Exception("Use supported way of picking directions")

    dir_derivs = (
        -cosines
        * np.array([tree.norm() for tree in (first_sample + second_sample)])[:, None]
    )

    statistic = compute_statistic(
        dir_derivs, list(range(0, size_1)), list(range(size_1, size_1 + size_2))
    )
    p_value = 0
    pdf_bs = []
    for _ in range(permutation_samples):
        indices = random.sample(range(size_1 + size_2), size_1 + size_2)
        index_1 = indices[:size_1]
        index_2 = indices[size_1:]
        stat_bs = compute_statistic(dir_derivs, index_1, index_2)
        if statistic <= stat_bs:
            p_value += 1
        pdf_bs.append(stat_bs)

    p_value = (p_value + 1) / (permutation_samples + 1)

    if plot_pdf:
        hist = plt.hist(
            pdf_bs, bins="auto", color="lightsteelblue", label="permutation statistics"
        )
        plt.vlines(statistic, 0, max(hist[0]), color="salmon", label="test statistic")
        # plt.title("permutation test, p={}".format(p_value))
        plt.legend()
        plt.savefig("permutation_test.pdf", bbox_inches="tight")
        plt.show()

    return p_value


def compute_statistic(
    dir_derivs: ndarray, index_1: list[int], index_2: list[int]
) -> float:
    # size_1 = len(index_1)
    # size_2 = len(index_2)

    derivs_1 = np.mean(dir_derivs[index_1], axis=0)
    derivs_2 = np.mean(dir_derivs[index_2], axis=0)
    statistic = np.max(np.abs(derivs_1 - derivs_2))
    # statistic = ((size_1*size_2)/(size_1 + size_2))**.5 * statistic

    return statistic


def bhv_test_residual(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    directions: Union[str, ndarray] = "single_split",
    permutation_samples: int = 49999,
    plot_pdf: bool = False,
) -> float:
    size_1 = len(first_sample)
    size_2 = len(second_sample)

    if isinstance(directions, str) and directions == "single_split":
        cosines = directions_single_split(first_sample + second_sample)
    elif isinstance(directions, str) and directions == "from_data":
        cosines = directions_in_data(first_sample + second_sample)
    elif isinstance(directions, ndarray) and directions.shape[0] == size_1 + size_2:
        cosines = directions
    else:
        raise Exception("Use supported way of picking directions")

    dir_derivs = (
        -cosines
        * np.array([tree.norm() for tree in (first_sample + second_sample)])[:, None]
    )
    # np.hstack(
    #    (
    #        dir_derivs,
    #        np.array([tree.norm() for tree in (first_sample + second_sample)])[:, None],
    #    )
    # )

    statistic = compute_statistic(
        dir_derivs, list(range(0, size_1)), list(range(size_1, size_1 + size_2))
    )
    residuals = copy(dir_derivs)
    residuals[:size_1] -= np.mean(dir_derivs[:size_1], axis=0)
    residuals[size_1:] -= np.mean(dir_derivs[size_1:], axis=0)
    p_value = 0
    pdf_bs = []
    for _ in range(permutation_samples):
        # indices = random.sample(range(size_1 + size_2), size_1 + size_2)
        # index_1 = indices[:size_1]
        # index_2 = indices[size_1:]
        index_1 = random.choices(range(size_1), k=size_1)
        index_2 = [size_1 + i for i in random.choices(range(size_2), k=size_2)]
        # index_2 = indices[size_1:]
        stat_bs = compute_statistic(residuals, index_1, index_2)
        if statistic <= stat_bs:
            p_value += 1
        pdf_bs.append(stat_bs)

    p_value = p_value / permutation_samples

    if plot_pdf:
        hist = plt.hist(
            pdf_bs, bins="auto", color="lightsteelblue", label="bootstrap statistics"
        )
        plt.vlines(statistic, 0, max(hist[0]), color="salmon", label="test statistic")
        plt.title("residual bootstrap test, p={}".format(p_value))
        plt.legend()
        plt.show()

    return p_value
