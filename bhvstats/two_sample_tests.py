from typing import Optional, Union
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from numpy import ndarray
from bhvstats.t2_test import twosamplet2test_unequal, paired_t2_test
from bhvstats.t_test import onesample_t_test
from bhvstats.mult_hyp import (
    holm_correction,
    simes_correction,
    bonferroni_correction,
)
from bhvstats.tree_distance import distance
from bhvstats.phylo_tree import PhyloTree


# TODO Update the documentation for all functions..

# TODO rename to bhv_test_hs for Hotteling-Simes


def two_sample_test_sticky(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    directions: Union[str, ndarray] = "single_split",
    permutation_samples: int = 49998,
    plot_pdf: bool = False,
) -> float:
    """
    Peforms a Kolmogorov-Smirnov test over a finite selection of the
    directional derivatives of the Frechet function. The quantiles can be
    either determined through permutations.

    Parameters
    ----------
    first_sample : list[PhyloTree]
        First sample of phylogenetic trees.
    second_sample : list[PhyloTree]
        Second sample of phylogenetic trees.
    directions : {"single_splits", "from_data", ndarray}
        Determines what directions are considered during testing. If an ndarray
        is given, it is assumed that the directional derivatives are already
        computed.
    bootstrap_samples
        The number of iterations for the bootstrap.
    plot_pdf : Boolean, optional
        If set to True, a histogram of all bootstrap/permutation statistics is
        plotted. Default is False.

    Return
    -------
    p_value : float
        The p-value for the given data.

    """
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


def two_sample_test_sticky_bootstrap(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    directions: Union[str, ndarray] = "single_split",
    bootstrap_samples: int = 49998,
    plot_pdf: bool = False,
) -> float:
    """
    Peforms a Kolmogorov-Smirnov test over a finite selection of the
    directional derivatives of the Frechet function. The quantiles can be
    either determined through bootsrapping.

    Parameters
    ----------
    first_sample : list[PhyloTree]
        First sample of phylogenetic trees.
    second_sample : list[PhyloTree]
        Second sample of phylogenetic trees.
    directions : {"single_splits", "from_data", ndarray}
        Determines what directions are considered during testing. If an ndarray
        is given, it is assumed that the directional derivatives are already
        computed.
    bootstrap_samples
        The number of iterations for the bootstrap.
    plot_pdf : Boolean, optional
        If set to True, a histogram of all bootstrap/permutation statistics is
        plotted. Default is False.

    Return
    -------
    p_value : float
        The p-value for the given data.

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


def two_sample_test_sticky_hotelling(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    ratio=0.5,
    directions: Union[str, ndarray] = "single_split",
    correction="simes",
    plot_pvalues: bool = False,
) -> float:
    """
    Calculates a two-sample t-squared test for phylogenetic tree data with
    respect to the degrees of stickiness in randomized directions at the cone
    point.
    The Reference directions are randomly selected from the data itself.
        It is assumed that the samples are independet.

    Parameters
    ----------
    first_sample : list[PhyloTree]
        First sample of phylogenetic trees.
    second_sample : list[PhyloTree]
        Second sample of phylogenetic trees.
    ratio : float, optional
        The ratio of random selection of directions
        to the total amount of directions in the data. Default is 0.5.
    directions : {"single_splits", "from_data"}
        Determines what directions are considered during testing.
    correction : {"simes", "holm"}, optional
        The correction method for multiple hypothesis testing.
    plot_pvalues : Boolean, optional
        If set to True, a histogram of all individual p-values is plotted.
        Default is False.

    Return
    -------
    p_value : float
        The p-value for the given data.

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
                "The dimension of the given matrix do not match \
                             the sample size."
            )
    else:
        raise TypeError("Use supported way of picking directions")

    ndir = cosines.shape[1]
    iterations = 100 * int(np.log(ndir) * ndir)

    tree_norms = np.array([tree.norm() for tree in (first_sample + second_sample)])[
        :, None
    ]
    dir_derivs = -cosines * tree_norms

    p_vals = directions_test(size_1, size_2, dir_derivs, iterations, ratio)

    if plot_pvalues:
        print(p_vals)
        plt.hist(p_vals, bins="auto")
        plt.show()

    # multiplehypothesis correction
    if correction == "simes":
        p_value = simes_correction(p_vals)
    elif correction == "holm":
        p_value = holm_correction(p_vals)
    elif correction == "bonferroni":
        p_value = bonferroni_correction(p_vals)
    else:
        raise Exception(
            "Please use a supported method for multiple \
                        hypothesis correction"
        )
    return p_value


def paired_test_sticky_bhv(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    ratio=0.5,
    directions="single_split",
    correction="simes",
    plot_pvalues: bool = False,
) -> float:
    """
    Calculates a paired t-squared test for phylogenetic tree data with
    respect to the degrees of stickiness in randomized directions at the cone
    point.
    The Reference directions are randomly selected from the data itself.
    It is assumed that the samples are independet.

    Parameters
    ----------
    first_sample : list[PhyloTree]
        First sample of phylogenetic trees.
    second_sample : list[PhyloTree]
        Second sample of phylogenetic trees.
    ratio : float, optional
        The ratio of random selection of directions
        to the total amount of directions in the data. Default is 0.5.
    directions : {"single_splits", "from_data"}
        Determines what directions are considered during testing.
    correction : {"simes", "holm"}, optional
        The correction method for multiple hypothesis testing.
    plot_pvalues : Boolean, optional
        If set to True, a histogram of all individual p-values is plotted.
        Default is False.

    Return
    -------
    p_value : float
        The p-value for the given data.

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

    ndir = cosines.shape[1]
    iterations = 1000 * int(np.log(ndir) * ndir)

    tree_norms = np.array([tree.norm() for tree in (first_sample + second_sample)])[
        :, None
    ]
    dir_derivs = -cosines * tree_norms

    p_vals = directions_test(size_1, size_2, dir_derivs, iterations, ratio, paired=True)

    if plot_pvalues:
        print(p_vals)
        plt.hist(p_vals, bins="auto")
        plt.show()

    # multiplehypothesis correction
    if correction == "simes":
        p_value = simes_correction(p_vals)
    elif correction == "holm":
        p_value = holm_correction(p_vals)
    else:
        raise Exception(
            "Please use a supported method for multiple \
                        hypothesis correction"
        )
    return p_value


def paired_test_sticky_univ(
    first_sample: list[PhyloTree],
    second_sample: list[PhyloTree],
    directions="single_split",
    correction="simes",
    plot_pvalues: Optional[bool] = False,
) -> float:
    """
    Performs a one sample test for stickiness in given directions of BHV-space.

    Parameters
    ----------
    sample
        The sticky sample.
    directions : {"single_splits", "from_data"}
        Determines what directions are considered during testing.
    correction : {"simes", "holm"}, optional
        The correction method for multiple hypothesis testing.
    plot_pvalues : Boolean, optional
        If set to True, a histogram of all individual p-values is plotted.
        Default is False.


    Returns
    -------
    p_value : float
        The p-value for the given data.
    """

    if directions == "single_split":
        cosines = directions_single_split(first_sample + second_sample)
    elif directions == "from_data":
        cosines = directions_in_data(first_sample + second_sample)
    else:
        raise Exception("Use supported way of picking directions")

    size = len(first_sample)
    derivatives = np.zeros(cosines.shape)
    for i in range(size):
        t_norm_1 = first_sample[i].norm()
        t_norm_2 = second_sample[i].norm()
        derivatives[i] = -t_norm_1 * cosines[i, :]
        derivatives[size + i] = -t_norm_2 * cosines[size + i, :]

    derivatives = derivatives[:size] - derivatives[size:]
    p_vals = onesample_t_test(derivatives, side="both")

    if plot_pvalues:
        print(p_vals)
        plt.hist(p_vals, bins="auto")
        plt.show()

    # multiplehypothesis correction
    if correction == "simes":
        p_value = simes_correction(p_vals)
    elif correction == "holm":
        p_value = holm_correction(p_vals)
    elif correction == None:
        p_value = list(p_vals)
    else:
        raise Exception(
            "Please use a supported method for multiple \
                        hypothesis correction"
        )

    return p_value


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


def directions_test(
    size_1: int,
    size_2: int,
    dir_derivs: ndarray,
    iterations: int,
    ratio: float,
    paired: Optional[bool] = False,
) -> ndarray:
    """
    For a given set of directions, perform multiple T2-tests with respect to
    the degrees of stickiness by randomly selecting a subset of directions.

    Parameters
    ----------
    first_sample : list[PhyloTree]
        First sample,
    second_sample : list[PhyloTree]
        Second sample.
    cosines : ndarray
        The cosines for the given directions.
    iterations : int
        Number of iterations.
    ratio : float
        The ratio of random selection of directions
        to the total amount of directions in the data.

    Returns
    -------
    p-vals : list[float]
        The list of p-values.
    """

    n_dir = dir_derivs.shape[1]
    if paired:
        n_select = int(ratio * (size_1))
    else:
        n_select = int(ratio * (size_1 + size_2))
    p_vals = []
    for _ in range(iterations):
        # random selection of samples
        randtop = random.sample(range(n_dir), n_select)

        # compute the folded data
        first_sample_folded = dir_derivs[:size_1, randtop]
        second_sample_folded = dir_derivs[size_1:, randtop]

        # sometimes numerical errors cause the covariance to be singular
        # in these cases, the iteration is just skipped
        try:
            if paired:
                p_vals.append(paired_t2_test(first_sample_folded, second_sample_folded))
            else:
                p_vals.append(
                    twosamplet2test_unequal(first_sample_folded, second_sample_folded)
                )
            pass
        except:
            continue

    p_vals = np.array(p_vals)
    return p_vals


def directions_in_data(sample: list[PhyloTree]) -> ndarray:
    """
    Fetches possible directions for testing directly from the data.

    Parameters
    ----------
    sample : list[PhyloTree]
        The data.

    Returns
    -------
    cosines : ndarray
        The cosines of the angles between the data and reasonable directions.
    """
    # first step: compute matrix of pairwise cosines of angles
    cos_pairwise = comp_theta(sample)

    possdir = list(range(len(sample)))
    # remove redundant directions with only one
    # split to avoid singular covariance matrices
    nsplits = np.array([t.get_splitcount() for t in sample])
    snglsplt = list(np.where(nsplits == 1)[0])
    tbrem = []
    for j in snglsplt:
        spl = list(sample[j].get_splits())[0]
        if spl in tbrem:
            possdir.remove(j)
        else:
            tbrem.append(spl)

    # also, we need to remove all star trees
    # as possible directions from dataset
    red_dir = list(np.where(np.diag(cos_pairwise) == 0)[0])
    possdir = [e for e in possdir if e not in red_dir]
    # TODO make it nice
    cosines = cos_pairwise.T[possdir].T

    return cosines


def directions_single_split(sample: list[PhyloTree]) -> ndarray:
    """
    Fetches possible directions for testing directly by considering Trees with
    single splits present in the data.

    Parameters
    ----------
    sample : list[PhyloTree]
        The data.

    Returns
    -------
    cosines : ndarray
        The cosines of the angles between the data and reasonable directions.
    """
    # make a list of splits in the data and in each step, add a tree w/ that
    # split of length 1
    leafcount = sample[0].get_leafcount()
    splits = []
    poss_dir = []
    for tree in sample:
        for split in tree.get_splits():
            if (split not in splits) and (tree.splits[split] > 0):
                splits.append(split)
                direction = PhyloTree(leafcount)
                direction.add_split(split, 1)
                poss_dir.append(direction)
    cosines = comp_theta(sample, poss_dir)

    # need to remove directions if one of the sample is partly sticky in these
    # directions
    return cosines


def cos_tree(tree_1: PhyloTree, tree_2: PhyloTree) -> float:
    """
    Computes the cosine of the Alexandrov angles of two phylogenetic trees at
    the cone point.


    Parameters
    ----------
    tree_1 : PhyloTree
        First tree.
    tree_2 : PhyloTree
        Second tree.

    Returns
    -------
    cos_alex : float
        The cosine of the Alexandrov angle.
    """
    norm1 = tree_1.norm()
    norm2 = tree_2.norm()
    # if one of the norms is 0, aka the tree is the star tree,
    # set the cosine to 0 as we multiply it with 0 later anyway
    if norm1 == 0 or norm2 == 0:
        cos_alex = 0
    # use law of cosines to get the derivative
    else:
        dist = distance(tree_1, tree_2)
        cos_alex = (norm1**2 + norm2**2 - dist**2) / (2 * norm1 * norm2)

    return cos_alex


def comp_theta(
    sample: list[PhyloTree],
    sample_2: Optional[list[PhyloTree]] = None,
) -> ndarray:
    """
    Computes a matrix of the pairwise cosines of the Alexandrov angles at the
    cone point for given lists of pyhlogenetic trees.

    Parameters
    ----------
    sample : list[PhyloTree]
        A list of phylogenetic trees.
    sample_2 : list[PhyloTree]
        A second list of phylogenetic trees. If none is given, it defaults to
        the first sample. Default is None.

    Returns
    -------
    cosines : ndarray
        A matrix for the pairwise cosines.
    """
    size_1 = len(sample)
    # if None, then the matrix is symmetric and we need to do less
    if sample_2 is None:
        cosines = np.zeros((size_1, size_1))
        for i in range(size_1):
            tree_1 = sample[i]
            for j in range(i, size_1):
                tree_2 = sample[j]
                cos_alex = cos_tree(tree_1, tree_2)
                cosines[i, j] = cos_alex
                cosines[j, i] = cos_alex

    else:
        size_2 = len(sample_2)
        cosines = np.zeros((size_1, size_2))
        for i in range(size_1):
            tree_1 = sample[i]
            for j in range(size_2):
                tree_2 = sample_2[j]
                cos_alex = cos_tree(tree_1, tree_2)
                cosines[i, j] = cos_alex

    return cosines


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
