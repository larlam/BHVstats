import numpy as np
from numpy import ndarray
from bhvstats.phylo_tree import PhyloTree
from bhvstats.phylo_split import PhyloSplit


def proj_trees(
    mean: PhyloTree, sample: list[PhyloTree], vertex_dict=False
) -> tuple[list[PhyloTree], ...]:
    """
    Projects a list of phylogenetic trees onto the orthogonal component of the
    tangent cone at a given reference tree.


    Parameters
    ----------
    mean : PhyloTree
        The reference tree.
    sample : list[PhyloTree]
        The list of PhyloTrees to be projected.

    Returns
    -------
    sample_proj : list[PhyloTree]
        The projected trees.
    """

    sample_proj = []
    # determine splits which are "interior", i.e. cause the tanget space to be a
    # product of multiple lower-dimensional tree spaces
    surviving = surviving_splits(mean)

    projections = {s: [] for s in surviving}
    projections = {**projections, **{"remaining": []}}

    for tree in sample:
        treeproj = project_tree(mean, tree, surviving)
        # now bisect the trees at the surviving edges
        # and store them in their respective lists
        for split in surviving:
            treeproj, tree_s = treeproj.bisect(split)
            projections[split].append(tree_s)
        # finally, add the remaining tree, too
        projections["remaining"].append(treeproj)

    sample_proj = tuple(projections.values())

    return sample_proj


def surviving_splits(tree: PhyloTree) -> list[PhyloSplit]:
    """
    Determines the splits which are "interior", i.e. cause the tangent space to
    be a product of multiple lower-dimensional tree spaces.

    Parameters
    ----------
    tree : PhyloTree
        The reference tree.

    Returns
    -------
    surviving : list[PhyloSplit]
        The list of splits which are "interior".
    """

    surviving = []
    leafcount = tree.get_leafcount()
    tree_mat = tree.as_matrix()[leafcount + 1 :, :-1]
    projmat = projection_matrix(tree)
    tree_mat_pr = tree_mat @ projmat
    Nproj = projmat.shape[1] - 1
    j = np.where(
        np.abs(np.sum(tree_mat_pr, axis=1) - 2 - (Nproj - 3) / 2)
        <= (Nproj - 3) / 2
    )[0]
    for s in j:
        surviving.append(vector_to_split(tree_mat[s], leafcount))

    return surviving


def project_tree(
    mean: PhyloTree,
    tree: PhyloTree,
    surviving: list[PhyloSplit],
) -> PhyloTree:
    """
    Projects a phylogenetic trees onto the orthogonal component of the
    tangent cone at a given reference tree.

    Parameters
    ----------
    mean : PhyloTree
        The reference tree.
    tree : PhyloTree
        The tree to be projected.
    projmat : ndarray
        The projection matrix for the splits.

    Returns
    -------
    PhyloTree
        The projected tree.
    """

    leafcount = mean.get_leafcount()
    compmat = compatibility_mat(mean, tree)

    comp_splits = np.where(np.mean(compmat, axis=0) == 1)[0].tolist()

    splitmat = tree.as_matrix()[leafcount + 1 :]
    splitmat = splitmat[comp_splits, :]

    treeproj = PhyloTree(mean.get_leafcount())

    for i in range(splitmat.shape[0]):
        s = vector_to_split(splitmat[i, :-1], leafcount)
        if s not in mean.get_splits():
            treeproj.add_split(s, splitmat[i, -1])

    for s in surviving:
        treeproj.add_split(s, 1.0)

    return treeproj


def vector_to_split(
    split: ndarray,
    N: int,
) -> PhyloSplit:
    """
    Converts a vector to a split.

    Parameters
    ----------
    split : ndarray
        The vector representing the split.
    N : int

    Returns
    -------
    PhyloSplit
        The split.
    """
    l = np.where(split > 0)[0].tolist()
    s = PhyloSplit(l, N)
    return s


def projection_matrix(
    mean: PhyloTree,
) -> ndarray:
    """
    Computes the projection matrix for splits onto the orthogonal directions
    at the given tree.


    Parameters
    ----------
    mean : PhyloTree
        The reference tree for the projection.
    N : int
        The initial leafcount.
    Nproj : int
        The target leafcount.
    """
    leafcount = mean.get_leafcount()

    meanmat = mean.as_matrix()[leafcount + 1 :, :-1]
    projmat = np.eye(leafcount + 1)

    leafcount_cur = leafcount
    # find splits to be removed
    v = np.sum(meanmat, axis=1)
    redr = np.where((v == 2) + (v == (leafcount_cur - 1)))[0]
    while len(redr) != 0:
        # for each split to be removed, the first leaf in array is removed
        redc = []
        for i in redr:
            if v[i] == 2:
                j = np.where(meanmat[i] == 1)[0][0]
                redc.append(j)
            else:
                j = np.where(meanmat[i] == 0)[0][-1]
                redc.append(j)

        # probably should delete rows first to avoid errors
        meanmat = np.delete(meanmat, redc, axis=1)
        meanmat = np.delete(meanmat, redr, axis=0)
        projmat = projmat @ np.delete(np.eye(leafcount_cur + 1), redc, axis=1)

        leafcount_cur = meanmat.shape[1] - 1
        # find splits to be removed in next iteration
        v = np.sum(meanmat, axis=1)
        redr = np.where((v == 2) + (v == (leafcount_cur - 1)))[0]

    return projmat


def compatibility_mat(tree_1: PhyloTree, tree_2: PhyloTree) -> ndarray:
    """
    Computes the compatibility matrix for splits.
    """

    leafcount = tree_1.get_leafcount()
    t_mat_1 = tree_1.as_matrix()[leafcount + 1 :, :-1]
    t_mat_2 = tree_2.as_matrix()[leafcount + 1 :, :-1]

    n_add = t_mat_1.shape[0]
    n_rem = t_mat_2.shape[0]

    compatibility = np.zeros((n_add, n_rem))
    t_mat_1 = np.expand_dims(t_mat_1, 1)
    t_mat_2 = np.expand_dims(t_mat_2, 0)
    compatibility = t_mat_1 - t_mat_2
    compatibility = np.all(compatibility >= 0, axis=2) + np.all(
        compatibility <= 0, axis=2
    )
    compatibility += np.all((t_mat_1 + t_mat_2) < 2, axis=2)

    return compatibility
