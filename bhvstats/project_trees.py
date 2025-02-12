import numpy as np
from numpy import ndarray
from bhvstats.tree_distance import tree_distance
from bhvstats.phylo_tree import PhyloTree
from bhvstats.phylo_split import PhyloSplit
from bhvstats.phylo_geod import PhyloGeod


# TODO Should rewrite it completely, is unnecessarily complicated


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
    # we start by removing the splits and meanwhile constructing a
    # projection matrix to apply to the splits later
    projmat = projection_matrix(mean)

    surviving = surviving_splits(mean, projmat)

    projections = {s: [] for s in surviving}
    projections = {**projections, **{"remaining": []}}

    for tree in sample:
        treeproj = project_tree(mean, tree, projmat, surviving)
        # now bisect the trees at the surviving edges
        # and store them in their respective lists
        for split in surviving:
            treeproj, tree_s = treeproj.bisect(split)
            projections[split].append(tree_s)
        # finally, add the remaining tree, too
        projections["remaining"].append(treeproj)

    sample_proj = tuple(projections.values())

    if vertex_dict:
        return sample_proj, projmat
    else:
        return sample_proj


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


# TODO optional typing hints
def project_tree(
    mean: PhyloTree,
    tree: PhyloTree,
    projmat: ndarray,
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

    N = mean.get_leafcount()
    Nproj = projmat.shape[1] - 1
    d = tree_distance(mean, tree)

    directions, d_spine = get_dir(mean, tree)
    treeproj = PhyloTree(Nproj)
    dirsplits = list(directions.keys())
    treematproj = np.zeros((len(dirsplits), Nproj + 2))
    for i in range(len(dirsplits)):
        split = dirsplits[i]
        arr = split_to_vector(split, N)
        v = arr[:-1][np.newaxis]
        treematproj[i, :-1] = (v @ projmat)[0]
        treematproj[i, -1] = directions[split]

    # rescale the split lengths by the distance between the two trees
    # norm = (np.sum(treematproj[:, -1] ** 2) + d_spine) ** 0.5
    # treematproj[:, -1] = d * treematproj[:, -1] / norm

    # delete all 0 or 1 rows
    spl_sums = np.sum(treematproj[:, :-1], axis=1)
    treematproj = treematproj[
        np.where(np.abs(spl_sums - 1 - Nproj / 2) <= Nproj / 2)
    ]

    treeproj.from_matrix(treematproj)
    for s in surviving:
        treeproj.add_split(s, 1.0)

    return treeproj


def get_dir(
    tree1: PhyloTree, tree2: PhyloTree
) -> tuple[dict[PhyloSplit, float], float]:
    """
    Computes the initial step of the geodesic path between the trees.

    Parameters
    ----------
    tree1 : PhyloTree
        The first tree.
    tree2 : PhyloTree
        The second tree.

    Returns
    -------
    tuple[dict[PhyloSplit, float]]:
        A dictionary with the respective splits and their lengths and
        the squared norm of the removed splits.
    """

    added_splits = {}
    d_rem = 0.0

    g = PhyloGeod(tree1, tree2)

    # this is a quick fix, should rewrite it completely
    for path in g.get_subpaths():
        b0 = path.get_path()[0]
        _, block_add = b0.get_block()
        for i in range(block_add.shape[0]):
            added_splits[
                vector_to_split(block_add[i, :-1], tree1.get_leafcount())
            ] = block_add[i, -1]

    for split in tree1.get_splits():
        if split in g.get_common():
            d_rem += g.commonsplits[split] ** 2
        else:
            d_rem += tree1.splits[split] ** 2

    return added_splits, d_rem


def surviving_splits(tree: PhyloTree, projmat: ndarray) -> list[PhyloSplit]:
    surviving = []
    N = tree.get_leafcount()
    M = tree.as_matrix()[N + 1 :, :-1]
    M = M @ projmat
    # sort M by the position of the split
    # lower sum -> further down the tree
    # can remove them first when bisecting
    M = M[np.argsort(M.sum(axis=1)), :]
    Nproj = projmat.shape[1] - 1
    j = np.where(
        np.abs(np.sum(M, axis=1) - 2 - (Nproj - 3) / 2) <= (Nproj - 3) / 2
    )[0]
    for s in j:
        surviving.append(vector_to_split(M[s], Nproj))

    return surviving


# TODO incluce these as methods for splits?
def split_to_vector(
    split: PhyloSplit,
    N: int,
) -> ndarray:
    """
    Converts a split to its binary vector representation.

    Parameters
    ----------
    split : PhyloSplit
        The split to be converted.
    N : int
        The number of leaves.

    Returns
    -------
    ndarray
        The converted split.
    """
    arr = np.zeros(N + 2)
    _, s = split.get_split()
    arr[s] = np.ones(len(s))

    return arr


def vector_to_split(
    split: ndarray,
    N: int,
) -> PhyloSplit:
    l = np.where(split > 0)[0].tolist()
    s = PhyloSplit(l, N)
    return s
