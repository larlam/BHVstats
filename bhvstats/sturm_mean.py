import random
from bhvstats.tree_distance import distance
from bhvstats.eval_geod import eval_geod
from bhvstats.phylo_tree import PhyloTree


def sturm_mean(sample: list[PhyloTree]) -> PhyloTree:
    """
    Approximates the Frechet mean of a set of phylogenetic trees via Sturm's
    algorithm.

    Parameters
    ----------
    sample : list[PhyloTree]
        A list of phylogenetic trees.

    Returns
    -------
    cur_mean : PhyloTree
        An approximation of the Sturm mean.
    """
    leaves = sample[0].get_leafcount()
    size = len(sample)

    # for two points, there is no need to use Sturms algorithm
    if size == 2:
        cur_mean = eval_geod(sample[0], sample[1], 0.5)
        return cur_mean

    i = 0
    cur_mean = PhyloTree(leaves)
    dist = 1
    # TODO stopping criterion
    ind = 0
    while ind < 5:
        prev_mean = cur_mean
        j = random.randint(0, size - 1)
        treej = sample[j]
        cur_mean = eval_geod(prev_mean, treej, 1 / (i + 2))
        dist = distance(prev_mean, cur_mean)
        if dist < 1e-4:
            ind += 1
        else:
            ind = 0
        i += 1
    print("Iterations: " + str(i))
    print("Epsilon: " + str(dist))
    return cur_mean


def frechet_function(tree: PhyloTree, sample: list[PhyloTree]) -> float:
    """
    Computes the Frechet function at a point for a given set of trees.

    ----------
    tree : PhyloTree
        The point at which the function is to be evaluated.
    sample : list[PhyloTree]
        A list of phylogenetic trees.

    Returns
    -------
    frech : float
        The value of the Frechet function.
    """
    size = len(sample)
    frech = 0.0
    for t in sample:
        frech += distance(tree, t) ** 2
    frech = frech / size
    return frech
