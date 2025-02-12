from bhvstats.phylo_tree import PhyloTree
from bhvstats.phylo_geod import PhyloGeod


def eval_geod(tree1: PhyloTree, tree2: PhyloTree, t: float) -> PhyloTree:
    """
    Evaluates a geodesic between two trees at a given point.

    Parameters
    ----------
    tree1 : PhyloTree
        The starting point of the geodesic.
    tree2 : PhyloTree
        The end point of the geodesic.
    t : float
        The position where the geodeisc is supposed to be evaluated.

    Returns
    -------
    g_t : PhyloTree
        The evaluated point of the geodesic.
    """

    g = PhyloGeod(tree1, tree2)
    g_t = g.eval(t=t)

    return g_t
