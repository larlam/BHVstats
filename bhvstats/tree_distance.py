"""
In this module, there are functions to either compute the geodesic distance
between two phylogenetic trees in the BHV-space or to evaluate a geodesic at a
given point.
"""

from bhvstats.phylo_geod import PhyloGeod
from bhvstats.phylo_tree import PhyloTree


def distance(
    tree1: PhyloTree,
    tree2: PhyloTree,
    exterior=False,
) -> float:
    """
    Computes the geodesic distance of two phylogenetic trees in the BHV-space.

    Parameters
    ----------
    tree1 : PhyloTree
        The first tree.
    tree2 : PhyloTree
        The second tree.
    exterior: bool, optional
        Determines whether exterior edges are taken into account. The default is False.

    Returns
    -------
    dist : float
        The geodesic distance.

    """

    geod = PhyloGeod(tree1, tree2)
    dist = geod.get_path_length(exterior=exterior)

    return dist


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
