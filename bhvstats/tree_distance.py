from bhvstats.phylo_geod import PhyloGeod
from bhvstats.phylo_tree import PhyloTree


def tree_distance(
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
