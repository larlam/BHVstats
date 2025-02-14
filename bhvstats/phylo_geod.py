"""
This module contains the class PhyloGeod, which represents a geodesic path
between two trees.
"""

import numpy as np
from bhvstats.phylo_simple_geod import PhyloSimpleGeod
from bhvstats.phylo_split import PhyloSplit
from bhvstats.phylo_tree import PhyloTree


class PhyloGeod:

    def __init__(self, tree1: PhyloTree, tree2: PhyloTree):
        if tree1.get_leafcount() != tree2.get_leafcount():
            raise Exception("The number of leaves must be equal.")
        else:
            self.leafcount = tree1.get_leafcount()

        self.tree1 = tree1
        self.tree2 = tree2
        # bisect the tree
        bisected_trees, commonsplits = bisect_at_commons(tree1, tree2)
        self.commonsplits = commonsplits
        # self.subpaths = [PhyloSimpleGeod(t1, t2) for t1,t2 in bisections]
        self.subpaths = [
            PhyloSimpleGeod(t1, t2)
            for t1, t2 in bisected_trees
            if (t1.get_splitcount() + t2.get_splitcount()) > 0
        ]

        # store path length
        dist = 0
        for path in self.subpaths:
            dist += path.path_length() ** 2

        dist += np.sum(np.array(list(self.commonsplits.values())) ** 2)
        self.path_length = dist**0.5

    def get_path_length(self, exterior=False) -> float:
        """
        Returns the length of the path.

        Parameters
        ----------
        exterior : Boolean
            Whether pendant lenghts should be included in the distance.

        Returns
        -------
        float
            Length of the geodesic.
        """
        dist = self.path_length
        if exterior:
            len1 = self.tree1.get_pendant_lengths()
            len2 = self.tree2.get_pendant_lengths()
            leafcont = len2 - len1
            leafcont = np.sum(leafcont**2)
            dist = (dist**2 + leafcont) ** 0.5
        return dist

    def get_subpaths(self) -> list[PhyloSimpleGeod]:
        return self.subpaths

    def get_trees(self):
        return self.tree1, self.tree2

    def get_common(self):
        return self.commonsplits

    def eval(self, t) -> PhyloTree:
        if t == 0:
            g_t = self.tree1
        elif t == 1:
            g_t = self.tree2

        else:
            N = self.leafcount
            g_t = PhyloTree(N)

            # TODO optimize this procedure

            # start with pendants
            l1 = self.tree1.get_pendant_lengths()
            l2 = self.tree2.get_pendant_lengths()
            lt = (1 - t) * l1 + t * l2
            M = np.zeros((N + 1, N + 2))
            M[:, : N + 1] = np.eye(N + 1)
            M[:, N + 1] = lt

            # add splits from subpaths
            for path in self.subpaths:
                M = np.vstack((M, path.eval(t)))

            g_t.from_matrix(M)

            # lastly, add common splits
            for com in self.commonsplits.keys():
                lcom = (1 - t) * self.tree1.splits[com] + t * self.tree2.splits[com]
                g_t.add_split(com, lcom)

        return g_t


def bisect_at_commons(
    tree1: PhyloTree, tree2: PhyloTree
) -> tuple[list[PhyloSimpleGeod], dict[PhyloSplit, float]]:
    """
    Recursively bisects two trees at common splits into pairs of trees with
    fewer leaves.

    Parameters
    ----------
    tree1 : PhyloTree
        The first tree.
    tree2 : PhyloTree
        The second tree.
    """
    commonsplits = {}
    bisections = []

    common = None
    for split in tree1.splits.keys():
        if split in tree2.splits.keys():
            common = split
            break

    if common == None:
        bisections.append((tree1, tree2))

    else:
        st11, st12 = tree1.bisect(common)
        st21, st22 = tree2.bisect(common)
        bs1, cs1 = bisect_at_commons(st11, st21)
        bs2, cs2 = bisect_at_commons(st12, st22)

        bisections = bs1 + bs2
        commonsplits = {**cs1, **cs2}
        dcom = abs((tree1.splits[common] - tree2.splits[common]))
        commonsplits[common] = dcom

    return bisections, commonsplits
