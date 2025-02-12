import numpy as np
from bhvstats.phylo_path import PhyloPath
from bhvstats.phylo_block import PhyloBlock
from bhvstats.phylo_tree import PhyloTree


class PhyloSimpleGeod(PhyloPath):

    def __init__(self, tree1: PhyloTree, tree2: PhyloTree):
        """
        A geodesic path in a BHV space between trees with no common splits.


        Parameters
        ----------
        T1 : PhyloTree
            Starting tree.
        T2 : PhyloTree
            Target tree.
        """
        self.tree1 = tree1
        self.tree2 = tree2
        self.leafcount = tree1.get_leafcount()

        splits1 = tree1.as_matrix()[self.leafcount + 1 :]
        splits2 = tree2.as_matrix()[self.leafcount + 1 :]
        B = PhyloBlock(splits1, splits2)
        super().__init__([B])

        # On intialization, the path is extended until it is a geodesic.
        # update the path until the amount of blocks stops growing
        ind = len(self.blocks) - 1
        while ind != len(self.blocks):
            ind = len(self.blocks)
            for i in range(ind):
                self.gtp_extend(i)

    def eval(self, t: float) -> np.ndarray:
        """
        Evaluates the geodesic at a given relative point.

        Parameters
        ----------
        t : float
            The relative time point.

        Returns
        -------
        PhyloTree
            The tree at the given time point.

        """

        if t == 0:
            g_t = self.tree1

        elif t == 1:
            g_t = self.tree2

        elif 0 < t < 1:
            g_t = np.empty((0, self.leafcount + 2))
            for block in self.get_path():
                g_t = np.vstack((g_t, block.eval(t)))
        else:
            raise Exception("Please pick a value between 0 and 1.")

        return g_t
