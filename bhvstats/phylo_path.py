from __future__ import annotations
import numpy as np
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from bhvstats.phylo_block import PhyloBlock


# A class that includes objects in a path-space
# The important part is gtp_extend used for computing the geodesic distance
# of phylogenetic trees


class PhyloPath:
    def __init__(self, blocks: list[PhyloBlock]):
        """
        An object for paths connecting two points in BHV-spaces.
        Used for computing geodesic distances.


        Parameters
        ----------
        l : list[PhyloBlock]
            A list of PhyloBlock objects.
        """
        self.blocks = blocks

    def __len__(self) -> int:
        """
        Returns the number of blocks, not the length of the path in the
        BHV-space.


        Returns
        -------
        length : int
            Number of blocks.
        """
        length = len(self.blocks)
        return length

    def __getitem__(self, key) -> PhyloBlock:

        if isinstance(key, int):
            val = self.blocks[key]
        # elif isinstance(key, slice):
        # start, stop, step = key.indices(len(self))
        # val = [self[i] for i in range(start, stop, step)]
        # val = phylo_path(val)
        else:
            raise Exception("Invalid argument type: {}".format(type(key)))
        return val

    def get_path(self) -> list[PhyloBlock]:
        """
        Returns the list of blocks.

        Returns
        -------
        list[PhyloBlock]
            List of the path's blocks.
        """
        return self.blocks

    def path_length(self) -> float:
        """
        Returns the length of the path.


        Returns
        -------
        d : float
            Length of the path.
        """
        dist = 0.0
        for block in self.blocks:
            block_a, block_b = block.get_block()
            dist_i = np.linalg.norm(block_a[:, -1])
            dist_i += np.linalg.norm(block_b[:, -1])
            dist += dist_i**2

        dist = dist**0.5
        return dist

    def gtp_extend(
        self,
        pos: int,
    ):
        """
        The extension problem is solved for a given block. If it has a valid
        solution, the block is split.


        Parameters
        ----------
        pos : int
            Position of the block that is to be extended.
        """

        block_old = self.blocks[pos]
        # if the block is already non-extendable -> do nothing
        if block_old.extendable == 0:
            return
        compatibility = block_old.compatibility

        block_rem, block_add = block_old.get_block()
        if block_rem.shape[0] == 0 or block_add.shape[0] == 0:
            self.blocks[pos].extendable = 0
            return

        n_rem = len(block_rem)
        n_add = len(block_add)

        # compute the total squared weight of C and D
        weight_rem = np.linalg.norm(block_rem[:, -1])
        weight_add = np.linalg.norm(block_add[:, -1])

        # solve the max flow problem
        graph = nx.DiGraph()

        # source arcs
        for i in range(n_rem):
            graph.add_edge(
                "S", str(i), capacity=(block_rem[i, -1] / weight_rem) ** 2
            )
        # add edge if two splits are incompatible
        incomp_rem, incomp_add = np.where(compatibility == 0)
        for i in range(len(incomp_rem)):
            graph.add_edge(
                str(incomp_rem[i]), str(n_rem + incomp_add[i]), capacity=1
            )
        # drain arcs
        for j in range(n_add):
            graph.add_edge(
                str(n_rem + j),
                "T",
                capacity=(block_add[j, -1] / weight_add) ** 2,
            )

        # the default flow_func (preflow_push) crashes occasionally
        # didn't oberserve this behaviour with shortest_augmenting_path

        weight, cut = nx.minimum_cut(
            graph, "S", "T", flow_func=shortest_augmenting_path
        )
        cut = list(cut[0])

        # compute the covering
        # compute the new partitions
        rem_1, rem_2, add_1, add_2 = [], [], [], []
        for i in range(n_rem):
            if str(i) in cut:
                rem_2.append(i)
            else:
                rem_1.append(i)
        for j in range(n_add):
            if str(j + n_rem) in cut:
                add_2.append(j)
            else:
                add_1.append(j)

        # if there is a solution to the extension problem, split the block
        # it seems like there are some rounding errors,
        # so we try to accomodate for that
        if weight < 1 - 1e-7 and (add_2 or rem_2):
            bnew1 = PhyloBlock(
                block_rem[rem_1],
                block_add[add_1],
                compatibility=compatibility[rem_1][:, add_1],
            )
            bnew2 = PhyloBlock(
                block_rem[rem_2],
                block_add[add_2],
                compatibility=compatibility[rem_2][:, add_2],
            )

            self.blocks[pos] = bnew1
            self.blocks.insert(pos + 1, bnew2)

        else:
            self.blocks[pos].extendable = 0
