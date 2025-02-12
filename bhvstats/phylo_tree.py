# Classes for phylogenetic trees and their splits
# to use for testing in BHV space
from __future__ import annotations
import numpy as np
from numpy import ndarray
from bhvstats.phylo_split import PhyloSplit
from bhvstats.forest import Forest


class PhyloTree:
    def __init__(self, leafcount: int):
        """
        A class for phylogenetic trees.

        Parameters
        ----------
        leafcount : int
            The number of leaves.
        """

        self.num_leaves = leafcount
        self.splits = {}
        self.num_splits = 0
        self.pendant_lenghts = np.zeros(leafcount + 1)

    def __eq__(self, other):
        if isinstance(other, PhyloTree):
            eq_splits = self.splits == other.splits
            eq_leaves = np.array_equal(
                self.pendant_lenghts, other.pendant_lenghts
            )
            return eq_splits and eq_leaves
        return False

    def add_split(self, partition: list[int] | PhyloSplit, weight: float):
        """
        Adds a split to the tree.


        Notes
        -----
        If the partition corresponds to a pendant edge, the respective pendant
        length is added instead.

        Parameters
        ----------
        partition : list[int]
            List with leafs of one of the two subsets of the split.
        weight : float
            Length of the edge.

        """
        leafcount = self.num_leaves

        if weight < 0:
            raise Exception("Lengths must not be negative.")

        is_list = isinstance(partition, list)
        is_split = isinstance(partition, PhyloSplit)
        if is_list or is_split:
            # check if it corresponds to an interior or pendandt edge
            # if so, add the pendant length

            if is_list and len(partition) == 1:
                i = partition[0]
                self.pendant_lenghts[i] = weight

            elif is_list and len(partition) == leafcount:
                # find integer missing in partition
                i = [j for j in range(leafcount + 1) if j not in partition]
                i = i[0]
                self.pendant_lenghts[i] = weight

            else:
                if weight == 0:
                    # print("The split was not added as its length is 0.")
                    return
                if is_list:
                    new_split = PhyloSplit(partition, leafcount)
                else:
                    new_split = partition
                # check if the split is compatible with the existing ones
                for split in self.splits.keys():
                    if new_split.is_compatible(split) == False:
                        print(
                            "The split was not added as it is not compatible"
                            + "with the existing ones."
                        )
                        return

                self.splits[new_split] = weight
                self.num_splits = len(self.get_splits())
                return

    def add_pendant_lengths(self, pendants: list[int], weights: list[float]):
        """
        Adds pendant lengths.

        Parameters
        ----------
        positions : list[int]
            List of the pendants.

        weight : list[float]
            Lengths of the respective pendant edges.

        """
        npend = len(pendants)
        if npend != len(weights):
            raise Exception("Amount of pendants and lenghts do not match.")
        for i in range(npend):
            pos = pendants[i]
            weight = weights[i]
            if weight < 0:
                raise Exception("Lengths must not be negative.")
            self.pendant_lenghts[pos] = weight

    def from_matrix(self, matrix: ndarray):
        """
        Adds splits from a matrix-representation of a phylogenetic tree.

        Parameters
        ----------
        matrix : array
            The splits in matrix representation.
        """
        mat = matrix[:, :-1]
        lengths = matrix[:, -1]

        nsplits, leafcount = mat.shape
        if leafcount != self.num_leaves + 1:
            raise Exception("The number of leaves do not coincide.")
        if nsplits != len(lengths):
            raise Exception(
                "The amount of splits and the number of \
                                lenghts must be equal."
            )

        for i in range(nsplits):
            arr = mat[i]
            split_i = np.where(arr == 1)[0].tolist()
            self.add_split(split_i, lengths[i])

    def as_matrix(self) -> ndarray:
        """
        Returns the tree as matrix.


        Returns
        -------
        matrix : ndarray
            The tree as matrix.
        """
        # start with root + leafs
        leafcount = self.get_leafcount()
        matrix = np.zeros((leafcount + 1, leafcount + 2))
        for i in range(0, leafcount + 1):
            matrix[i, i] = 1
            matrix[i, leafcount + 1] = self.pendant_lenghts[i]

        # now, add all splits as additional rows
        for split in self.get_splits():
            row_s = np.zeros(leafcount + 2)
            _, partition = split.get_split()
            for leaf in partition:
                row_s[leaf] = 1
            row_s[leafcount + 1] = self.splits[split]
            matrix = np.vstack([matrix, row_s])

        return matrix

    # TODO import dict_keys type
    def get_splits(self):
        """
        Returns the splits of the tree.

        Returns
        -------
        dict_keys
            Splits in the tree.

        """
        return list(self.splits.keys())

    def get_leafcount(self) -> int:
        """
        Returns the amount of leafs.

        Returns
        -------
        int
            Number of leafs

        """

        return self.num_leaves

    def get_splitcount(self) -> int:
        """
        Returns the  amount of splits.

        Returns
        -------
        int
            Number of splits.
        """
        return self.num_splits

    def get_pendant_lengths(self) -> ndarray:
        """
        Returns the pendant lengths.

        Returns
        -------
        ndarray
            Pendant lenghts.
        """

        return self.pendant_lenghts

    def norm(self) -> float:
        """
        Returns the distance to the origin ("norm") of the tree.

        Returns
        -------
        float
            Distance to the origin.

        """

        return np.linalg.norm(list(self.splits.values()))

    def normalize(self):
        """
        Normalizes the edge lengths to have a distance of 1 to the origin.
        """

        c = self.norm()
        for s in self.get_splits():
            self.splits[s] = self.splits[s] / c

    def bisect(
        self,
        split: PhyloSplit,
    ) -> tuple[PhyloTree, PhyloTree]:
        """
        Bisects the tree at a given split. The "midpoint" is then added as a
        new leaf in the resulting two trees.

        Parameters
        ----------
        split : PhyloSplit
            The split to be removed.

        Returns
        -------
        tuple
            Tuple of the two PhyloTree-objects resulting from the split.

        """

        # create two new (empty) trees
        leafcount = self.num_leaves
        tree1 = PhyloTree(leafcount)
        tree2 = PhyloTree(leafcount)

        lst1 = []
        lst2 = []
        # check for all other splits, whether the first subset is a subset of
        # bs1 and store them in a list
        l_splits = self.get_splits()
        l_splits.remove(split)
        for s in l_splits:
            if s < split:
                lst2.append(s)
            else:
                lst1.append(s)

        # add the splits to the trees
        for s in lst1:
            tree1.add_split(s, self.splits[s])

        for s in lst2:
            tree2.add_split(s, self.splits[s])

        return tree1, tree2

    def as_graph(self) -> Forest:
        """
        Converts the tree to a graph.

        Returns
        -------
        Forest
            The phylogenetic tree in graph representation.

        """
        graph = Forest()
        cur_node = self.num_leaves + 1
        graph.add_node(str(cur_node + 1))

        # start by creating the star tree
        for i in range(cur_node):
            graph.add_node(str(i))
            graph.add_edge(
                str(i), str(cur_node), weight=self.pendant_lenghts[i]
            )
        cur_node += 1

        # next, we need a list of all splits
        # in particular, we want the partition w/o the root
        splt_lst = []
        splt_szs = []
        splt_lns = []
        for split in self.get_splits():
            # by design, the second part is the one w/o the root
            _, partition = split.get_split()
            splt_lst.append(partition)
            splt_szs.append(len(partition))
            splt_lns.append(self.splits[split])

        while len(splt_lst) > 0:
            # take partition with the smallest size
            i = splt_szs.index(max(splt_szs))
            s = splt_lst[i]

            # add new vertex
            # delete these edges
            # afterwards, connect the vertices to the new vertex &
            # the new to the old vertex
            vnew = str(cur_node)
            graph.add_node(vnew)

            # find the common vertex all vertices are connected to since there
            # are no loops, suff. to check only for two nodes

            neigh1 = list(graph.neighbors(str(s[0])))
            neigh2 = list(graph.neighbors(str(s[0])))

            for w in neigh2:
                if w in neigh1:
                    vcom = w

            # next replace the edges to vcom with edges to N
            for j in range(len(s)):
                v = str(s[j])
                length = graph[v][vcom]["weight"]
                graph.remove_edge(v, vcom)
                graph.add_edge(v, vnew, weight=length)

            # connect the new to the old vertex
            graph.add_edge(vcom, vnew, splt_lns[i])

            # remove the split from our lists
            splt_lst.pop(i)
            splt_szs.pop(i)
            splt_lns.pop(i)
            cur_node += 1
        return graph

    def from_newick(self, s: str, leaves: dict[str, int], delimiter=","):
        """
        Imports the information about the tree from a representation in Newick
        format.

        Parameters
        ----------
        s : str
            The newick representation.
        leaves : dict[str, int]
            A dictionary for the leaves.
        delimiter :
            The delimiter used for the Newick format.
        """
        from stickytests.bhv.new2phylo import new2phylo

        tree = new2phylo(s, leaves, delimiter)
        mat = tree.as_matrix()
        self.from_matrix(mat)
