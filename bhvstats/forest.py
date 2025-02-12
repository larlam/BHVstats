import networkx as nx
import numpy as np
from networkx import Graph
from copy import copy


class Forest(Graph):

    def __init__(self):
        super().__init__()

    def add_edge(self, frm, to, weight=0.0):
        """
        Add an edge to the forest.

        Parameters
        ----------
        frm : str
            First node.
        to : str
            Second node.
        cost : float
            Weight of the edge.
        """

        if frm not in self.nodes:
            self.add_node(frm)
        if to not in self.nodes:
            self.add_node(to)

        if to in nx.node_connected_component(self, frm):
            print(
                "Edge wasn't added as it'd create a loop: "
                + str(frm)
                + " -> "
                + str(to)
            )
            return None
        return super().add_edge(frm, to, weight=weight)

    def compute_split(self, frm, to):
        """
        Computes the split for a given edge by how it partitions the leaves.

        Parameters
        ----------
        frm : str
            First node.
        to : str
            Second node.

        Returns
        -------
        split : tuple
            The split.
        """
        # remove edge temporarily and get the connected vertices

        # copying the graph won't work as the dictionary vert_dict
        # still uses the same Vertex objects as keys
        length = self[frm][to]["weight"]

        self.remove_edge(frm, to)

        partition1 = nx.node_connected_component(self, frm)
        partition2 = nx.node_connected_component(self, to)

        # add the edge back
        self.add_edge(frm, to, weight=length)

        partition1 = [vert for vert in partition1 if self.degree[vert] == 1]
        partition2 = [vert for vert in partition2 if self.degree[vert] == 1]
        split = (partition1, partition2)
        return split

    def trim(self, vert):
        """
        Removes a vertex from the graph.

        Parameters
        ----------
        vert : str
            The vertex to be removed.
        """
        if vert not in self.nodes:
            raise ValueError("Vertex not in graph.")

        elif self.degree[vert] != 2:
            raise ValueError("Vertex has to have degree 2.")

        else:
            neighbors = list(self.neighbors(vert))
            length = (
                self[neighbors[0]][vert]["weight"]
                + self[neighbors[1]][vert]["weight"]
            )

            self.remove_node(vert)
            self.add_edge(neighbors[0], neighbors[1], weight=length)

    def convert_phylo(self, leaves: dict[str, int]):
        """
        Converts the tree to a PhyloTree object.


        Parameters
        ----------
        leaves : dict[str, int]
            The labels of the leaves.
        Returns
        -------
        phtree : PhyloTree
            The tree as a PhyloTree.
        """

        if not nx.is_tree(self):
            raise ValueError("Graph is not a tree.")

        # import locally to avoid circ. dependency
        from stickytests.bhv.phylo_tree import PhyloTree

        graph = copy(self)
        # start by trimming vertices with degree 2 & merging connected edges
        ind = 1
        while ind != 0:
            ind = 0
            vertices = list(graph.nodes)
            for vert_label in vertices:
                if self.degree[vert_label] == 2:
                    ind += 1
                    graph.trim(vert_label)

        # get list of vertices with degree 1 & one with the internal vertices
        vdeg1 = []
        vinternal = []

        for vert in graph.nodes:
            if graph.degree[vert] == 1:
                vdeg1.append(vert)
            else:
                vinternal.append(vert)

        # get list of internal edges/splits
        splits = []
        edge_lengths = []

        if 0 not in leaves.values():
            root_candidates = [
                vert for vert in vdeg1 if str(vert) not in leaves.keys()
            ]
            if len(root_candidates) != 1:
                raise ValueError("Graph has multiple possible roots.")
            leaves = copy(leaves)
            leaves[str(root_candidates[0])] = 0

        for vert1 in vinternal:
            neighb = list(graph.neighbors(vert1))
            for vert2 in neighb:
                partition1, _ = graph.compute_split(vert2, vert1)
                length = graph[vert1][vert2]["weight"]
                splits.append(partition1)
                edge_lengths.append(length)

        leafcount = len(vdeg1) - 1
        phtree = PhyloTree(leafcount)
        for i, split in enumerate(splits):
            # convert vertex labels to leaf labels
            part = [leaves[str(j)] for j in split]
            phtree.add_split(part, edge_lengths[i])

        return phtree

    def get_distmat(self) -> np.ndarray:

        paths = dict(nx.all_pairs_dijkstra_path_length(self))
        vdeg1 = []
        for vert in self.nodes:
            if self.degree[vert] == 1:
                vdeg1.append(vert)

        distmat = np.zeros((len(vdeg1), len(vdeg1)))
        for i, leaf1 in enumerate(vdeg1):
            for j, leaf2 in enumerate(vdeg1[i + 1 :]):
                distmat[i, i + j + 1] = paths[leaf1][leaf2]

        distmat += distmat.T
        return distmat
