"""
This module contains functions to convert Newick strings to PhyloTree objects.
"""

import copy
import numpy as np
from bhvstats.phylo_tree import PhyloTree
from bhvstats.forest import Forest


def load_treefile(
    filename: str, leaves: dict, delimiter=",", encoding="None"
) -> list[PhyloTree]:
    """
    Loads a tree file and returns the trees as PhyloTree objects.

    Parameters
    ----------
    filename : str
        The name of the file.
    leaves : dict
        The correspondence for the leaves in the Newick string.
    delimiter :
        The delimiter used in the Newick string.
    encoding :
        The encoding of the file.

    Returns
    -------
    trees : list[PhyloTree]
        The trees as PhyloTree objects.
    """
    trees = []
    with open(filename, "r", encoding=encoding) as file:
        for line in file:
            newick = line.strip(";\n")
            phylo = new2phylo(newick, leaves, delimiter)
            trees.append(phylo)
    return trees


def new2phylo(newick: str, leaves: dict, delimiter=",") -> PhyloTree:
    """
    Construct a phylogenetic tree from Newick representation.


    Parameters
    ----------
    newick : str
        A phylogenetic tree in Newick format.
    leaves : dict
        The correspondence for the leaves in the Newick string.
    delimiter :
        The delimiter used in the Newick string.

    Returns
    -------
    phylo : PhyloTree
        The tree as a PhyloTree object.
    """
    tree_g = Forest()
    # add root and 1 vertex
    parent = "p1"

    tree_g.add_edge("0", str(parent))

    # dont want spaces, replace delim w/ comma
    newick = newick.replace(" ", "")
    if delimiter != ",":
        newick = newick.replace(delimiter, ",")
    if newick[-1] == ";":
        newick = newick[:-1]

    add_children(tree_g, newick, parent)
    if 0 in leaves.values():
        tree_g.remove_node("0")
        tree_g.trim("p1")

    leaves_new = copy.copy(leaves)

    phylo = tree_g.convert_phylo(leaves_new)

    return phylo


def new2distmat(newick: str, leaves: dict, delimiter=",") -> np.ndarray:
    """
    Construct a distance matrix from Newick representation.

    Parameters
    ----------
    newick : str
        A phylogenetic tree in Newick format.
    leaves : dict
        The correspondence for the leaves in the Newick string.
    delimiter :
        The delimiter used in the Newick string.

    Returns
    -------
    dist_mat : list[list[float]]
        The distance matrix.
    """
    phylo = new2phylo(newick, leaves, delimiter)

    return phylo.as_matrix()


def add_children(tree: Forest, newick: str, parent: str):
    """
    Construts a tree from a Newick format.

    Parameters
    ----------
    tree : Tree
        The tree to be constructed.
    newick : str
        A phylogenetic tree in Newick format.
    parent : str
        The current parent node.
    """
    # recursively add children until the full tree is constructed
    string_cleaned, artif_nodes = cln_newick(newick, parent)
    dic = str2dict(string_cleaned)
    for key, value in dic.items():
        tree.add_edge(parent, key, value)
    if artif_nodes:
        for node, newick in artif_nodes.items():
            add_children(tree, newick, node)


def str2dict(string: str) -> dict:
    """
    Converts a string to a dictionary.

    Parameters
    ----------
    string : str
        A phylogenetic tree in Newick format.

    Returns
    -------
    dictionary : dict
        The dictionary.
    """
    dic = {}
    string_cp = copy.copy(string)
    string_cp = string_cp[1:-1]
    pos_kom = string_cp.find(",")
    while pos_kom != -1 or string_cp != "":
        if pos_kom != -1:
            string2 = string_cp[:pos_kom]
            string_cp = string_cp[pos_kom + 1 :]
        else:
            string2 = string_cp
            string_cp = ""
        pos_col = string2.find(":")
        dic[string2[:pos_col]] = float(string2[pos_col + 1 :])
        pos_kom = string_cp.find(",")
    return dic


def cln_newick(newick: str, parent: str) -> tuple[str, dict]:
    """
    Replaces all nested partitions in a Newick string with vertices.

    Parameters
    ----------
    newick : str
        A phylogenetic tree in Newick format.
    parent : str
        The current parent node.

    Returns
    -------
    cleaned : tuple[str, dict]
        The cleaned string and a dictionary detailing the correspondence
        between the new vertices and the removed ones.
    """
    artif_vertices = {}
    newick = newick[1:-1]
    i = newick.find("(")
    k = 0

    while i != -1:
        j = find_closure(newick, i)
        rem = newick[i : j + 1]
        # make sure no two vertices end up with the same name
        newvert = parent + "_" + str(k)
        artif_vertices[newvert] = rem
        k += 1
        newick = newick[:i] + newvert + newick[j + 1 :]
        i = newick.find("(")

    newick = "(" + newick + ")"

    cleaned = (newick, artif_vertices)
    return cleaned


def find_closure(string: str, position: int) -> int:
    """
    Finds the matching closing bracket for an opening bracket in a string.

    Parameters
    ----------
    string : str
        The string.
    position : int
        Position of the opening bracket.

    Returns
    -------
    pos_cl : int
        Position of the closing bracket.
    """
    # TODO: exceptions
    i = 1
    position_current = position
    while i != 0:
        position_current += 1
        char = string[position_current]
        if char == "(":
            i += 1
        elif char == ")":
            i -= 1
    pos_cl = position_current
    return pos_cl
