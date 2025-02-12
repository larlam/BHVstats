import pytest
from bhvstats.phylo_split import PhyloSplit
from bhvstats.phylo_tree import PhyloTree


def get_tree():
    t = PhyloTree(5)
    t.add_split([2, 3, 4, 5], 1.0)
    t.add_split([3, 4, 5], 0.5)
    t.add_split([4, 5], 0.25)
    t.add_pendant_lengths(list(range(6)), list(range(6)))
    return t


def test_equality():
    t = get_tree()
    assert t == t


def test_inequality():
    t = get_tree()
    t2 = PhyloTree(5)
    assert t != t2


def test_matrix_conversion():
    t = get_tree()
    N = t.get_leafcount()
    M = t.as_matrix()
    t2 = PhyloTree(N)
    t2.from_matrix(M)
    assert t == t2


def graph_conversion():
    t = get_tree()
    N = t.get_leafcount()
    G = t.as_graph()
    leaves = {str(i): i for i in range(N + 1)}
    t2 = G.convert_phylo(leaves)
    assert t == t2


def test_bisect():
    t = get_tree()
    t1 = PhyloTree(5)
    t1.add_split([2, 3, 4, 5], 1.0)
    t2 = PhyloTree(5)
    t2.add_split([4, 5], 0.25)
    s = PhyloSplit([3, 4, 5], 5)
    tb1, tb2 = t.bisect(s)
    assert (t1 == tb1) and (t2 == tb2)
