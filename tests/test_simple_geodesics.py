import pytest
from bhvstats.phylo_simple_geod import PhyloSimpleGeod
from bhvstats.phylo_tree import PhyloTree


def get_cone_geod():
    t1 = PhyloTree(5)
    t1.add_split([1, 2], 4)
    t2 = PhyloTree(5)
    t2.add_split([0, 2, 5], 10)
    g = PhyloSimpleGeod(t1, t2)
    return g


def test_cone_path():
    g = get_cone_geod()
    assert len(g) == 1


def test_path_length():
    g = get_cone_geod()
    assert g.path_length() == 14
