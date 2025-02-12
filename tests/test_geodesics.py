import pytest
from bhvstats.phylo_geod import PhyloGeod
from bhvstats.phylo_tree import PhyloTree
from bhvstats.tree_distance import tree_distance


def get_geod():
    t1 = PhyloTree(7)
    t1.add_split([5, 6], 0.88)
    t1.add_split([3, 4], 1)
    t1.add_split([3, 4, 5, 6], 0.47)
    t1.add_split([2, 3, 4, 5, 6], 0.73)
    t1.add_split([1, 2, 3, 4, 5, 6], 0.83)

    t2 = PhyloTree(7)
    t2.add_split([3, 4], 0.5)
    t2.add_split([3, 4, 5], 0.15)
    t2.add_split([2, 3, 4, 5], 0.87)
    t2.add_split([6, 7], 0.42)
    t2.add_split([2, 3, 4, 5, 6, 7], 0.7)

    g = PhyloGeod(t1, t2)
    return g


def test_geod_len():
    g = get_geod()
    assert g.get_path_length() == 2.7293989189202916


def test_revert_dir():
    g = get_geod()
    t1, t2 = g.get_trees()
    g_rev = PhyloGeod(t2, t1)
    assert g.get_path_length() == g_rev.get_path_length()


def test_midpoint():
    g = get_geod()
    midp = g.eval(0.5)
    t1, t2 = g.get_trees()

    d1 = tree_distance(t1, midp)
    d2 = tree_distance(t2, midp)
    midlen = 2.7293989189202916 / 2
    assert d1 == pytest.approx(midlen) and d2 == pytest.approx(midlen)
