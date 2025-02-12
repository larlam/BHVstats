import numpy as np
from bhvstats.new2phylo import new2phylo

NEWICK_1 = "(((A:1.0,B:1.0):1.0,C:1.0):1.0,D:1.0)"
NEWICK_2 = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0)"


def test_newick_rooted():
    labels = {"A": 1, "B": 2, "C": 3, "D": 4}
    tree = new2phylo(NEWICK_1, labels)
    mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.all(tree.as_matrix() == mat)


def test_newick_unrooted():
    labels = {"A": 0, "B": 1, "C": 2, "D": 3}
    tree = new2phylo(NEWICK_2, labels)
    mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 2.0],
        ]
    )
    assert np.all(tree.as_matrix() == mat)
