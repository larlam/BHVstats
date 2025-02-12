from bhvstats.forest import Forest
import numpy as np


def get_forest():
    forest = Forest()

    forest.add_node(0)
    forest.add_node(1)
    forest.add_node(2)
    forest.add_node(3)
    forest.add_node(4)
    forest.add_node("b1")
    forest.add_node("b2")
    forest.add_node("b3")
    forest.add_edge(0, "b1", 1)
    forest.add_edge(1, "b1", 1)
    forest.add_edge("b2", "b1", 1)
    forest.add_edge("b2", "b3", 1)
    forest.add_edge("b2", 2, 1)
    forest.add_edge("b3", 3, 1)
    forest.add_edge("b3", 4, 1)

    return forest


def test_split():
    f = get_forest()
    assert f.compute_split("b1", "b2")[0] == [0, 1] or f.compute_split(
        "b1", "b2"
    )[0] == [1, 0]
    assert f.compute_split("b2", "b3")[1] == [3, 4] or f.compute_split(
        "b2", "b3"
    )[1] == [4, 3]


def test_weight():
    f = get_forest()
    assert f["b2"]["b3"]["weight"] == 1


def test_trim():
    f = get_forest()
    f.remove_edge("b2", "b3")
    f.add_edge("b4", "b3", 1)
    f.add_edge("b2", "b4", 1)
    f.trim("b4")

    assert f["b2"]["b3"]["weight"] == 2


def test_conversion_unrooted():
    f = get_forest()
    f2 = f.convert_phylo(leaves={str(i): i for i in range(5)})
    assert f2.get_leafcount() == 4
    assert f2.get_splitcount() == 2


def test_conversion_rooted():
    f = get_forest()
    f2 = f.convert_phylo(leaves={str(i): i for i in range(1, 5)})
    assert f2.get_leafcount() == 4
    assert f2.get_splitcount() == 2


def test_distmat():
    f = get_forest()
    mat = f.get_distmat()
    assert np.all(
        mat
        == np.array(
            [
                [0.0, 2.0, 3.0, 4.0, 4.0],
                [2.0, 0.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 0.0, 3.0, 3.0],
                [4.0, 4.0, 3.0, 0.0, 2.0],
                [4.0, 4.0, 3.0, 2.0, 0.0],
            ]
        )
    )
