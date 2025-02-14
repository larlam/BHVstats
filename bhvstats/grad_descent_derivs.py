"""
This module contains the class DerivDescent for finding a local minimum for the
directional derivatives of the Frechet function.
"""

import random
from copy import deepcopy
import numpy as np
from numpy import ndarray
from bhvstats.tree_distance import distance, eval_geod
from bhvstats.phylo_tree import PhyloTree


class DerivDescent:
    def __init__(self, sample: list[PhyloTree], initial_guess=None):
        """
        A class to find a local minimum for the directional derivatives of the
        Frechet function given sample of phylogenetic trees with the cone point
        as Frechet mean.

        Parameters
        ----------
        sample : list[PhyloTree]
            The sample.
        initial_guess : PhyloTree
            A possible initial_guess guess. If none is chosen, a point in the
            sample is chosen.
        """
        # we need to deepcopy everything to avoid manipulating the data set
        self.__sample = deepcopy(sample)
        self.__lengths = np.array([tree.norm() for tree in sample])

        if initial_guess is None:
            self.__cur_dir = deepcopy(self.__sample[0])
        else:
            self.__cur_dir = deepcopy(initial_guess)

        self.__cur_dir.normalize()
        self.__cosines = compute_cosines(self.__cur_dir, self.__sample)
        self.__cur_deg = -np.mean(self.__cosines * self.__lengths)
        self.__iteration = 0

    def get_current(self) -> tuple[PhyloTree, float]:
        """
        Yields both the current direction and value for the degrees of
        stickiness.

        Returns
        -------
        tuple[PhyloTree, float]
            The current proposed direction and its corresponding directional
            derivative.
        """
        self.__cosines = compute_cosines(self.__cur_dir, self.__sample)
        self.__cur_deg = -np.mean(self.__cosines * self.__lengths)
        current = (self.__cur_dir, self.__cur_deg)
        return current

    def update(self):
        """
        Performs an update for the proposed direction.
        """
        # pick a direction that is less then pi away
        possdir = np.where(np.abs(self.__cosines) < 1)[0]
        prop_dir = random.choices(possdir, weights=self.__lengths[possdir], k=1)[0]
        # angle = np.arccos(self.__cosines[prop_dir])
        prop_dir = self.__sample[prop_dir]
        angle = np.arccos(
            np.clip(compute_cosines(self.__cur_dir, [prop_dir])[0], -1, 1)
        )
        prop_dir.normalize()
        # need to distinguish between the step size on the sphere and BHV space
        stepsize_ang = np.sin(angle) / (self.__iteration + 2)
        stepsize_bhv = np.sin(angle * stepsize_ang) / np.sin(
            np.pi / 2 - angle * (stepsize_ang - 0.5)
        )
        stepsize_bhv = stepsize_bhv / (2 * np.sin(angle / 2))
        # print(stepsize_bhv)

        # compute geodesic and BHV space and project it back to the sphere
        new_dir = eval_geod(self.__cur_dir, prop_dir, stepsize_bhv)
        new_dir.normalize()

        # update all relevant instance properties
        self.__iteration += 1
        self.__cur_dir = new_dir


def compute_cosines(direction: PhyloTree, sample: list[PhyloTree]) -> ndarray:
    """
    Computes the cosines for the Alexandrov angle at the cone point between a
    fixed direction and a sample of phylogenetic trees.

    Parameters
    ----------
    direction : PhyloTree
        The direction in which
    sample : list[PhyloTree]
        A list of trees.

    Returns
    -------
    cosines : ndarray
        An array containing the cosines.
    """
    sample_size = len(sample)
    cosines = np.zeros(sample_size)
    for i in range(sample_size):
        tree = sample[i]
        dist = distance(direction, tree)
        norm = tree.norm()
        if norm != 0:
            cosines[i] = (norm**2 + 1 - dist**2) / (2 * norm)

    return cosines
