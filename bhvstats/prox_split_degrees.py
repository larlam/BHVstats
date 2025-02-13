import random
from copy import deepcopy
import numpy as np
from numpy import ndarray
from scipy.optimize import fsolve, minimize
from bhvstats.eval_geod import eval_geod
from bhvstats.tree_distance import distance
from bhvstats.phylo_tree import PhyloTree


class ProxSplit:
    def __init__(self, sample: list[PhyloTree], initial_guess=None):
        """
        A class to find a local minimum for the degrees of stickiness for a
        given sample of phylogenetic trees with the cone point as Frechet mean.

        Parameters
        ----------
        sample : list[PhyloTree]
            The sample.
        initial_guess : PhyloTree
            A possible initial_guess guess. If none is chosen, a point in the
            sample is chosen.
        """
        # we need to deepcopy everything to avoid manipulating the data set
        self.sample = deepcopy(sample)
        self.lengths = np.array([tree.norm() for tree in sample])
        self.lengthsum = np.sum(self.lengths)

        if initial_guess is None:
            self.cur_dir = deepcopy(self.sample[0])
        else:
            self.cur_dir = deepcopy(initial_guess)

        self.cur_dir.normalize()
        self.cosines = compute_cosines(self.cur_dir, self.sample)
        self.cur_deg = -np.mean(self.cosines * self.lengths)
        self.iteration = 0

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
        current = (self.cur_dir, self.cur_deg)
        return current

    def update(self):
        # Additional methods specific to the child class
        """
        Performs an update for the proposed direction.
        """
        for i, prop_dir in enumerate(self.sample):
            angle = np.arccos(compute_cosines(self.cur_dir, [prop_dir])[0])
            if 0 < angle < np.pi:
                prop_dir.normalize()
                # need to distinguish between the step size on the sphere and BHV space
                # stepsize_ang = np.sin(angle) / (self.__iteration + 2)
                # fnc = lambda x: (self.iteration + 1) * x**2 - self.lengths[i] / self.lengthsum * np.cos(angle - x)
                fnc = lambda x: (self.iteration + 1) ** 0.05 * x**2 - self.lengths[
                    i
                ] / self.lengthsum * np.cos(angle - x)
                stepsize_ang = (
                    minimize(fnc, np.pi / 2, bounds=[(0, np.pi)]).x[0] / np.pi
                )
                stepsize_bhv = np.sin(angle * stepsize_ang) / np.sin(
                    np.pi / 2 - angle * (stepsize_ang - 0.5)
                )
                stepsize_bhv = stepsize_bhv / (2 * np.sin(angle / 2))

                # compute geodesic and BHV space and project it back to the sphere
                new_dir = eval_geod(self.cur_dir, prop_dir, stepsize_bhv)
                new_dir.normalize()
                self.cur_dir = new_dir

            # update all relevant instance properties
            self.iteration += 1
        self.cosines = compute_cosines(self.cur_dir, self.sample)
        self.cur_deg = -np.mean(self.cosines * self.lengths)


class ProxSplitRandom(ProxSplit):
    def __init__(self, sample: list[PhyloTree], initial_guess=None):
        super().__init__(sample, initial_guess)
        # Additional initialization code for the child class

    def update(self):
        """
        Performs an update for the proposed direction.
        """
        # pick a direction that is less then pi away
        possdir = np.where(np.abs(self.cosines) < 1)[0]
        prop_dir = random.choices(possdir, weights=self.__lengths[possdir], k=1)[0]
        angle = np.arccos(self.cosines[prop_dir])
        prop_dir = self.sample[prop_dir]
        prop_dir.normalize()
        # need to distinguish between the step size on the sphere and BHV space
        # stepsize_ang = np.sin(angle) / (self.__iteration + 2)
        fnc = lambda x: (self.iteration + 1) * x - np.sin(angle - x)
        stepsize_ang = fsolve(fnc, [0, np.pi])[0] / np.pi
        stepsize_bhv = np.sin(angle * stepsize_ang) / np.sin(
            np.pi / 2 - angle * (stepsize_ang - 0.5)
        )
        stepsize_bhv = stepsize_bhv / (2 * np.sin(angle / 2))

        # compute geodesic and BHV space and project it back to the sphere
        new_dir = eval_geod(self.cur_dir, prop_dir, stepsize_bhv)
        new_dir.normalize()

        # update all relevant instance properties
        self.iteration += 1
        self.cur_dir = new_dir
        self.cosines = compute_cosines(self.cur_dir, self.sample)
        self.cur_deg = -np.mean(self.cosines * self.lengths)


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
