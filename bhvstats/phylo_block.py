import numpy as np
from typing import Optional
from stickytests.bhv.phylo_split import PhyloSplit
from copy import copy


class PhyloBlock:
    def __init__(
        self,
        rem: np.ndarray,
        add: np.ndarray,
        compatibility: Optional[np.ndarray] = None,
    ):
        """
        A class for the individual blocks of a curve in a BHV space.


        Parameters
        ----------
        rem : dict
            Removed splits.
        add : dict
            Added splits.
        """
        self.rem = rem
        self.add = add
        self.len_add = np.linalg.norm(add[:, -1])
        self.len_rem = np.linalg.norm(rem[:, -1])
        # 1 stands for unknown, if 0 then it is not extendable
        self.extendable = 1

        if compatibility is None:
            n_add = add.shape[0]
            n_rem = rem.shape[0]
            compatibility = np.zeros((n_add, n_rem))
            rem = np.expand_dims(rem, 1)
            add = np.expand_dims(add, 0)
            compatibility = (rem - add)[:, :, :-1]
            compatibility = np.all(compatibility >= 0, axis=2) + np.all(
                compatibility <= 0, axis=2
            )
            compatibility += np.all((rem + add)[:, :, :-1] < 2, axis=2)

        self.compatibility = compatibility

    def get_block(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the block.

        Returns
        -------
        tuple[dict[PhyloSplit, float],
                                     dict[PhyloSplit, float]]
            The block.
        """
        return self.rem, self.add

    def get_length(self) -> float:
        """
        Computes the length of the block.

        Returns
        -------
        length : float
            The length of the block.
        """
        length = self.len_rem + self.len_add
        return length

    def eval(self, t: float) -> np.ndarray:
        """
        Given a point, evaluates the position of the block.

        Parameters
        ----------
        t : float
            The position.

        Returns
        -------
        dict[PhyloSplit, float]
            The splits and their lengths at the given position.
        """

        cutoff = t * self.get_length()
        if self.get_length() == 0:
            splits = np.empty((0, self.rem.shape[1]))

        elif cutoff < self.len_rem:
            ratio = 1 - cutoff / self.len_rem
            splits = copy(self.rem)
            splits[:, -1] *= ratio
        else:
            ratio = (cutoff - self.len_rem) / self.len_add
            splits = copy(self.add)
            splits[:, -1] *= ratio

        return splits
