from __future__ import annotations


class PhyloSplit:
    def __init__(self,
                 lst : list,
                 N : int
                 ):
        """
        A class for splits in phylogenetic trees.

        Parameters
        ----------
        lst : list
            List containing one subset of the partition.
        N : int
            The number of leaves.
        """

        # order the list, compute the complement,
        # save the one with 0 as first in a tuple
        l1 = lst
        l1.sort()
        l2 =  [i for i in range(N+1) if i not in lst]

        if len(l1) == 0:
            raise Exception("The list must contain at least one integer.")

        if len(l1) + len(l2) != N + 1:
            raise Exception("The list must contain unique integers in the range from 0 to N.")

        if 0 in l1:
            self.partition = l1, l2
        else:
            self.partition = l2, l1


    def __str__ (self):
        return "{} | {}".format(self.partition[0], self.partition[1])

    def __lt__(self, other):
        s12 = set(self.get_split()[1])
        s22 = set(other.get_split()[1])

        return s12 < s22


    def __eq__(self,
               other : PhyloSplit
               ) -> bool:
        if isinstance(other, PhyloSplit):
            return self.partition == other.partition
        else:
            return False


    def __hash__(self):
        # convert lists to tuple to make it hashable, shouldn't pose a problem
        # as the lists are ordered
        # we need phylo_splits to be hashable to use it as keys in dictionaries
        # for the trees
        return hash((tuple(self.partition[0]), tuple(self.partition[1])))


    def get_split(self) -> tuple[list, list]:
        """
        Returns the partition.

        Returns
        -------
        tuple[list,list]
            Two lists representing the partition.

        """
        return self.partition


    # object for documentation
    def is_compatible(self,
                      other : PhyloSplit
                      ) -> bool:
        """
        Checks whether two splits are compatible.

        Parameters
        ----------
        other : phylo_split
            The other split.

        Returns
        -------
        bool
            Compatibility of the splits.

        Raises
        ------
        Exception:
            Only accepts phylo_split objects.
        """

        if isinstance(other, PhyloSplit):
            l1 = list(self.partition)
            l2 = list(other.partition)
            for i in l1:
                for j in l2:
                    inter = list(set(i) & set(j))
                    if len(inter) == 0:
                        return True
            else:
                return False

        else:
            raise Exception("Input must be phylo_split object.")
