"""
bhvstats
========

bhvstats is a Python package with tools for statistical analysis in BHV spaces
of phylogenetic trees.
"""

from bhvstats.forest import Forest
from bhvstats.phylo_tree import PhyloTree
from bhvstats.phylo_split import PhyloSplit
from bhvstats.phylo_simple_geod import PhyloSimpleGeod
from bhvstats.phylo_geod import PhyloGeod
from bhvstats.eval_geod import eval_geod
from bhvstats.grad_descent_degrees import DegreeDescent
from bhvstats.new2phylo import new2phylo, new2distmat
from bhvstats.project_trees import proj_trees
from bhvstats.prox_split_degrees import ProxSplit
from bhvstats.sturm_mean import sturm_mean, frechet_function
