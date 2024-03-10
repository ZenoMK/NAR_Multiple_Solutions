import numpy as np
import clrs._src.dfs_sampling as dfs_sampling
import clrs._src.algorithms.check_graphs as check_graphs

def check_uniqueness_dfs(probMatrices, n_samples = 10):
    uniques = []
    valids = []
    # extract correct datatype
    probMatrices = dfs_sampling.extract_probMatrices(probMatrices)
    for i in probMatrices:
        samples = [dfs_sampling.sample_upwards(i) for j in range(n_samples)]
        unique_trees = set(samples)

        # save the fraction of unique samples
        uniques.append(len(unique_trees)/n_samples)

        valid_trees = [check_graphs.check_valid_dfsTree(i,j) for j in unique_trees]
        valids.append(sum(valid_trees)/n_samples)

    return uniques,valids
