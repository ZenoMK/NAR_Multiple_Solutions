import numpy as np
import clrs._src.dfs_sampling as dfs_sampling
import clrs._src.algorithms.check_graphs as check_graphs

def check_uniqueness_dfs(probMatrices, n_samples = 10):
    uniques = []
    valids = []

    # issue: we are getting back list of trees from sample upwards, so we need to merge those in a way where
    # every index corresponds to the probmatrix we are sampling from
    #breakpoint()
    samples = np.array([dfs_sampling.sample_upwards(probMatrices) for j in range(n_samples)])
    probMatrices_format = dfs_sampling.extract_probMatrices(probMatrices)
    for i in range(len(probMatrices_format)):

        samples_matrix_i = samples[:,i]

        # code from here: https://stackoverflow.com/questions/26514179/set-of-list-of-lists-in-python
        unique_trees = [list(item) for item in set([tuple(row) for row in samples_matrix_i])]


        # save the fraction of unique samples
        uniques.append(len(unique_trees)/n_samples)

        valid_trees = [check_graphs.check_valid_dfsTree(probMatrices_format[i],j) for j in unique_trees]
        valids.append(sum(valid_trees)/n_samples)

    return uniques,valids
