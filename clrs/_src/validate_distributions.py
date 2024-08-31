from clrs._src.algorithms.BF_beamsearch import BF_beamsearch, BF_greedysearch
from clrs._src.dfs_sampling import get_parent_tree_upwards, single_sample_upwards

from clrs._src.algorithms.check_graphs import check_valid_BFpaths, check_valid_dfsTree
from clrs._src.dfs_sampling import extract_probMatrices
from sciplotlib import style as spstyle
from clrs._src.algorithms.graphs import bellman_ford, dfs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('graph1 hooks to BF_collect_and_eval in log_experiments.py')
print('edge_reuse_matrix_list works on dummy example')

# TODO: make make_edge_reuse_matrix_list split by unique/valid trees


#------------------------------------------
# DO THE THING WITH LINE GRAPHS
# GIVEN, a distribution "preds", and an adjacency matrix
# 1. run sampling algorithms increasingly many times on preds
# 2. count how many valid unique solutions
# 3. count how many invalid unique solutions
#------------------------------------------

# solution, beamOrGreedy?, valid?, unique?, how_many_times_found?, (maybe also adjacency matrix)


def DFS_graph1_df(A, pred, num_solutions_extracted):
    # columns for dataframe
    upwards_trees = []
    upwards_valid = []
    upwards_unique = []

    altUpwards_trees = []
    altUpwards_valid = []
    altUpwards_unique = []

    # frequency dicts
    upwards_dict = dict()
    altUpwards_dict = dict()

    # core routine
    num_sampled = 0
    while num_sampled < num_solutions_extracted:
        num_sampled += 1
        # extract new trees
        up_tree = single_sample_upwards(pred)
        alt_tree = get_parent_tree_upwards(pred)
        # save them for later dataframe
        upwards_trees.append(up_tree)
        altUpwards_trees.append(alt_tree)

        # valid?
        upwards_valid.append(check_valid_dfsTree(A, up_tree))
        altUpwards_valid.append(check_valid_dfsTree(A, alt_tree))

        # unique?
        hash_up = hash(tuple(up_tree))
        upwards_unique.append(hash_up not in upwards_dict.keys())
        hash_alt = hash(tuple(alt_tree))
        altUpwards_unique.append(hash_alt not in altUpwards_dict.keys())

        # update freq dict
        if hash_up not in upwards_dict.keys():
            upwards_dict[hash_up] = 1
        else:
            upwards_dict[hash_up] += 1

        if hash_alt not in altUpwards_dict.keys():
            altUpwards_dict[hash_alt] = 1
        else:
            altUpwards_dict[hash_alt] += 1

    df = pd.DataFrame.from_dict(
            {'up_trees': upwards_trees, 'unique_up': upwards_unique, 'valid_up': upwards_valid,
             'alt_trees': altUpwards_trees, 'unique_alt': altUpwards_unique, 'valid_alt': altUpwards_valid}
    )
    df['total_unique_seen'] = df['unique_alt'].cumsum()
    df['total_valid_seen'] = df['valid_alt'].cumsum()
    df['unique_and_valid'] = (df['unique_alt'] & df['valid_alt']).cumsum()
    print('validate...only doing alt')

    return df, A, pred

def average_dataframes(df_list):
    #breakpoint()
    combined_df = pd.concat(df_list)
    # u & v
    mean_unique_and_valid = combined_df['unique_and_valid'].groupby(combined_df.index).mean()
    median_unique_and_valid = combined_df['unique_and_valid'].groupby(combined_df.index).mean()
    max_uv = combined_df['unique_and_valid'].groupby(combined_df.index).max()
    min_uv = combined_df['unique_and_valid'].groupby(combined_df.index).min()
    # just unique
    mean_unique = combined_df['total_unique_seen'].groupby(combined_df.index).mean()
    median_unique = combined_df['total_unique_seen'].groupby(combined_df.index).mean()
    max_u = combined_df['total_unique_seen'].groupby(combined_df.index).max()
    min_u = combined_df['total_unique_seen'].groupby(combined_df.index).min()
    # just valid
    mean_valid = combined_df['total_valid_seen'].groupby(combined_df.index).mean()
    median_valid = combined_df['total_valid_seen'].groupby(combined_df.index).mean()
    max_v = combined_df['total_valid_seen'].groupby(combined_df.index).max()
    min_v = combined_df['total_valid_seen'].groupby(combined_df.index).min()
    # summarize

    df = pd.DataFrame.from_dict({'mean_unique_and_valid': mean_unique_and_valid,
                                 'median_unique_and_valid': median_unique_and_valid,
                                 'max_uv': max_uv,
                                 'min_uv': min_uv,
                                 'mean_unique': mean_unique,
                                 'median_unique': median_unique,
                                 'max_u': max_u,
                                 'min_u': min_u,
                                 'mean_valid': mean_valid,
                                 'median_valid': median_valid,
                                 'max_v': max_v,
                                 'min_v': min_v
                                 })
    df['samples_seen'] = df.index+1
    return df


def DFS_plot(df):
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    plt.plot(df.index + 1, df.total_unique_seen, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title('num_unique by num_sampled')
    plt.xlabel('num_sampled')
    plt.ylabel('num_unique')
    plt.show()


#------------------------------------------
# GRAPH NUM UNIQUE by NUM SAMPLES
# - test num solutions in distribution
#------------------------------------------

def validate_distributions(As, Ss, outsOrPreds, numSolsExtracting, flag, edge_reuse_BF= False):
    #breakpoint()
    probMatrix_list = extract_probMatrices(outsOrPreds)
    dataframes = []
    pMs = []
    for ix in range(len(probMatrix_list)):
        #breakpoint()
        A = As[ix]
        startNode = Ss[ix]
        probMatrix = probMatrix_list[ix]
        # build a plot,
        if flag=='BF':
            dataframes.append(make_n_unique_by_n_extracted_df(A=A, s=startNode, pred=probMatrix, num_solutions_extracted=numSolsExtracting))
        elif edge_reuse_BF:
            dataframes.append(make_edge_reuse_matrix_list(A, startNode, probMatrix,numSolsExtracting))
        elif flag=='DFS':
            df, A, pM = DFS_graph1_df(A=A, pred=probMatrix, num_solutions_extracted=numSolsExtracting)
            dataframes.append(df)
            pMs.append(pM)
        else:
            raise ValueError('no flag given to validate_distributions')
    return dataframes, As, pMs

def plot_n_unique_by_n_extracted(df, graphsize):
    '''df produced by make_n_unique_by_n_extracted_df'''
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df)
    # u & v
    total_uv_seen_beam_mean = df['total_uv_seen_beam'].groupby(df.index).mean()
    total_uv_seen_beam_std = df['total_uv_seen_beam'].groupby(df.index).std()

    total_uv_seen_greedy_mean = df['total_uv_seen_greedy'].groupby(df.index).mean()
    total_uv_seen_greedy_std = df['total_uv_seen_greedy'].groupby(df.index).std()

    total_uv_seen_bf_mean = df['total_unique_seen_bf'].groupby(df.index).mean()
    total_uv_seen_bf_std = df['total_unique_seen_bf'].groupby(df.index).std()

    plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_beam_mean, marker='o', linestyle='-', color = "blue", label = "Beamsearch")
    plt.fill_between([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_beam_mean-total_uv_seen_beam_std, total_uv_seen_beam_mean+total_uv_seen_beam_std, color = "blue", alpha = 0.15)

    plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='x', linestyle='-', color="red", label = "Greedy")
    plt.fill_between([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean - total_uv_seen_greedy_std,
                     total_uv_seen_greedy_mean + total_uv_seen_greedy_std, color="red", alpha=0.15)
    #plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_bf_mean, marker='v', linestyle='-',
             color="green", label="Bellman-Ford")
    plt.fill_between([i for i in range(len(total_uv_seen_bf_mean))],
                     total_uv_seen_bf_mean - total_uv_seen_bf_std,
                     total_uv_seen_bf_mean + total_uv_seen_bf_std, color="red", alpha=0.15)
    plt.legend(loc = "upper left")
    #plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title('Unique solutions vs sampled solutions')
    plt.xlabel('Sampled solutions')
    plt.ylabel('Unique and valid solutions')
    plt.savefig(f"plot_unique_by_extracted_{graphsize}.png")

def make_n_unique_by_n_extracted_df(A, s, pred, num_solutions_extracted):
    '''

    Args:
        A: an adjacency matrix
        s: starting node index (e.g. 5)
        pred: a predecessor array encoding a distribution of solutions (e.g. typical output of NN: [[],[],[]])
        num_solutions_extracted: a list of places where you want uniqueness evaluated (e.g. [5,25,100]

    Returns:
        df: pandas dataframe carrying info needed for plot

    '''
    num_samples_drawn = 0
    #
    bf_frequency_dict = dict()
    beam_frequency_dict = dict()
    greedy_frequency_dict = dict()
    greedy_hashes = set()
    beam_hashes = set()
    #
    beam_valid = None
    greedy_valid = None
    #
    bf_trees_col = []
    greedy_trees_col = []
    beam_trees_col = []
    valid_greedy = []
    valid_beam = []
    #
    unique_bf = []
    unique_greedy = []
    unique_beam = []
    times_found_greedy = []
    times_found_beam = []

    while num_samples_drawn < num_solutions_extracted:
        # take a sample, see if it's unique.
        tree1 = BF_beamsearch(A, s, pred)
        tree2 = BF_greedysearch(A, s, pred)
        tree3 = adj_matrix_to_parent_tree(bellman_ford(A,s,deterministic=True)[0])
        #
        beam_trees_col.append(tree1)
        greedy_trees_col.append(tree2)
        bf_trees_col.append(tree3)
        #

        # is it valid?
        valid_beam.append(check_valid_BFpaths(A, s, tree1))
        valid_greedy.append(check_valid_BFpaths(A, s, tree2))


        # how many times have we seen it before?
        hash_beam = hash(tuple(tree1))
        hash_greedy = hash(tuple(tree2))
        hash_bf = hash(tuple(tree3))
        #breakpoint()

        # if not seen before, add 1. else add 0, so we can cumulatively sum df column
        unique_beam.append(hash_beam not in beam_frequency_dict.keys())
        unique_greedy.append(hash_greedy not in greedy_frequency_dict.keys())
        unique_bf.append(hash_bf not in bf_frequency_dict.keys())

        # abcd?
        if hash_beam not in beam_frequency_dict.keys():
            beam_frequency_dict[hash_beam] = 1
        else:
            beam_frequency_dict[hash_beam] += 1

        if hash_greedy not in greedy_frequency_dict.keys():
            greedy_frequency_dict[hash_greedy] = 1
        else:
            greedy_frequency_dict[hash_greedy] += 1

        if hash_bf not in bf_frequency_dict.keys():
            bf_frequency_dict[hash_bf] = 1
        else:
            bf_frequency_dict[hash_bf] += 1

        #times_found_beam.append()
        #times_found_greedy.append()
        # TODO: SAVE THE DICTIONARY AT THE END, with tree specifics

        num_samples_drawn += 1
    # FIXME interval code nonsense

    df = pd.DataFrame.from_dict(
            {'beam_sols': beam_trees_col, 'unique_beam':unique_beam, 'valid_beam':valid_beam,
             'investment_bankers': greedy_trees_col, 'unique_greedy': unique_greedy, 'valid_greedy': valid_greedy,
             'bellman_ford_sols':bf_trees_col, 'unique_bf':unique_bf}
    )
    df['total_uv_seen_beam'] = (df['unique_beam'] & df['valid_beam']).cumsum()
    df['total_uv_seen_greedy'] = (df['unique_greedy'] & df['valid_greedy']).cumsum()
    df['total_unique_seen_bf'] = df['unique_bf'].cumsum()
    df.index += 1


    return df












#------------------------------------------
# GRAPH EDGE REUSE BY NUM SAMPLES
# - test similarity among solutions
#------------------------------------------

def postprocess_edge_reuse_matrix_list(matrix_lists):
    '''make ready for plot'''
    # this is all for a single graph, where each matrix represents a subset of edges... perhaps in graph... extracted from parent tree
    # at each interval, sum adjacency matrices, calculate frequency, report score

    #n_samples_list = []
    medians = []
    means = []
    for matrix_list in matrix_lists:
        median_list = []
        mean_list = []
        for ix in range(len(matrix_list)):
            #n_samples_list.append(ix+1)

            # sum first how-many np.arrays
            summing_list = matrix_list[:ix+1]
            sum_matrix = np.sum(summing_list, axis=0)
            frac_matrix = sum_matrix/(ix+1)
            # compute summary stats
            median = np.median(frac_matrix)
            mean = np.mean(frac_matrix)

            # save them
            median_list.append(median)
            mean_list.append(mean)
        medians.append(median_list)
        means.append(mean_list)

    df = pd.DataFrame.from_dict(
        {'greedy_medians': medians[0], 'greedy_means': means[0],
         'beam_medians': medians[1], 'beam_means': means[1],
         'bf_medians': medians[2], 'bf_means': means[2]}
        )

    return df

def plot_edge_reuse_matrix_list_mean(df, graphsize):
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df)
    # u & v
    mean_edge_reuse_beam_mean = df['beam_means'].groupby(df.index).mean()
    mean_edge_reuse_beam_std = df['beam_means'].groupby(df.index).std()

    mean_edge_reuse_greedy_mean = df['greedy_means'].groupby(df.index).mean()
    mean_edge_reuse_greedy_std = df['greedy_means'].groupby(df.index).std()

    mean_edge_reuse_bf_mean = df['bf_means'].groupby(df.index).mean()
    mean_edge_reuse_bf_std = df['bf_means'].groupby(df.index).std()

    plt.plot([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_beam_mean, marker='o', linestyle='-',
             color="blue", label="Beamsearch")
    plt.fill_between([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_beam_mean - mean_edge_reuse_beam_std,
                     mean_edge_reuse_beam_mean + mean_edge_reuse_beam_std, color="blue", alpha=0.15)

    plt.plot([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_greedy_mean, marker='x', linestyle='-',
             color="red", label="Greedy")
    plt.fill_between([i for i in range(len(mean_edge_reuse_beam_mean))],
                     mean_edge_reuse_greedy_mean - mean_edge_reuse_greedy_std,
                     mean_edge_reuse_greedy_mean + mean_edge_reuse_greedy_std, color="red", alpha=0.15)
    # plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    plt.plot([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_bf_mean, marker='v', linestyle='-',
             color="green", label="Bellman-Ford")
    plt.fill_between([i for i in range(len(mean_edge_reuse_beam_mean))],
                     mean_edge_reuse_bf_mean - mean_edge_reuse_bf_std,
                     mean_edge_reuse_bf_mean + mean_edge_reuse_bf_std, color="red", alpha=0.15)
    plt.legend(loc="upper left")
    plt.plot(df.n_samples, df.medians, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, 1))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title('Mean average edge reuse by num sampled')
    plt.ylabel('Mean average edge reuse')
    plt.xlabel('Samples')
    plt.savefig("edge_reuse_mean_"+str(graphsize)+".png")
    plt.close()

def make_edge_reuse_matrix_list(A, s, pred, num_solutions_extracted):
    '''
    plot edge reuse by num samples...

    Args:
        A:
        s:
        pred:
        num_solutions_extracted:

    Returns:
        matrices: list of matrices with edges used in parent trees

    '''
    # gather many solutions
    sol_counter = 0
    greedy_matrices = []
    beam_matrices = []
    bf_matrices = []

    while sol_counter < num_solutions_extracted:
        sol_counter += 1
        greedy_tree = BF_greedysearch(A, s, pred)
        beam_tree = BF_beamsearch(A, s, pred)
        bf_tree = bellman_ford(A,s,deterministic=True)

        # convert tree to adjacency matrix
        greedy_matrix = parent_tree_to_adj_matrix(greedy_tree)
        beam_matrix = parent_tree_to_adj_matrix(beam_tree)
        bf_matrix = parent_tree_to_adj_matrix(bf_tree)

        # save tree adj matrix
        greedy_matrices.append(greedy_matrix)
        beam_matrices.append(beam_matrix)
        bf_matrices.append(bf_matrix)

    return [greedy_matrices, beam_matrices, bf_matrices]

def parent_tree_to_adj_matrix(tree):
    size = len(tree)    # n_vertices
    M = np.zeros((size, size))
    for ix in range(size):
        M[int(tree[ix]), ix] = 1     # edge points tree[ix] to ix, bcuz parent tree
    return M

def adj_matrix_to_parent_tree(A):
    size = len(A)
    tree = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if A[i,j] == 1:
                tree[j] = i
    return tree

def graph3(A, s, pred, num_solutions_extracted):
    '''
    TODO: do it by edit distance

    Args:
        A:
        s:
        pred:
        num_solutions_extracted:

    Returns:

    '''
    raise NotImplementedError




if __name__ == '__main__':

    print('testing _src/validate_distributions.py')

    test_A = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    test_s = 1

    test_pred = np.array([
        [0.1, 0.3, 0.6],
        [0.2, 0.8, 0.0],
        [0.4, 0.5, 0.1]
    ])

    test_intervals = 50

    '''
    df1 = make_n_unique_by_n_extracted_df(A=test_A, s=test_s, pred=test_pred, num_solutions_extracted=test_intervals)
    plot_n_unique_by_n_extracted(df1)

    ms = make_edge_reuse_matrix_list(A=test_A, s=test_s, pred=test_pred, num_solutions_extracted=test_intervals)
    df = postprocess_edge_reuse_matrix_list(ms)
    plot_edge_reuse_matrix_list(df)
    '''
    df_list = []
    for i in range(100):
        df, A, pM = DFS_graph1_df(A=test_A, pred=test_pred, num_solutions_extracted=test_intervals)
        df_list.append(df)
    df = average_dataframes(df_list)


    plt.plot(df.samples_seen, df.median_unique, marker='o', linestyle='-')