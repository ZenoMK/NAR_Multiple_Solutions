from clrs._src.algorithms.BF_beamsearch import BF_beamsearch, BF_greedysearch

from clrs._src.algorithms.check_graphs import check_valid_BFpaths, check_valid_dfsTree
from clrs._src.dfs_sampling import extract_probMatrices

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('graph1 hooks to BF_collect_and_eval in log_experiments.py')
print('graph2 works on dummy example')

# TODO: make graph2 split by unique/valid trees


#------------------------------------------
# DO THE THING WITH LINE GRAPHS
# GIVEN, a distribution "preds", and an adjacency matrix
# 1. run sampling algorithms increasingly many times on preds
# 2. count how many valid unique solutions
# 3. count how many invalid unique solutions
#------------------------------------------

# solution, beamOrGreedy?, valid?, unique?, how_many_times_found?, (maybe also adjacency matrix)


#------------------------------------------
# GRAPH NUM UNIQUE by NUM SAMPLES
# - test num solutions in distribution
#------------------------------------------

def validate_distributions(As, Ss, outsOrPreds, numSolsExtracting):
    breakpoint()
    probMatrix_list = extract_probMatrices(outsOrPreds)
    dfs = []
    for ix in range(len(probMatrix_list)):
        #breakpoint()
        A = As[ix]
        startNode = Ss[ix]
        probMatrix = probMatrix_list[ix]
        # build a plot,
        dfs.append(graph1(A=A, s=startNode, pred=probMatrix, num_solutions_extracted=numSolsExtracting))
    return dfs

def plot1(df):
    plt.plot(df.index, df.unique_beam, 'o')
    plt.show()
    return df

def plotty(df):
    # make df cum sum of num unique
    df['total_unique_seen'] = df['unique_beam'].cumsum()
    # plot
    plt.plot(df.index + 1, df.total_unique_seen, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.show()
    return df

def scheming(df):
    # make df cum sum of num unique
    df['total_unique_seen'] = df['unique_beam'].cumsum()
    # plot
    plt.figure()
    plt.plot(df.index + 1, df.total_unique_seen, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    return plt.gca()

def graph1(A, s, pred, num_solutions_extracted):
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
    beam_frequency_dict = dict()
    greedy_frequency_dict = dict()
    greedy_hashes = set()
    beam_hashes = set()
    #
    beam_valid = None
    greedy_valid = None
    #
    greedy_trees_col = []
    beam_trees_col = []
    valid_greedy = []
    valid_beam = []
    #
    unique_greedy = []
    unique_beam = []
    times_found_greedy = []
    times_found_beam = []

    while num_samples_drawn < num_solutions_extracted:
        # take a sample, see if it's unique.
        tree1 = BF_beamsearch(A, s, pred)
        tree2 = BF_greedysearch(A, s, pred)
        #
        beam_trees_col.append(tree1)
        greedy_trees_col.append(tree2)
        #

        # is it valid?
        valid_beam.append(check_valid_BFpaths(A, s, tree1))
        valid_greedy.append(check_valid_BFpaths(A, s, tree2))


        # how many times have we seen it before?
        hash_beam = hash(tuple(tree1))
        hash_greedy = hash(tuple(tree2))
        #breakpoint()

        # if not seen before, add 1. else add 0, so we can cumulatively sum df column
        unique_beam.append(hash_beam not in beam_frequency_dict.keys())
        unique_greedy.append(hash_greedy not in greedy_frequency_dict.keys())

        # abcd?
        if hash_beam not in beam_frequency_dict.keys():
            beam_frequency_dict[hash_beam] = 1
        else:
            beam_frequency_dict[hash_beam] += 1

        if hash_greedy not in greedy_frequency_dict.keys():
            greedy_frequency_dict[hash_greedy] = 1
        else:
            greedy_frequency_dict[hash_greedy] += 1

        #times_found_beam.append()
        #times_found_greedy.append()
        # TODO: SAVE THE DICTIONARY AT THE END, with tree specifics

        num_samples_drawn += 1
    # FIXME interval code nonsense

    df = pd.DataFrame.from_dict(
            {'beam_sols': beam_trees_col, 'unique_beam':unique_beam, 'valid_beam':valid_beam,
             'investment_bankers': greedy_trees_col, 'unique_greedy': unique_greedy, 'valid_greedy': valid_greedy}
    )
    df['total_unique_seen'] = df['unique_beam'].cumsum()

    #df.to_csv('../../results/figure_fodder')
    #breakpoint()
    #plotty(df)

    return df



#------------------------------------------
# GRAPH EDGE REUSE BY NUM SAMPLES
# - test similarity among solutions
#------------------------------------------

def postprocess_graph2(matrix_list):
    '''make ready for plot'''
    # this is all for a single graph, where each matrix represents a subset of edges... perhaps in graph... extracted from parent tree
    # at each interval, sum adjacency matrices, calculate frequency, report score
    n_samples_list = []
    median_list = []
    mean_list = []
    for ix in range(len(matrix_list)):
        n_samples_list.append(ix+1)

        # sum first how-many np.arrays
        summing_list = matrix_list[:ix]
        sum_matrix = np.sum(summing_list, axis=0)
        frac_matrix = sum_matrix/(ix+1)

        # compute summary stats
        median = np.median(frac_matrix)
        mean = np.mean(frac_matrix)

        # save them
        median_list.append(median)
        mean_list.append(mean)

    df = pd.DataFrame.from_dict(
        {'n_samples': n_samples_list, 'medians': median_list, 'means': mean_list}
        )

    return df

def plot_graph2(df):
    plt.plot(df.n_samples, df.medians, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, 1))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.show()

def graph2(A, s, pred, num_solutions_extracted):
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
    matrices = []

    while sol_counter < num_solutions_extracted:
        sol_counter += 1
        greedy_tree = BF_greedysearch(A, s, pred)

        # convert tree to adjacency matrix
        greedy_matrix = parent_tree_to_adj_matrix(greedy_tree)

        # save tree adj matrix
        matrices.append(greedy_matrix)

    return matrices

def parent_tree_to_adj_matrix(tree):
    size = len(tree)    # n_vertices
    M = np.zeros((size, size))
    for ix in range(size):
        M[int(tree[ix]), ix] = 1     # edge points tree[ix] to ix, bcuz parent tree
    return M




def graph3(A, s, pred, num_solutions_extracted):
    '''
    do it by edit distance

    Args:
        A:
        s:
        pred:
        num_solutions_extracted:

    Returns:

    '''
    raise NotImplementedError

# solutions, ORIGINAL GRAPH (adj matrix).


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

    df = graph1(A=test_A, s=test_s, pred=test_pred, num_solutions_extracted=test_intervals)
    #plt.plot(df.index + 1, df.total_unique_seen, marker='o', linestyle='-')
    #plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    #plt.show()
    ms = graph2(A=test_A, s=test_s, pred=test_pred, num_solutions_extracted=test_intervals)
    df = postprocess_graph2(ms)

    plot_graph2(df)