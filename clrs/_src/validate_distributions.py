from algorithms.BF_beamsearch import BF_beamsearch, BF_greedysearch

from algorithms.check_graphs import check_valid_BFpaths, check_valid_dfsTree

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('YIKES 2, works on dummy test but not hooked to main code')


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

def plot1(df):
    plt.plot(df.index, df.unique_beam, 'o')
    plt.show()
    return df

def plotty(df):
    # make df cum sum of num unique
    df['total_unique_seen'] = df['unique_beam'].cumsum()
    # plot
    plt.plot(df.index+1, df.total_unique_seen, 'o')
    plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.show()
    return df

def graph1(A, s, pred, num_sample_intervals):
    '''

    Args:
        A: an adjacency matrix
        s: starting node index (e.g. 5)
        pred: a predecessor array encoding a distribution of solutions (e.g. typical output of NN: [[],[],[]])
        num_sample_intervals: a list of places where you want uniqueness evaluated (e.g. [5,25,100]

    Returns:
        df: pandas dataframe carrying info needed for plot

    '''
    pointer_in_intervals = 0
    interval = num_sample_intervals[pointer_in_intervals]
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

    while num_samples_drawn < interval:
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

    #df.to_csv('../../results/figure_fodder')
    #breakpoint()
    plotty(df)

    return df





#------------------------------------------
# GRAPH EDGE REUSE BY NUM SAMPLES
# - test similarity among solutions
#------------------------------------------


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

    test_intervals = [5]

    graph1(A=test_A, s=test_s, pred=test_pred, num_sample_intervals=test_intervals)