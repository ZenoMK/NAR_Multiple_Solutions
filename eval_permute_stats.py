# Fixme: smth weird with bellman ford correctness to OG graph. Should be the same,
#  since relabeling graph shouldnt affect path length or path structure

import pandas as pd
import numpy as np
from torch.utils.hipify.hipify_python import compute_stats


from bf_dfs_verifiers import henry, check_valid_BFpaths


# TODO: print num permutations, dfs_df.groupby('GraphID').size()[0]

# ----- GOAL: GET ACCURACY AND UNIQUENESS FOR n=5, n=16, n=64 ----- #
# that means, run this 3 times, reading in files for n=5,16,64
def invert_pred(pred, perm):
    inv_perm = np.argsort(perm)
    return [inv_perm[x] for x in pred]


def preprocess(df):
    df['Perms'] = df['Perms'].apply(lambda A: A.tolist())           # get rid of jax array impl
    df['Preds'] = df['Preds'].apply(lambda x: [int(i) for i in x])  # make ints not floats, since we use them to index
    df['hashablePreds'] = df['Preds'].apply(lambda ls: tuple(ls))   # make tuples if we want to count distinctness
    df_sorted = df.sort_values(by='GraphID')                        # make easy reading
    df_sorted['ogA'] = df_sorted.apply(lambda row: row['As'][np.ix_(np.argsort(row['Perms']), np.argsort(row['Perms']))], axis=1)
    df_sorted['fakeOGpred'] = df_sorted.apply(lambda row: invert_pred(row['Preds'], row['Perms']), axis=1)
    df_sorted['hashablefakeOGPreds'] = df_sorted['fakeOGpred'].apply(lambda ls: tuple(ls))
    return df_sorted


def compute_bf_stats(df):
    df = preprocess(df)
    # figure out where source got mapped to
    df['source'] = df['Perms'].apply(lambda perm: perm[0])

    df['BFvalid'] = df.apply(lambda row: check_valid_BFpaths(A=row['As'], s=row['source'], parentpath=row['Preds']), axis=1)
    df['ogBFvalid'] = df.apply(lambda row: check_valid_BFpaths(A=row['ogA'], s=0, parentpath=row['Preds']), axis=1) # this should be same cuz permutation

    print('Graph Size: ', len(df['Perms'][0]))
    print('validity relative to OG graph', df['ogBFvalid'].mean())
    print('validity relative to input (permuted) graph', df['BFvalid'].mean())
    distinct = df.groupby('GraphID')['hashablePreds'].nunique().mean()
    print('uniqueness, ', distinct)

    inflated_distinct = df.groupby('GraphID')['hashablefakeOGPreds'].nunique().mean()
    print('werid uniqueness', inflated_distinct)
    return


def compute_dfs_stats(df):
    df = preprocess(df)

    df['ogDFSvalid'] = df.apply(lambda row: henry(G=row['ogA'], F=row['Preds']), axis=1)
    df['DFSvalid'] = df.apply(lambda row: henry(G=row['As'], F=row['Preds']), axis=1)

    print('validity relative to OG graph', df['ogDFSvalid'].mean())
    print('validity relative to input (permuted) graph', df['DFSvalid'].mean())
    distinct = df.groupby('GraphID')['hashablePreds'].nunique().mean()
    print('uniqueness, ', distinct)
    inflated_distinct = df.groupby('GraphID')['hashablefakeOGPreds'].nunique().mean()
    print('werid uniqueness', inflated_distinct)
    return


if __name__ == '__main__':
    print('DFS BAByyyy ============')

    dfs_df = pd.read_pickle('dfs_testingpermute_10_.pkl')
    dfs_stats = compute_dfs_stats(dfs_df)

    print('Bellman Ford ============')

    bf_df = pd.read_pickle('bellman_ford_testingpermute_10_.pkl')
    bf_stats = compute_bf_stats(bf_df)



# ----------------------------------------------------------------------------------------------------------------------
# SCRATCH N SNIFF ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


#
# df0 = pd.read_pickle('testingpermute-100train.pkl')
# df = df0
# # ----- Preprocess ----- #
# df['Perms'] = df['Perms'].apply(lambda A: A.tolist())
# df['Preds'] = df['Preds'].apply(lambda x: [int(i) for i in x])
# df['hashablePreds'] = df['Preds'].apply(lambda ls: tuple(ls))
# df_sorted = df.sort_values(by='GraphID')
#
# print('Graph Size: ', len(df['Perms'][0]))
# # we have kinda 2 sensible options for unique/validity
# # BEST VALIDITY: are you valid to the original graph (but why would this be correct for DFSO? it wouldnt)
# df_sorted['ogA'] = df_sorted.apply(lambda row: row['As'][np.ix_(np.argsort(row['Perms']),np.argsort(row['Perms']))], axis=1)
# df_sorted['ogDFSvalid'] = df_sorted.apply(lambda row: henry(G=row['ogA'], F=row['Preds']), axis=1) # this would be crazy if worked
# print('validity relative to OG graph', df_sorted['ogDFSvalid'].mean())
#
# # SECOND BEST: are you valid to your graph (this implies DFS correct without ordered restarts, but is still stricter)
# df_sorted['DFSvalid'] = df_sorted.apply(lambda row: henry(G=row['As'], F=row['Preds']), axis=1)
# print('validity relative to input (permuted) graph', df_sorted['DFSvalid'].mean())
#
# # BEST UNIQUENESS: are you unique?
# distinct = df_sorted.groupby('GraphID')['hashablePreds'].nunique().mean()
# print('uniqueness, ', distinct)
#
# # SECOND BEST: are you unique after undoing the permutation ...?
# def invert_pred(pred, perm):
#     inv_perm = np.argsort(perm)
#     return [inv_perm[x] for x in pred]
#
# df_sorted['fakeOGpred'] = df_sorted.apply(lambda row: invert_pred(row['Preds'], row['Perms']), axis=1)
# df_sorted['hashablefakeOGPreds'] = df_sorted['fakeOGpred'].apply(lambda ls: tuple(ls))
# inflated_distinct = df_sorted.groupby('GraphID')['hashablefakeOGPreds'].nunique().mean()
# print('werid uniqueness', inflated_distinct)




# and then we can get specific into Unique and Valid?


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# # ----------- OLD N SCRATCH ------- #
# # turn all columns into nice lists of ints
# #df['As'] = df['As'].apply(lambda A: A.tolist())
# df['Perms'] = df['Perms'].apply(lambda A: A.tolist())
# df['Preds'] = df['Preds'].apply(lambda x: [int(i) for i in x])
# df['hashablePreds'] = df['Preds'].apply(lambda ls: tuple(ls))
#
# # compute valids
# from bf_dfs_verifiers import henry, check_valid_BFpaths
#
# # sort by GraphID
# df_sorted = df.sort_values(by='GraphID')
#
# df_sorted['DFSvalid'] = df_sorted.apply(lambda row: henry(G=row['As'], F=row['Preds']), axis=1)
#
# df_sorted
#
#
# df_sorted.groupby('GraphID')['DFSvalid'].mean()
#
# df_sorted.iloc[9]
#
# df_sorted['DFSvalid'].mean()
#
#
#
# # compute uniques
# df_sorted.groupby('GraphID')['hashablePreds'].nunique()
# df_sorted.groupby('GraphID')['hashablePreds'].nunique().mean()
#
# # --------- UNDO THE PERMUTATION, TO EVALUATE PREDS N STUFF RELATIVE TO ORIGINAL GRAPH? --------- #
# # np.argsort undoes the permutation i think
# # P_inv = np.argsort(Perm)
# # A_original = A_permuted[np.ix_(P_inv, P_inv)]
#
# # restore A
# df_sorted['ogA'] = df_sorted.apply(lambda row: row['As'][np.ix_(np.argsort(row['Perms']),np.argsort(row['Perms']))], axis=1)
#
# # restore pred
# # pred = [0,2,0,0]
# # pinv = [2,3,0,1]
# # newpred = [pinv[x] for x in pred]
# def invert_pred(pred, perm):
#     inv_perm = np.argsort(perm)
#     return [inv_perm[x] for x in pred]
#
# df_sorted['fakeOGpred'] = df_sorted.apply(lambda row: invert_pred(row['Preds'], row['Perms']), axis=1)
#
#
#
# # comparing these two shows undoing the perms works, restoring OG As
# df_sorted['HogA'] = df_sorted['ogA'].apply(lambda x: tuple(map(tuple, x)))
# df_sorted.groupby('GraphID')['HogA'].nunique() # hashable ogAs
#
# df_sorted['HAs'] = df_sorted['As'].apply(lambda x: tuple(map(tuple, x)))
# df_sorted.groupby('GraphID')['HAs'].nunique()
#
# # Valids relative to og graph
# df_sorted['ogDFSvalid'] = df_sorted.apply(lambda row: henry(G=row['ogA'], F=row['Preds']), axis=1) # this would be crazy if worked
# df_sorted['fullogDFSvalid'] = df_sorted.apply(lambda row: henry(G=row['ogA'], F=row['fakeOGpred']), axis=1)  # this might work with random restart dfs model
#
#
# # fixme: something weird is happening with ogDFSvalid 1000train, everything valid? does this mean permutations are not happening or smth
# # fixme: working hypothesis is that the predictions are happening relative to the OG adjacency matrix, and that accuracy 100% on n=4 makes sense
# # fixme: ok when you do bigger graphs, like n=32, it's 0% accuracy so no probs
# # fixme: THE BIG QUESTION IS WHY ARE THINGS MORE ogDFSvalid (current prediction, old adjacency matrix) than DFSvalid (current prediction, current adj matrix)
# df_sorted.groupby('GraphID')['ogDFSvalid'].mean()
# df_sorted.groupby('GraphID')['DFSvalid'].mean()
# df_sorted.groupby('GraphID')['fullogDFSvalid'].mean()
# df_sorted['ogDFSvalid'].mean()      # 95%
# df_sorted['DFSvalid'].mean()        # 26%
# df_sorted['fullogDFSvalid'].mean()  # 7%
#
#
# # compute uniques after inverting
# df_sorted['hashablefakeOGPreds'] = df_sorted['fakeOGpred'].apply(lambda ls: tuple(ls))
# df_sorted.groupby('GraphID')['hashablefakeOGPreds'].nunique()