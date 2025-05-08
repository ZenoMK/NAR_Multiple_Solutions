
import pandas as pd
import numpy as np
from torch.utils.hipify.hipify_python import compute_stats


from bf_dfs_verifiers import dfsverify, check_valid_BFpaths, bellman_ford_cost, agnostic_dfsverify
from test_permute import *

from scratch import draw_graph_with_highlights


# ----- GOAL: GET ACCURACY AND UNIQUENESS FOR n=5, n=16, n=64 ----- #
# that means, run this 3 times, reading in files for n=5,16,64


# A = np.array([[0,1,2],
#               [3,4,5],
#               [6,7,8]])
# perm = [2,0,1]
# A_perm = A[np.ix_(perm, perm)]
# print(A_perm)

def are_all_matrices_equal(group):
    """helper fn, called to find out whether As were permuted, use like `df.groupby('GraphID')['As'].agg(are_all_matrices_equal)`"""
    first = group.iloc[0]
    return all(np.array_equal(first, mat) for mat in group)


def preprocess(df):
    df['Perms'] = df['Perms'].apply(lambda A: A.tolist())           # get rid of jax array impl
    df['hashPerms'] = df['Perms'].apply(lambda ls: tuple(ls))
    df['Preds'] = df['Preds'].apply(lambda x: [int(i) for i in x])  # make ints not floats, since we use them to index
    df['hashablePreds'] = df['Preds'].apply(lambda ls: tuple(ls))   # make tuples if we want to count distinctness
    df_sorted = df.sort_values(by='GraphID')                        # make easy reading
    df_sorted['p_inv'] = df['Perms'].apply(lambda perm: invert_permutation(perm))  # add p_inv
    df_sorted['ogA'] = df_sorted.apply(lambda row: permute_adjacency_matrix(row['As'], row['p_inv']), axis=1) # restore og
    df_sorted['fakeOGpred'] = df_sorted.apply(lambda row: permute_parentpath(row['Preds'], row['p_inv']), axis=1)
    df_sorted['hashablefakeOGPreds'] = df_sorted['fakeOGpred'].apply(lambda ls: tuple(ls))
    return df_sorted


def compute_bf_stats(df):
    df = preprocess(df)
    # figure out where source got mapped to
    df['ogS'] = df.apply(lambda row: row['p_inv'][row['Ss']], axis=1)

    # TEST PERMUTATIONS AS INTENDED?
    # breakpoint()
    # test_adj(adj=df['ogA'][0], permuted_adj=df['As'][0], permutation=df['Perms'][0])
    # test_parent_paths(adj=df['ogA'][0], parentpath=df['fakeOGpred'][0], permuted_adj=df['As'][0], permuted_pp=df['Preds'][0], perm=df['Perms'][0])

    # breakpoint()
    # print('unpermuted costs', bellman_ford_cost(df['ogA'][0], df['ogS'][0]))
    # print('permuted costs', bellman_ford_cost(df['As'][0], df['Ss'][0]))
    # print('permuted validity', check_valid_BFpaths(df['As'][0], df['Ss'][0], df['Preds'][0]))
    # print('unpermuted validity', check_valid_BFpaths(df['ogA'][0], df['ogS'][0], df['fakeOGpred'][0]))
    # breakpoint()

    # TFAE cuz permutation
    df['valid(ogA,ogS,ogP)'] = df.apply(lambda row: check_valid_BFpaths(A=row['ogA'], s=row['ogS'], parentpath=row['fakeOGpred']), axis=1)
    df['valid(A,S,P)'] = df.apply(lambda row: check_valid_BFpaths(A=row['As'], s=row['Ss'], parentpath=row['Preds']), axis=1)

    df['valid(ogA,S,P)'] = df.apply(lambda row: check_valid_BFpaths(A=row['ogA'], s=row['Ss'], parentpath=row['Preds']), axis=1)
    df['valid(A,ogS,P)'] = df.apply(lambda row: check_valid_BFpaths(A=row['As'], s=row['ogS'], parentpath=row['Preds']), axis=1)

    # correctness
    print('Graph Size: ', len(df['Perms'][0]))
    print('weird: valid(ogA,S,P)', df['valid(ogA,S,P)'].mean())
    print('weird: valid(A,ogS,P)', df['valid(A,ogS,P)'].mean())
    print('valid(ogA,ogS,ogP)', df['valid(ogA,ogS,ogP)'].mean())
    print('valid(A,S,P)', df['valid(A,S,P)'].mean())

    # variety
    distinct = df.groupby('GraphID')['hashablePreds'].nunique().mean()
    print('unique-ness, ', distinct)
    deduplicated_distinct = df.groupby('GraphID')['hashablefakeOGPreds'].nunique().mean()
    print('weird unique-ness', deduplicated_distinct)
    num_perms = df.groupby('GraphID').size()[0]
    print('out of', num_perms, ' permutations')

    #breakpoint()
    # fixme: if permute everything, you care about ogA,S,P and dedup distinct || If permute pos, you care about A,S,P and distinct
        # you can reduce this to whether As are identical within graph ID or not.
    #breakpoint()
    As_were_permuted_flag = df.groupby('GraphID')['ogA'].agg(are_all_matrices_equal).all()
    if As_were_permuted_flag:
        adf = df[df['valid(ogA,S,P)']] # only look at valid rows, see if there's any variety
        uv = adf.groupby('GraphID')['hashablePreds'].nunique() # these are all accurate to ogA, so if any are different thats big whoop
    else:
        adf = df[df['valid(A,S,P)']] # these are all accurate to A (which is og graph since not permuted), so if any are different thats big whoop
        uv = adf.groupby('GraphID')['hashablePreds'].nunique()

    # --- filter variety first
    #mask = adf.groupby('GraphID')['hashablePreds'].nunique() != 1
    #graph_ids = mask[mask].index
    #fdf = df[df['GraphID'].isin(graph_ids)]
    #fdf[['GraphID', 'Perms', 'hashablePreds', 'valid(A,S,P)']]
    #breakpoint()

    # summary?
    sdf = df[['valid(ogA,ogS,ogP)', 'valid(A,S,P)', 'valid(ogA,S,P)']]

    return sdf, distinct, deduplicated_distinct, num_perms, (uv.mean()/num_perms, uv.std()/num_perms) # uv is unique and valid



def compute_dfs_stats(df):
    df = preprocess(df)

    # ORDER MATTERS - dfsverify
    df['dfsverify(ogA,P)'] = df.apply(lambda row: dfsverify(G=row['ogA'], F=row['Preds']), axis=1)
    df['dfsverify(A,ogP)'] = df.apply(lambda row: dfsverify(G=row['As'], F=row['fakeOGpred']), axis=1)
    # Sensible
    df['dfsverify(A,P)'] = df.apply(lambda row: dfsverify(G=row['As'], F=row['Preds']), axis=1)
    df['dfsverify(ogA,ogP)'] = df.apply(lambda row: dfsverify(G=row['ogA'], F=row['fakeOGpred']), axis=1) # these can vary relative to DFSvalid, since order matters for dfsverify so permutation affects correctness

    print('weird: dfsverify(ogA, P)', df['dfsverify(ogA,P)'].mean())
    print('weird2: dfsverify(A, ogP)', df['dfsverify(A,ogP)'].mean())
    print('dfsverify(ogA,ogP)', df['dfsverify(ogA,ogP)'].mean())
    print('dfsverify(A, P)', df['dfsverify(A,P)'].mean())


    # RANDOM
    df['rando'] = df['Preds'].apply(lambda pred: np.random.randint(0,len(pred),len(pred)))
    df['randoValid1'] = df.apply(lambda row: dfsverify(G=row['As'], F=row['rando']), axis=1)
    df['randoValid2'] = df.apply(lambda row: dfsverify(G=row['ogA'], F=row['rando']), axis=1)
    print('dfsverify(A,rando):', df['randoValid1'].mean())
    print('dfsverify(ogA,rando):', df['randoValid2'].mean())

    # ORDER DOESNT - AGNOSTIC
    df['agnostic(ogA,P)'] = df.apply(lambda row: agnostic_dfsverify(G=row['ogA'], F=row['Preds']), axis=1)
    df['agnostic(ogA,ogP)'] = df.apply(lambda row: agnostic_dfsverify(G=row['ogA'], F=row['fakeOGpred']), axis=1)
    df['agnostic(A,P)'] = df.apply(lambda row: agnostic_dfsverify(G=row['As'], F=row['Preds']), axis=1)
    print('agnostic(ogA, ogP)', df['agnostic(ogA,ogP)'].mean())
    print('agnostic(A,P)', df['agnostic(A,P)'].mean())
    # diff_df = df[df['agnostic(ogA,ogP)']!=df['agnostic(A,P)']] # these should be the same, but if not its handy to inspect
    # assert len(diff_df) == 0
    # breakpoint()
    # print('oga', diff_df['ogA'].iloc[0])
    # print('As', diff_df['As'].iloc[0])
    # print('Perms', diff_df['Perms'].iloc[0])
    # print('actual', diff_df['Preds'].iloc[0])
    # print('og', diff_df['fakeOGpred'].iloc[0])
    #
    # print('OG:', diff_df['agnostic(ogA,ogP)'].iloc[0])
    # draw_graph_with_highlights(diff_df['ogA'].iloc[0], diff_df['fakeOGpred'].iloc[0])
    # print('actual', diff_df['agnostic(A,P)'].iloc[0])
    # draw_graph_with_highlights(diff_df['As'].iloc[0], diff_df['Preds'].iloc[0])
    #test_adj(diff_df['ogA'].iloc[0], diff_df['As'].iloc[0], diff_df['Perms'].iloc[0])
    #test_parent_paths(diff_df['ogA'].iloc[0], diff_df['fakeOGpred'].iloc[0], diff_df['As'].iloc[0], diff_df['Preds'].iloc[0], diff_df['Perms'].iloc[0])
    #breakpoint()

    distinct = df.groupby('GraphID')['hashablePreds'].nunique().mean()
    print('uniqueness, ', distinct)
    deduplicated_distinct = df.groupby('GraphID')['hashablefakeOGPreds'].nunique().mean()
    print('weird uniqueness', deduplicated_distinct)
    num_perms = df.groupby('GraphID').size()[0]
    print('out of', num_perms, ' permutations')


    As_were_permuted_flag = df.groupby('GraphID')['ogA'].agg(are_all_matrices_equal).all()
    if As_were_permuted_flag:
        adf = df[df['agnostic(ogA,P)']] # who is valid
        uv = adf.groupby('GraphID')['hashablePreds'].nunique() # how many are valid per graph

    else:
        adf = df[df['agnostic(A,P)']]
        uv = adf.groupby('GraphID')['hashablePreds'].nunique()

    #adf = df[df['dfsverify(A,P)']]  # only look at valid rows, see if theres any variety
    #adf.groupby('GraphID')['hashablePreds'].nunique() # BEHOLD, VARIETY DISAPPEARS || was this also true for our sampling methods? one correct many different?
    #mask = adf.groupby('GraphID')['hashablePreds'].nunique() != 1
    #graph_ids = mask[mask].index
    #fdf = df[df['GraphID'].isin(graph_ids)]
    #breakpoint()


    # summary? important cols
    sdf = df[['dfsverify(A,P)', 'dfsverify(ogA,ogP)', 'agnostic(ogA,ogP)', 'agnostic(A,P)', 'dfsverify(ogA,P)', 'agnostic(ogA,P)']]
    #breakpoint()
    return sdf, distinct, deduplicated_distinct, num_perms, (uv.mean()/num_perms, uv.std()/num_perms)


if __name__ == '__main__':
    pass
    print('DFS BAByyyy ============')

    #dfs_df = pd.read_pickle('dfs_testingpermute_69420_.pkl')
    dfs_df = pd.read_pickle('dfs_testingpermute_50_.pkl')
    dfs_stats = compute_dfs_stats(dfs_df)

    print('Bellman Ford ============')

    bf_df = pd.read_pickle('bellman_ford_testingpermute_50_.pkl')
    bf_stats = compute_bf_stats(bf_df)


#bf_df = pd.read_pickle('bellman_ford_testingpermute_50_.pkl')

# ----------------------------------------------------------------------------------------------------------------------
# SCRATCH N SNIFF ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# def sanity_check_permuting_bf(A1, s1, perm, path):
#     """test to see if permuting stuff works with check_valid_BF_paths"""
#     pass
#
# perm = [2, 0, 3, 1]
# p_inv = np.argsort(perm)
# s = np.argmax([0, 0, 1, 0])
# permS = np.argmax([1, 0, 0, 0])
# unpermS = perm[permS]
#
#
# print(unpermS, s)
#
#
# ogA = np.array(
# [[0.79427108, 0., 0.81151536, 0.44486781],
#  [0., 0., 0.78179651, 0.21992143],
# [0.81151536, 0.78179651, 0.08532829, 0.],
# [0.44486781, 0.21992143, 0., 0.07875175]])
#
# permA = np.array(
#  [[0.08532829, 0.81151536, 0.,         0.78179651],
#  [0.81151536, 0.79427108, 0.44486781, 0.        ],
#  [0.,         0.44486781, 0.07875175, 0.21992143],
#  [0.78179651, 0.,         0.21992143, 0.        ]]
# )





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
# df_sorted['dfsverify(ogA,P)'] = df_sorted.apply(lambda row: dfsverify(G=row['ogA'], F=row['Preds']), axis=1) # this would be crazy if worked
# print('validity relative to OG graph', df_sorted['dfsverify(ogA,P)'].mean())
#
# # SECOND BEST: are you valid to your graph (this implies DFS correct without ordered restarts, but is still stricter)
# df_sorted['dfsverify(A,P)'] = df_sorted.apply(lambda row: dfsverify(G=row['As'], F=row['Preds']), axis=1)
# print('validity relative to input (permuted) graph', df_sorted['dfsverify(A,P)'].mean())
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
# from bf_dfs_verifiers import dfsverify, check_valid_BFpaths
#
# # sort by GraphID
# df_sorted = df.sort_values(by='GraphID')
#
# df_sorted['dfsverify(A,P)'] = df_sorted.apply(lambda row: dfsverify(G=row['As'], F=row['Preds']), axis=1)
#
# df_sorted
#
#
# df_sorted.groupby('GraphID')['dfsverify(A,P)'].mean()
#
# df_sorted.iloc[9]
#
# df_sorted['dfsverify(A,P)'].mean()
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
# df_sorted['dfsverify(ogA,P)'] = df_sorted.apply(lambda row: dfsverify(G=row['ogA'], F=row['Preds']), axis=1) # this would be crazy if worked
# df_sorted['fullogDFSvalid'] = df_sorted.apply(lambda row: dfsverify(G=row['ogA'], F=row['fakeOGpred']), axis=1)  # this might work with random restart dfs model
#
#
# # fixme: something weird is happening with ogDFSvalid 1000train, everything valid? does this mean permutations are not happening or smth
# # fixme: working hypothesis is that the predictions are happening relative to the OG adjacency matrix, and that accuracy 100% on n=4 makes sense
# # fixme: ok when you do bigger graphs, like n=32, it's 0% accuracy so no probs
# # fixme: THE BIG QUESTION IS WHY ARE THINGS MORE ogDFSvalid (current prediction, old adjacency matrix) than DFSvalid (current prediction, current adj matrix) || no longer true
# df_sorted.groupby('GraphID')['dfsverify(ogA,P)'].mean()
# df_sorted.groupby('GraphID')['dfsverify(A,P)'].mean()
# df_sorted.groupby('GraphID')['fullogDFSvalid'].mean()
# df_sorted['dfsverify(ogA,P)'].mean()      # 95%
# df_sorted['dfsverify(A,P)'].mean()        # 26%
# df_sorted['fullogDFSvalid'].mean()  # 7%
#
#
# # compute uniques after inverting
# df_sorted['hashablefakeOGPreds'] = df_sorted['fakeOGpred'].apply(lambda ls: tuple(ls))
# df_sorted.groupby('GraphID')['hashablefakeOGPreds'].nunique()



# SANITY: when you make everything the identity permutation, you get back to the same result dfs and bellman ford
# TRY: permuting other parts of feedback.features?
# ---- BELLMAN FORD
#   1) reversing pos seems to do nothing to bellman_ford n=4
#   2) reversing s definitely affects bellman_ford
#   3) reversing A seems to do nothing to bellman_ford n=4
#   4) reversing adj seems to affect bellman_ford n=4 # fixme: this seems important
#   5) GIVEN --hint_mode none, no hints matter ||| hints can matter slightly with encoded_decoded

# ---- DFS
#   1) reversing pos seems to do lots (pos is meant to encode sequential data)
#   2) reversing A does something
#   3) reversing adj does something
#   4) GIVEN --hint_mode none, no hints matter ||| hints can matter slightly with encoded_decoded
#
