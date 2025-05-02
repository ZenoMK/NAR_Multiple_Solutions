import pandas as pd
import numpy as np

df = pd.read_pickle('testingpermute-100train.pkl')#pd.read_pickle('testingpermute-1000-manyperms.pkl')  #pd.read_pickle('testingpermute-1000train.pkl')

# turn all columns into nice lists of ints
#df['As'] = df['As'].apply(lambda A: A.tolist())
df['Perms'] = df['Perms'].apply(lambda A: A.tolist())
df['Preds'] = df['Preds'].apply(lambda x: [int(i) for i in x])
df['hashablePreds'] = df['Preds'].apply(lambda ls: tuple(ls))

# compute valids
from bf_dfs_verifiers import henry, check_valid_BFpaths

# sort by GraphID
df_sorted = df.sort_values(by='GraphID')

df_sorted['DFSvalid'] = df_sorted.apply(lambda row: henry(G=row['As'], F=row['Preds']), axis=1)

df_sorted


df_sorted.groupby('GraphID')['DFSvalid'].mean()

df_sorted.iloc[9]

df_sorted['DFSvalid'].mean()



# compute uniques
df_sorted.groupby('GraphID')['hashablePreds'].nunique()
df_sorted.groupby('GraphID')['hashablePreds'].nunique().mean()

# --------- UNDO THE PERMUTATION, TO EVALUATE PREDS N STUFF RELATIVE TO ORIGINAL GRAPH? --------- #
# np.argsort undoes the permutation i think
# P_inv = np.argsort(Perm)
# A_original = A_permuted[np.ix_(P_inv, P_inv)]

# restore A
df_sorted['ogA'] = df_sorted.apply(lambda row: row['As'][np.ix_(np.argsort(row['Perms']),np.argsort(row['Perms']))], axis=1)

# restore pred
# pred = [0,2,0,0]
# pinv = [2,3,0,1]
# newpred = [pinv[x] for x in pred]
def invert_pred(pred, perm):
    inv_perm = np.argsort(perm)
    return [inv_perm[x] for x in pred]

df_sorted['fakeOGpred'] = df_sorted.apply(lambda row: invert_pred(row['Preds'], row['Perms']), axis=1)



# comparing these two shows undoing the perms works, restoring OG As
df_sorted['HogA'] = df_sorted['ogA'].apply(lambda x: tuple(map(tuple, x)))
df_sorted.groupby('GraphID')['HogA'].nunique() # hashable ogAs

df_sorted['HAs'] = df_sorted['As'].apply(lambda x: tuple(map(tuple, x)))
df_sorted.groupby('GraphID')['HAs'].nunique()

# Valids relative to og graph
df_sorted['ogDFSvalid'] = df_sorted.apply(lambda row: henry(G=row['ogA'], F=row['Preds']), axis=1) # this would be crazy if worked
df_sorted['fullogDFSvalid'] = df_sorted.apply(lambda row: henry(G=row['ogA'], F=row['fakeOGpred']), axis=1)  # this might work with random restart dfs model


# fixme: something weird is happening with ogDFSvalid 1000train, everything valid? does this mean permutations are not happening or smth
# fixme: working hypothesis is that the predictions are happening relative to the OG adjacency matrix, and that accuracy 100% on n=4 makes sense
# fixme: ok when you do bigger graphs, like n=32, it's 0% accuracy so no probs
# fixme: THE BIG QUESTION IS WHY ARE THINGS MORE ogDFSvalid (current prediction, old adjacency matrix) than DFSvalid (current prediction, current adj matrix)
df_sorted.groupby('GraphID')['ogDFSvalid'].mean()
df_sorted.groupby('GraphID')['DFSvalid'].mean()
df_sorted.groupby('GraphID')['fullogDFSvalid'].mean()
df_sorted['ogDFSvalid'].mean()      # 95%
df_sorted['DFSvalid'].mean()        # 26%
df_sorted['fullogDFSvalid'].mean()  # 7%


# compute uniques after inverting
df_sorted['hashablefakeOGPreds'] = df_sorted['fakeOGpred'].apply(lambda ls: tuple(ls))
df_sorted.groupby('GraphID')['hashablefakeOGPreds'].nunique()