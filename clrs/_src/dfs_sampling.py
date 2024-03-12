import numpy as np

def sample_upwards(outsOrPreds):
    '''

    Args:
        outsOrPreds: A list of Datapoint or Dictionaries containing JaxArray Probability Matrices

    Returns:
        trees: A list of parent trees, one for each probability Matrix in unpacked outsOrPreds.
        Each parent tree, pi, an array of length := #nodes. pi[i] = 3 indicates 3 is the parent of node i.

    Data Structures:
        distlist: A list of probability matrices
        probMatrix: A matrix, #nodes by #nodes, where probMatrix[i][j] indicates the probability that node j is the parent of node i.
        leafiness: A sorted list. 0th element is the node least likely to be a parent: the sum its probMatrix column is the lowest
        pi: the candidate parent-tree being built
    '''
    trees = []
    #breakpoint()
    # PREPROCESS TO EXTRACT probMatrix
    for i in outsOrPreds:
        if type(i) == type({}):
            distlist = i["pi"].data
        else:
            distlist = i.data
        for probMatrix in distlist: # note, probMatrix is a jax ArrayImpl
            probMatrix = np.array(probMatrix) # deepcopy to numpy so mutable
            ### COMPUTATION HERE
            #. sort by leafiness
            leafiness = np.asarray(leafinessSort(probMatrix)) # shallowcopy jax array to numpy array so no problems indexing

            # turn probmatrix into true probability dist (summing to 1)
            #altered_ProbMatrix= probMatrix
            #. grab most leafy, find its parent, continue till already-discovered (self-parent or prev. iter).
            pi = np.full(len(probMatrix), np.inf)
            while sum(leafiness) > -len(leafiness):
                altered_ProbMatrix = rowWiseProb(probMatrix)
                leaf = leafiness[leafiness != -1][0]
                # sample the leafs parent
                parent = chooseUniformly(altered_ProbMatrix[leaf])
                pi[leaf] = parent # FIXME sometimes index error
                leafiness[leaf] = -1
                leafiness[parent] = -1
                altered_ProbMatrix[:,leaf] = 0 # set leaf's column to 0: leaf should be nobody's parent, unless there's a restart, to avoid cycles... BREAKS

                # sample up the tree until parent is the start node, a self-loop or already has a parent
                while pi[parent] == np.inf:
                    # sample up the tree
                    leaf = parent
                    parent = chooseUniformly(altered_ProbMatrix[leaf])
                    pi[leaf] = parent
                    # remove parent as potential
                    leafiness[leaf] = -1
                    leafiness[parent] = -1
                    altered_ProbMatrix[:, leaf] = 0 # set leaf's column to 0: leaf should be nobody's parent, unless there's a restart, to avoid cycles

            if sum(np.isin(pi, np.inf)) > 0:
                raise ValueError("Leaf with no parent")
            trees.append(pi)
    return trees

def sample_argmax(outsOrPreds):
    trees = []
    distlist = extract_probMatrices(outsOrPreds)
    for probM in distlist:
        amax = np.argmax(probM, axis=1)
        trees.append(amax)
    #breakpoint()
    return trees

def sample_argmax_listofdict(preds):
    trees = []
    for i in preds: # de-listify into dict, happens twice
        distlist = i["pi"].data
        for prob in distlist:
            amax = np.argmax(prob, axis=1)
            #print(amax)
            trees.append(amax)
    return trees

def sample_argmax_listofdatapoint(outputs):
    '''argmax'ing index for each row in probMatrix'''
    trees = []
    for i in outputs: #de-listify into datapoint
        distlist = i.data
        for prob in distlist:
            amax = np.argmax(prob, axis=1)
            #print(amax)
            trees.append(amax)
    return trees

def sample_random_list(outsOrPreds):
    '''Random Number for each row in probMatrix'''
    trees = []
    rng = np.random.default_rng()
    #breakpoint()
    for i in outsOrPreds:
        if type(i) == type({}):
            distlist = i["pi"].data
        else:
            distlist = i.data
        for probMatrix in distlist:
            pi = []
            for row in probMatrix:
                pi.append(rng.integers(len(row)))
            trees.append(pi)
            #breakpoint()
    return trees

def leafinessSort(probMatrix):
    '''
    Args:
        probMatrix: Expects probMatrix[i][j] to indicate the probability that node j is the parent of node i

    Returns: sorted-list of vertex indices where first node had lowest probability of being parent. (column with lowest sum).
    '''
    sums = np.sum(probMatrix, axis=0)
    # sort by sum column, remember the original column number
    leafiness = np.argsort(sums)
    return leafiness

def rowWiseProb(probMatrix):
    for row_ix in range(len(probMatrix)):
        if probMatrix[row_ix].sum() != 0:
            probMatrix[row_ix] = probMatrix[row_ix].astype(np.float64)/probMatrix[row_ix].astype(np.float64).sum()
    return probMatrix

def chooseUniformly(notProbArray):
    '''Expects notProbArray'''
    val = np.random.uniform(low=0, high=sum(notProbArray))
    sums = np.cumsum(notProbArray)
    for threshold_ix in range(len(sums)):
        if val < sums[threshold_ix]:
            return threshold_ix
    return np.random.randint(len(notProbArray))

def extract_probMatrices(outsOrPreds):
    ''''
    handles ugly formatting difference: outs a list of dicts of datapoints,
    preds a list of datapoints
    '''
    big_probmatrix_list = []
    for i in outsOrPreds:
        if type(i) == type({}):
            distlist = i["pi"].data
        else:
            distlist = i.data
        big_probmatrix_list.extend(distlist)
    return big_probmatrix_list

def explore_upwards(orphan_ix, parent_guesses, probMatrix):
    '''starting at node_ix, sample parents upwards until you find a node which already has a parent'''
    while parent_guesses[orphan_ix] == np.inf: # until you find node who already has parent,
        # sample parent according to row of notExactlyProbs
        parent_guess = chooseUniformly(probMatrix[orphan_ix])
        parent_guesses[orphan_ix] = parent_guess
        # try further up the tree
        orphan_ix = parent_guess
    return parent_guesses


def get_parent_tree_upwards(probMatrix):
    '''according to leafiness, explore upwards, until all nodes have parents (could be themselves)'''
    parent_guesses = np.full(len(probMatrix), np.inf)
    leafiness = leafinessSort(probMatrix)  # most leafy is least-likely to be a parent: lowest column sum of probMatrix
    for node_ix in leafiness:
        if parent_guesses[node_ix] == np.inf:
            parent_guesses = explore_upwards(node_ix, parent_guesses, probMatrix)
    if np.inf in parent_guesses:
        raise ValueError('not guessing parent for someone')
    return parent_guesses


def sample_altUpwards(outsOrPreds):
    # search up
    probMatrix_list = extract_probMatrices(outsOrPreds)
    pi_trees = []
    for probMatrix in probMatrix_list:
        # build a pi-tree, sampling up
        pi_trees.append(get_parent_tree_upwards(probMatrix))
    return pi_trees
