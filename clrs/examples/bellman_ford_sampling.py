import numpy as np


## Variant that selects keepset based on path cost
def bellmanford_beamsearch(A,s,probMatrix,beamwidth = 3):
    '''
    Beamsearch sampler given a probmatrix returned by Bellman-Ford
    :param A: adjacency matrix
    :param s: source node
    :param probMatrix: model output
    :param beamwidth: the number of candidate solutions at any point
    :return: sampled parent tree
    '''

    pi = np.zeros(len(probMatrix))

    # make source its own parent
    pi[s] = s

    # assign parent to every node
    for i in range(len(probMatrix)):
        # compute path to i
        best_path_stemming_from_s = None
        if i != s:
            # sample the beam
            candidates = [chooseUniformly(probMatrix[i]) for j in range(beamwidth)]
            for k in range(len(probMatrix)): # if there is a path, there is a path within V steps
                # sample the beam of the candidates:
                # consider the beam-many most-likely parents of current candidates
                candidate_parents = []
                for h in candidates:
                    candidate_parents.append([chooseUniformly(probMatrix[h]) for j in range(beamwidth)])
                # choose the three best paths
                cost = [A[candidates[l],i] + A[candidate_parents]




def costOfPath(A,path):
    cost = 0
    for i in range(1, len(path)):
        cost += A[path[i-1], path[i]] # edge cost from path-1 to path
    return cost

## Variant that selects keepset based on max(multiplied probability along path)
def bellmanford_beamsearch(A, s, probMatrix, beamwidth=3):
    '''
    Beamsearch sampler given a probmatrix returned by Bellman-Ford
    :param A: adjacency matrix
    :param s: source node
    :param probMatrix: model output
    :param beamwidth: the number of candidate solutions at any point
    :return: sampled parent tree
    '''

    pi = np.zeros(len(probMatrix))

    # make source its own parent
    pi[s] = s

    # assign parent to every node
    for i in range(len(probMatrix)):
        # compute path to i
        best_path_stemming_from_s = None
        if i != s:
            # try to build path s -> i, by backtracking from i
            # DATA STRUCTURE: tuples of (candidate_path, candidate_path_cost)
            candidate_back_paths = [[i] for j in range(beamwidth)] # candidates are paths in reverse order, terminating in i
            for candidate_back_path in candidate_back_paths:
                newParent = chooseUniformly(probMatrix[i])
            # ^ beginning with parent of i
            costs = [costOfPath(A, candidate) for candidate in candidates]
            for k in range(len(probMatrix)): # consider paths of length up-to V. if there is a path, there is a path within V steps.
                # consider beam-many parents of each of the current candidates
                candidate_parents = []
                for h in candidates:
                    # create pathlists where we append newest node to prev. path. Cost is +=
                    candidate_parents.append([chooseUniformly(probMatrix[h]) for j in range(beamwidth)])
                    # select the 3 best to continue with, repeat
                # choose the three best paths
                cost = [A[candidates[l], i] + A[candidate_parents]

## Deterministic variant that considers all parents from selects keepset based on max(multiplied probability along path)
                # would that be any different from argmax?

## beamsearch is deterministic...
