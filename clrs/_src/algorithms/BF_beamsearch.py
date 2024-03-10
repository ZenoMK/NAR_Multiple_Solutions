import numpy as np
import clrs._src.dfs_sampling as dfs_sampling

def BF_beamsearch(A, s, probMatrix, beamwidth=3):
    """
    Beamsearch sampler given a probmatrix returned by Bellman-Ford
    :param A: adjacency matrix
    :param s: source node
    :param probMatrix: model output
    :param beamwidth: the number of candidate solutions at any point
    :return: sampled parent tree
    """

    pi = np.zeros(len(probMatrix))

    # make source its own parent
    pi[s] = s

    # assign parent to every node
    for i in range(len(probMatrix)):
        # compute path to i
        if i != s:

            # sample the beam
            candidates = [dfs_sampling.chooseUniformly(probMatrix[i]) for j in range(beamwidth)]
            candidate_back_paths = [[i, candidates[j]] for j in range(beamwidth)]
            costs = [A[candidate,i] for candidate in candidates]

            for k in range(len(probMatrix)):
                # sample the beam of the candidates
                candidate_parents = []
                for h in candidates:
                    candidate_parents = [dfs_sampling.chooseUniformly(probMatrix[h]) for j in range(beamwidth)]
                    candidate_back_paths = [[h, candidate_parent] for candidate_parent in candidate_parents]
                    costs = [A[candidate, i] for candidate in candidates]

                    candidate_parents.append([dfs_sampling.chooseUniformly(probMatrix[h]) for j in range(beamwidth)])
                # choose the three best paths
                cost = [A[candidates[l], i] + A[candidate_parents]