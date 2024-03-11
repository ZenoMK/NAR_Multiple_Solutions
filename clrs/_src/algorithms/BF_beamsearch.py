import numpy as np
import clrs._src.dfs_sampling as dfs_sampling
#import graphs as graphs


def BF_beamsearch(A, s, probMatrix, beamwidth=3):
    """
    Beamsearch sampler given a probmatrix returned by Bellman-Ford
    :param A: adjacency matrix
    :param s: source node
    :param probMatrix: model output
    :param beamwidth: the number of candidate solutions at any point
    :return: sampled parent tree
    """
    # optimizations possible, keep low-cost shorter paths over extending to bad parents
    # flip-coin for tie-breaking equal-cost kept-parents
    # tune-beam
    # sample without replacement
    try:
        pi = np.zeros(len(probMatrix))

        # make source its own parent
        pi[s] = s

        # assign parent to every node
        for i in range(len(probMatrix)):
            # compute path to i
            if i != s:

                # paths terminate in i
                candidates_rev = [[i] for i in range(beamwidth)] # paths in reverse order. [a,b,c] indicates [c->b->a]
                candidates_cost = [0 for i in range(beamwidth)]
                best_path_cost = np.inf
                best_path_stemming_from_s = None

                for k in range(len(probMatrix)): # try paths of length up-to |V|, number of vertices
                    longer_paths = [] # list of paths, each path in reverse order
                    longer_path_costs = [] # list of path costs for new longer candidate paths

                    # Explore beam-many parents for each of the beam-many candidates
                    for candidate_ix in range(len(candidates_rev)):
                        print('ci', candidate_ix)
                        candidate_path = candidates_rev[candidate_ix]
                        print('cp-1', candidate_path)
                        highest_node = candidate_path[-1] # most recent node added, conceptually the progenitor of path
                        parent_probs = probMatrix[highest_node]

                        # Extend candidate path by new parent, calculate cost
                        # Store new path grown from this candidate, and its associated cost
                        for new_path_num in range(beamwidth):
                            candidate_parent = dfs_sampling.chooseUniformly(parent_probs)
                            # extend & store path
                            new_path = np.append(candidate_path, candidate_parent) # concatenate parent to h, conceptually, adding a parent to path progenitor
                            longer_paths.append(new_path)
                            # calculate & store cost
                            cost_of_new_edge = A[candidate_parent, highest_node]
                            if cost_of_new_edge == 0: # edge not in OG graph
                                cost_of_new_edge = np.inf
                            prev_cost = candidates_cost[candidate_ix]
                            new_cost = prev_cost + cost_of_new_edge
                            longer_path_costs.append(new_cost)
                            if candidate_parent == s:
                                break

                    # If any path begins with source node s, and has lower cost than current best path from s,
                    # save it. (remember, paths in target->source order, so path[-1] is starting node)
                    for path_ix in range(len(longer_paths)):
                        path = longer_paths[path_ix]
                        if path[-1] == s:
                            path_cost = longer_path_costs[path_ix]
                            if path_cost < best_path_cost:
                                best_path_stemming_from_s = path

                    # Select the (beam width)-many best paths (lowest weight in original graph); explored further next loop.
                    path_ixs_by_lowest_cost = np.argsort(longer_path_costs)
                    best_beamwidth_ixs = path_ixs_by_lowest_cost[:beamwidth]
                    candidates_rev = np.array(longer_paths)[best_beamwidth_ixs] # select paths for next-round according to best beam-many cost-minimizing indices

            if best_path_stemming_from_s is not None:
                pi[i] = best_path_stemming_from_s[1] # node before i on best_path_found
            else:
                print('no good path')
                #breakpoint() #oops! no good path
    except:
        print('other error')
        #breakpoint()
    print(pi)
    return pi


def BF_greedysearch():
    pass




if __name__ == '__main__':
    #testexample
    pM = np.array([[6.7942673e-03, 1.8254748e-06, 5.3676311e-04, 5.6340452e-04,
            9.9210370e-01],
           [8.2969433e-04, 9.4034225e-01, 5.7718944e-02, 2.6801512e-05,
            1.0822865e-03],
           [1.4186887e-01, 2.8001370e-02, 8.2659137e-01, 5.6764267e-05,
            3.4815797e-03],
           [8.1846189e-01, 3.9187955e-05, 9.7058117e-05, 1.7738120e-01,
            4.0207091e-03],
           [3.3088622e-03, 1.5012523e-07, 3.4113563e-07, 2.7682628e-07,
            9.9669027e-01]])

    pM_1 = np.array([[0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0.]])

    A = np.array([[1., 0., 1., 1., 1.],
                 [0., 0., 1., 0., 0.],
                 [1., 1., 0., 0., 0.],
                 [1., 0., 0., 1., 0.],
                 [1., 0., 0., 0., 1.]])

    s = 3

    result = BF_beamsearch(A,s,pM_1)
    correct = graphs.bellman_ford(A,s)
    print(correct)