import numpy as np
import clrs._src.dfs_sampling as dfs_sampling

def sample_beamsearch(As, Ss, outsOrPreds):
    probMatrix_list = dfs_sampling.extract_probMatrices(outsOrPreds)
    pi_trees = []
    for ix in range(len(probMatrix_list)):
        #breakpoint()
        A = As[ix]
        startNode = Ss[ix]
        probMatrix = probMatrix_list[ix]
        # build a pi-tree, sampling beam
        pi_trees.append(BF_beamsearch(A, startNode, probMatrix))
    return pi_trees
##
# Two mostly-equivalent methods: beamsearch and BF_beamsearch. BF_beamsearch seems to return leastcostpathparents as ints, beamsearch as floats
##
def beamsearch(A, s, probMatrix, beamwidth=3):
    """
    FIXME: AUG25 MAYBE DOESNT WORK AND UNUSED. INSTEAD USE BF_beamsearch... ORELSE JUST REQUIRES np.arrays

    nicely decomposed. calls path_to_i for each node. returns parent tree.
    LeastCost path parent tree is sufficient for all leastCost paths,  since if leastCost path s->pi[t]->t didn't use leastCost path s->pi[t], there would be a lowerCost path s->t using s->pi[t]
    """
    vertices = range(len(probMatrix))
    pi = np.zeros(len(vertices))
    pi[s] = s
    for t in vertices:
        if t != s:
            least_cost_path = beamsearch_least_cost_path(A, s, t, probMatrix, beamwidth)
            if least_cost_path is not None:
                pi[t] = least_cost_path[1] # paths in reverse order [a,b,c] means [c->b->a]. Path ends at t: [t,parent,...]
            else:
                pi[t] = t # consistent with samplers.py implementation, parent defaults to self, which can't be a shortest path parent (no neg. cycles). Condition caught in check_graphs
    return pi

def beamsearch_least_cost_path(A, s, t, probMatrix, beamwidth):
    '''compute path s->t. Note we go backwards, reconstructing path t->s'''
    # paths terminate in t
    path_guesses = [[t] for i in range(beamwidth)]  # paths in reverse order. [a,b,c] indicates [c->b->a]
    path_costs = [0 for i in range(beamwidth)]
    best_path_cost = np.inf
    best_path_from_s = None

    for path_length in range(1, len(probMatrix)):  # try paths of length up-to |V|, number of vertices
        longer_paths = []  # list of paths, each path in reverse order
        longer_path_costs = []  # list of path costs for new longer candidate paths

        # get paths of length path_length
        longer_paths, longer_path_costs = beamsearch_extend_paths(A, probMatrix, beamwidth, path_guesses, path_costs)
        #breakpoint()
        # If any path begins with source node s, and has lower cost than current best path from s,
        # save it. (remember, paths in target->source order, so path[-1] is starting node)
        best_path_from_s, best_path_cost = select_best_path_from_s(s, best_path_from_s, best_path_cost, longer_paths, longer_path_costs)
        #breakpoint()
        # filter for next iteration
        # Select the (beam width)-many best paths (lowest weight in original graph); explored further next loop.
        path_ixs_by_lowest_cost = np.argsort(longer_path_costs)
        best_3_ixs = path_ixs_by_lowest_cost[:beamwidth]
        path_guesses = np.array(longer_paths)[best_3_ixs]  # select paths for next-round according to best beam-many cost-minimizing indices
        #breakpoint()

    return best_path_from_s

def beamsearch_extend_paths(A, probMatrix, beamwidth, path_guesses, path_costs):
    #add_parent_of_source_to_paths
    # Explore beam-many parents for each of the beam-many candidates
    longer_paths = []  # list of paths, each path in reverse order
    longer_path_costs = []  # list of path costs for new longer candidate paths

    for path_ix in range(len(path_guesses)):
        #print('ci', candidate_ix)
        path = path_guesses[path_ix]
        path_cost = path_costs[path_ix]
        #print('cp-1', candidate_path)
        highest_node = path[-1]  # most recent node added, conceptually the progenitor of path
        parent_probs = probMatrix[highest_node]

        for new_path_num in range(beamwidth):
            new_path, new_path_cost = grow_path_by_parent_probs(A, path, path_cost, parent_probs)
            # Store new path grown from this candidate, and its associated cost
            longer_paths.append(new_path)
            longer_path_costs.append(new_path_cost)

    return longer_paths, longer_path_costs


def grow_path_by_parent_probs(A, path, path_cost, parent_probs):
    # Extend candidate path by new parent, calculate cost
    new_parent = dfs_sampling.chooseUniformly(parent_probs)

    # extend path
    new_path = np.append(path, new_parent)  # concatenate parent to h, conceptually, adding a parent to path progenitor

    # calculate cost
    breakpoint()
    cost_of_new_edge = A[new_parent, path[-1]]
    if cost_of_new_edge == 0:  # edge not in OG graph
        #breakpoint()
        cost_of_new_edge = np.inf
    new_cost = path_cost + cost_of_new_edge

def select_best_path_from_s(s, best_path, best_cost, new_paths, new_costs):
    assert len(new_paths) == len(new_costs)
    for ix in range(len(new_paths)):
        path = new_paths[ix]
        if path[-1] == s:
            if new_costs[ix] < best_cost:
                #breakpoint()
                best_path = path
                best_cost = new_costs[ix]
    #breakpoint()
    return best_path, best_cost


########################################################################
# All-in-one. Seems works!
########################################################################

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
    #try:
    pi = np.arange(len(probMatrix))

    # make source its own parent
    pi[s] = s

    # assign parent to every node
    for t in range(len(probMatrix)):
        # compute path to i
        if t != s:

            # paths terminate in i
            candidates_rev = [[t] for i in range(beamwidth)] # paths in reverse order. [a,b,c] indicates [c->b->a]
            candidates_cost = [0 for i in range(beamwidth)]
            best_path_cost = np.inf
            best_path_stemming_from_s = None
            #breakpoint()

            for k in range(len(probMatrix)): # try paths of length up-to |V|, number of vertices
                longer_paths = [] # list of paths, each path in reverse order
                longer_path_costs = [] # list of path costs for new longer candidate paths

                # Explore beam-many parents for each of the beam-many candidates
                for candidate_ix in range(len(candidates_rev)):
                    #breakpoint()
                    #print('ci', candidate_ix)
                    candidate_path = candidates_rev[candidate_ix]
                    #print('cp-1', candidate_path)
                    highest_node = candidate_path[-1] # most recent node added, conceptually the progenitor of path
                    parent_probs = probMatrix[highest_node]

                    # Extend candidate path by new parent, calculate cost
                    # Store new path grown from this candidate, and its associated cost
                    for new_path_num in range(beamwidth):
                        #breakpoint()
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

                # If any path begins with source node s, and has lower cost than current best path from s,
                # save it. (remember, paths in target->source order, so path[-1] is starting node)
                for path_ix in range(len(longer_paths)):
                    path = longer_paths[path_ix]
                    if path[-1] == s:
                        path_cost = longer_path_costs[path_ix]
                        if path_cost < best_path_cost:
                            best_path_stemming_from_s = path
                            best_path_cost = path_cost

                # Select the (beam width)-many best paths (lowest weight in original graph); explored further next loop.
                path_ixs_by_lowest_cost = np.argsort(longer_path_costs)
                best_3_ixs = path_ixs_by_lowest_cost[:beamwidth]
                candidates_rev = np.array(longer_paths)[best_3_ixs] # select paths for next-round according to best beam-many cost-minimizing indices

            if best_path_stemming_from_s is not None:
                pi[t] = best_path_stemming_from_s[1] # node before i on best_path_found
            else:
                print('BF_beamsearch.py, no good path', s, '->', t)
                #breakpoint() #oops! no good path
    #except:
    #    print('other error')
    #    breakpoint()

    return pi


def sample_greedysearch(As, Ss, outsOrPreds):
    probMatrix_list = dfs_sampling.extract_probMatrices(outsOrPreds)
    pi_trees = []
    for ix in range(len(probMatrix_list)):
        A = As[ix]
        startNode = Ss[ix]
        probMatrix = probMatrix_list[ix]
        # build a pi-tree, sampling beam
        pi_trees.append(BF_greedysearch(A, startNode, probMatrix))
        #breakpoint()
    return pi_trees

def BF_greedysearch(A, s, probMatrix, beamwidth=3):
    pi = np.zeros(len(probMatrix))
    pi[s] = s

    # sample parents for each non-source node
    for i in range(len(probMatrix)):
        if i != s:
            # sample candidate parents, ensure at least one parent is plausible (there exists an edge (parent,i))
            candidates_costs = np.full(beamwidth, np.inf)
            counter = 0
            while (candidates_costs == np.full(len(candidates_costs), np.inf)).all() and counter < 10:
                candidates = [dfs_sampling.chooseUniformly(probMatrix[i]) for j in range(beamwidth)]
                candidates_costs =[A[candidate, i] for candidate in candidates]
                # remove any parents without any edges to i
                for k in range(len(candidates_costs)):
                    if candidates_costs[k] == 0:
                        candidates_costs[k] = np.inf
                counter += 1

            # choose lowest-cost parent
            pi[i] = candidates[np.argmin(candidates_costs)]
    return pi






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


    #confusion = beamsearch(A,s,pM)
    #confusionprime = BF_beamsearch(A,s,pM)

    easy_PM_p3 = np.array([
        [1,0,0],
        [1,0,0],
        [0,1,0]
    ])
    easy_A_p3 = np.array([
        [0,1,0],
        [0,0,1],
        [0,0,0]
    ])
    easy_s = 0

    expect_easy = [0,0,1]

    a = beamsearch(easy_A_p3, easy_s, easy_PM_p3)
    aprime = BF_beamsearch(easy_A_p3, easy_s, easy_PM_p3)
    assert (a == aprime).all()
    assert (a == expect_easy).all()

    cycle_A = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    cycle_pm = np.array([
        [0,0,1],
        [1,0,0],
        [0,1,0]
    ])

    cycle_s = 1
    cycle_expect = [2,1,1]

    c = beamsearch(cycle_A, cycle_s, cycle_pm)
    cprime = BF_beamsearch(cycle_A, cycle_s, cycle_pm)
    assert(c == cprime).all()
    assert(c == cycle_expect).all()

    disconnect_A = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ])
    disconnect_pm = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ])

    disconnect_s = 2
    disconnect_expect = [0,1,2]

    d = beamsearch(disconnect_A, disconnect_s, disconnect_pm)
    dprime = BF_beamsearch(disconnect_A, disconnect_s, disconnect_pm)
    assert (d == dprime).all()
    assert (d == disconnect_expect).all()
    result = BF_greedysearch(A,s,pM_1)
    correct = graphs.bellman_ford(A,s)
    print(correct)
    print(result)
    print(check_graphs.check_valid_BFpaths(A,s,result))
