#
#
# CHECK THE DFS implementation and the checkvaliddfstree
#


import graphlib as gl
import networkx as nx
import numpy as np


# cyclic graph to test
cyclic_adj = np.array([
    [0,1,0],
    [0,0,1],
    [1,0,0]
])
cyclic_pi = [2,0,1]

acyclic_adj = np.array([
    [0,1],
    [0,0]
])
acyclic_pi = [0, 0]

disconnect_adj = np.array([
    [0,1,0],
    [0,0,0],
    [0,0,0]
])
disconnect_pi = [0,0,2]

edge_to_zero_adj = np.array([
    [0,0],
    [1,0]
])
edge_to_zero_pi = [0,1]

pullback = np.array([
    [0,0,1],
    [0,0,1],
    [0,0,0]
])
pullback_pi = [0,1,0]
bad_pullback_pi = [0,1,1]

################################################################
# DFS CHECK
################################################################
def check_valid_dfsTree(np_input_array, pi):
    '''checks: acyclic, dangling, edge-validity, and valid-start'''
    pi = pi[:] # copy pi. make sure don't mess with logging
    if pi[0] == 0: # correct start-node
        if are_valid_edges_parents(np_input_array, pi):
            if are_valid_order_parents(np_input_array, pi): # self-loops not reachable by lower_ix. Other parents reachable by ...
                pi = replace_self_loops_with_minus1(pi)
                if is_acyclic(pi): # no funky hiding parents. Should be implied by lower-node reachability.
                    return True
                else:
                    print('cycle')
            else:
                print('oo')
        else:
            print('not edges')
    else:
        print('wrong startnode')
    return False


def replace_self_loops_with_minus1(pi):
    for i in range(len(pi)):
        if pi[i] == i:
            pi[i] = -1
    return pi

def are_valid_edges_parents(np_input_array, pi):
    for i in range(len(pi)): # for node in graph.
        parent = pi[i]
        if parent != i: # Not a restart, check edge parent->child. (assume restarts are always valid, check later).
            #breakpoint()
            try:
                if np_input_array[parent][i] == 0: # no edge parent -> child
                    return False
            except:
                breakpoint()
    return True

def are_valid_order_parents(np_input_array, pi):
    """
    Checks whether self-loops stem from valid DFS execution.
        If you were reachable_by_earlier_node, but have self-loop, it's a problem
    :param input:
    :param pi:
    :return:
    """
    g = nx.from_numpy_array(np_input_array, create_using=nx.DiGraph)
    for i in range(len(pi)):
        if pi[i] == i:
            for j in range(i): # crucially, this does not run when i=0, so the starting 0,0 self-loop always permitted
                self_reachable_by_earlier_node = nx.has_path(g, j, i) # does this do it directed? lower-triangle?
                #breakpoint()
                if self_reachable_by_earlier_node:
                    return False
        else: # not self-loop, so not a restart, make sure parents are reachable by lowest ix node from which i is reachable
            for j in range(i):
                if nx.has_path(g,j,i): # lowest_ix node from which i is reachable
                    if not nx.has_path(g, j, pi[i]): # parent not reachable. Bad! Note nx considers each node as having a path to itself
                        return False
                    else:
                        break # this parent is ok! advance to outermost for loop
    return True



def is_acyclic(pi):
    """
    Function to check for cycles in a predecessor array returned by the model
    :param input: the adjacency matrix of the graph on which the model has inferred the predecessor array
    :param pi: the predecessor array
    :return: Boolean indicating acyclicity
    """
    ts = gl.TopologicalSorter()
    for i in range(len(pi)):
        ts.add(i, pi[i])
    try:
        ts.prepare()
        return True
    except ValueError as e:
        if isinstance(e, gl.CycleError):
            #print("I am a cycle error")
            return False
        else:
            raise e




################################################################
# DFS CHECK
################################################################
path = np.array([
    [0,1,0],
    [0,0,1],
    [0,0,0]
])

branch = np.array([
    [0,1,1],
    [0,0,0],
    [0,0,0]
])

diamond = np.array([
    [0,1,1,0],
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,0]
])

longshort = np.array([
    [0,1,0,0,1],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
])

def dfs(A):
    """Depth-first search (Moore, 1959)."""
    color = np.zeros(A.shape[0], dtype=np.int32)
    pi = np.arange(A.shape[0])
    s_prev = np.arange(A.shape[0]) # backtrack locations
    shuffled = np.arange(A.shape[0])
    np.random.shuffle(shuffled)
    for s in range(A.shape[0]):
        if color[s] == 0:
            s_last = s
            u = s
            v = s

            while True:
                if color[u] == 0:
                    color[u] = 1

                for v in shuffled:
                    if A[u, v] != 0:
                        if color[v] == 0:
                            pi[v] = u
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v
                            break

                if s_last == u: # no further edges
                    # for reading the below, remember you're s_last
                    color[u] = 2

                    if s_prev[u] == u: #
                        assert s_prev[s_last] == s_last
                        break

                    pr = s_prev[s_last]     # temp value, node b4 you
                    #s_prev[s_last] = s_last # remember you were here. You're your-own prev. Never comes-up. If you're backtracked-from you're marked, so never start, and you've exhausted outgoing edges, so no one will be back-tracking to you
                    s_last = pr             # prepare to backtrack

                # move to next node (deeper if found one, else to s_prev)
                u = s_last

    return pi


################################################################
# GRAPH TESTER
################################################################

def _random_er_graph(seed, nb_nodes, p=0.5, directed=False, acyclic=False,
                   weighted=False, low=0.0, high=1.0):
    """Random Erdos-Renyi graph."""
    rng = np.random.RandomState(seed)

    mat = rng.binomial(1, p, size=(nb_nodes, nb_nodes))
    if not directed:
      mat *= np.transpose(mat)
    elif acyclic:
      mat = np.triu(mat, k=1)
      p = rng.permutation(nb_nodes)  # To allow nontrivial solutions
      mat = mat[p, :][:, p]
    if weighted:
      weights = rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
      if not directed:
        weights *= np.transpose(weights)
        weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
      mat = mat.astype(float) * weights
    return mat




if __name__ == '__main__':
    seed = 42
    for i in range(1000):
        a = _random_er_graph(seed, 4)
        seed+=1
        print(dfs(a))
        if not check_valid_dfsTree(np_input_array=a, pi=dfs(a)):
            breakpoint()


