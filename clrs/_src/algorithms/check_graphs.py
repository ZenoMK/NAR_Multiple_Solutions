import graphlib as gl
import networkx as nx
import numpy as np
import chex
import copy
## f[0][0][1].data to get adjacency matrix from next(sampler) where sampler=test_samplers[0]

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


def check_valid_dfsTree(np_input_array, pi):
    '''checks: acyclic, dangling, edge-validity, and valid-start'''
    pi = copy.deepcopy(pi) # copy pi. make sure don't mess with logging
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
    pi = np.array(pi).astype(int)
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
                if self_reachable_by_earlier_node:
                    return False
        else:  # not self-loop, so not a restart, make sure parents are reachable by lowest ix node from which i is reachable
            for j in range(i):
                if nx.has_path(g, j, i):  # lowest_ix node from which i is reachable
                    if not nx.has_path(g, j, pi[i]):  # parent not reachable. Bad! (note nx considers each node as having a path to itself)
                        return False
                    else:
                        break  # this parent is ok! advance to outermost for loop
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


#print(is_acyclic(acyclic_adj, acyclic_pi))
#print(is_acyclic(cyclic_adj, cyclic_pi))

#print(is_acyclic(disconnect_adj, disconnect_pi))



##################################################################################################################
# BELLMAN-FORD CHECKER
##################################################################################################################
# Test whether model's pi represents valid bellman-ford



def bellman_ford_cost(A, s):
  """Bellman-Ford's single-source shortest path (Bellman, 1958).
  This has been taken and adapted from the original CLRS Bellman-Ford implementation"""


  chex.assert_rank(A, 2)

  A_pos = np.arange(A.shape[0])

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = 0
  msk[s] = 1
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
            d[v] = prev_d[u] + A[u, v]
            pi[v] = u
          msk[v] = 1
    if np.all(d == prev_d):
      break
  return d

def check_valid_BFpaths(A,s, parentpath):
    ''' 1. Computes true shortest path costs,
        2. Builds BF parent-tree subgraph of graph, computes shortest path costs on subgraph (model's paths)
    Note self-parent (pi[t] = t) is the default for no path s->t. '''

    true_costs = bellman_ford_cost(A,s)
    parentpath = np.array(parentpath).astype(int)

    # the adjacency matrix of the BF tree
    BF_tree_adj = np.zeros((len(parentpath),len(parentpath)))
    for i in range(len(parentpath)):
        #if parentpath[i] == i and i != s: # shortest path. forbids neg. weight cycle solutions
        #    return False
        #breakpoint()
        if A[parentpath[i],i] == 0 and parentpath[i] != i: # you're allowed to pick self-parents for unreachable nodes
            return False
        else:
            BF_tree_adj[parentpath[i],i] = A[parentpath[i],i]

    model_costs = bellman_ford_cost(BF_tree_adj, s)

    if (true_costs == model_costs).all():
        return True
    else:
        return False

test_graph = np.array([[0,1,1],
              [0,0,1],
              [0,0,0]])
s = 0
true_parentpath = [0,0,0]
false_parentpath = [0,0,1]

assert check_valid_BFpaths(test_graph,s,true_parentpath)
assert not check_valid_BFpaths(test_graph,s,false_parentpath)

d = np.array([
    [0., 0.],
    [0., 0.]])

s = 0

expect_d = [0,1]
assert check_valid_BFpaths(d,s,expect_d)