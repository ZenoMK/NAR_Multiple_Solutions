import graphlib as gl
import networkx as nx

## f[0][0][1].data to get adjacency matrix from next(sampler) where sampler=test_samplers[0]

# cyclic graph to test
cyclic_adj = [
    [0,1,0],
    [0,0,1],
    [1,0,0]
]
cyclic_pi = [2,0,1]

acyclic_adj = [
    [0,1],
    [0,0]
]
acyclic_pi = [0, 0]

def is_acyclic(input, pi):
    """
    Function to check for cycles in a predecessor array returned by the model
    :param input: the adjacency matrix of the graph on which the model has inferred the predecessor array
    :param pi: the predecessor array
    :return: Boolean indicating acyclicity
    """

    # if self-loop: is i reachable from a lower-indexed node?
        # if yes: return false
        # if no: replace its parent by the god node -1

    # Build networkit graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(pi)))
    for i in range(len(pi)):
        for j in range(len(pi)):
            if input[i][j] == 1:
                graph.add_edge(i,j)
    # no self-loop on the start node
    #pi[0] = -1

    # check self-looping conditions
    if is_valid_self_loops(input, pi):
        for i in range(len(pi)):
            if pi[i] == i:
                pi[i] = -1
    else:
        return False

    print(pi)





    ts = gl.TopologicalSorter()
    for i in range(len(pi)):
        ts.add(i, pi[i])
    try:
        ts.prepare()
        return True
    except ValueError as e:
        if isinstance(e, gl.CycleError):
            print("I am a cycle error")
            return False
        else:
            raise e


def is_valid_self_loops(input, pi):
    """
    Checks whether self-loops stem from valid DFS execution.
    :param input:
    :param pi:
    :return:
    """
    for i in range(len(pi)):
        if pi[i] == i:
            for j in range(i):
                paths = nx.has_path(input, j, i)
                simple_paths = paths.numberOfSimplePaths()
                if simple_paths > 0:
                    return False
    return True

print(is_acyclic(acyclic_adj, acyclic_pi))
print(is_acyclic(cyclic_adj, cyclic_pi))
