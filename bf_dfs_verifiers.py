# Jan 27
# implement Henry and Zeno's algorithms for testing DFS-verification to catch any slip-ups with code

# generate random graphs
# dfs those graphs to create forests
# run verification algorithms on forests to ensure all real forests are marked good

# create deliberately bad forests? or just random forests?
# run verification algorithms to ensure fake forests are marked bad
#   **there's a chance you randomly make a good forest, save those for manual investigation**

import networkx as nx               # graph library
import matplotlib.pyplot as plt     # visualize nx graphs
import random                       # shuffle lists
import numpy as np                  # adjacency matrices


# TODO: what goes wrong when you make Treedown exclude green nodes?
# TODO: rigorously test notgreen Treedown (needed for scratch ex.1 to be true)
    # you can build a tough example for henry and test it

# ---------------------------------------------------------------------------------------------------------------------
# -- Generating Random Graphs -- #
# ---------------------------------------------------------------------------------------------------------------------
random.seed(123)
NUM_TO_TEST = 10**3

def er_graph(n, p):
    # Generate random graph
    G = nx.erdos_renyi_graph(n, p, directed=True)
    return G


def random_subgraphs(G,p,n):
    return [random_subgraph(G,p) for i in range(n)]

def random_subgraph(G, p):
    G = nx.DiGraph(G)
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(G.nodes())
    # keep each edge with probability p
    random_edges = []
    for edge in G.edges():
        if random.random() < p:
            random_edges.append(edge)
    # Create a new subgraph with only these edges
    subgraph.add_edges_from(random_edges)
    return subgraph

def random_forestish():
    # ensure no cycles, make it a forest
    return


def random_forest_subgraph(graph, p):
    """Create a subgraph where edges are randomly removed, then extract spanning trees."""
    # Step 1: Keep each edge with probability p
    subgraph = nx.Graph() if not graph.is_directed() else nx.DiGraph()
    subgraph.add_nodes_from(graph.nodes())  # Keep all nodes

    for u, v in graph.edges():
        if random.random() < p:  # Keep edge with probability p
            subgraph.add_edge(u, v)

    # Step 2: Extract a spanning tree from each connected component
    forest = nx.Graph()  # A forest is an undirected graph with multiple trees
    forest.add_nodes_from(subgraph.nodes())  # Keep all nodes

    for component in nx.connected_components(subgraph):  # Find connected components
        tree = nx.minimum_spanning_tree(subgraph.subgraph(component))  # Get a spanning tree
        forest.add_edges_from(tree.edges())  # Add tree edges to the forest

    return forest


# ---------------------------------------------------------------------------------------------------------------------
# -- DFS with Ordered Restarts -- #
# ---------------------------------------------------------------------------------------------------------------------


def dfs(graph):
    visited = set()
    stack = []  # Start with the initial node
    traversal = []        # Store the traversal order
    all_nodes = {i for i in range(len(graph))}

    while len(visited) != len(graph):
        stack.append(min(all_nodes - visited))  # restart @ smallest unseen node
        while stack:
            node = stack.pop()  # Pop a node from the stack
            if node not in visited:
                visited.add(node)
                traversal.append(node)

                # Add neighbors to the stack in shuffle order
                nbrs = list(graph.neighbors(node))
                stack.extend(random.sample(nbrs, len(nbrs)))    # random.sample here shuffles nbrs for random edge explore

    return traversal

def forest_from_traversal(G, O):
    '''discoverer is closest preceding in order with edge?'''
    #print('assuming order and graph uniquely determine tree')
    forest = np.zeros((len(G), len(G)))
    back = list(reversed(O))
    for ix in range(len(back)):
        for jx in range(ix, len(back)):
            i = back[ix]
            j = back[jx]
            if (j,i) in G.edges:
                # i discovers j
                forest[j, i] = 1
                break
    return forest



# ---------------------------------------------------------------------------------------------------------------------
# -- Henry's O(n^4) -- #
# ---------------------------------------------------------------------------------------------------------------------
# fixme: how does regular henry deal with stuff like the false negative agnostic screenshot

def agnostic_henry(G,F):
    """supposed to be henry but order doesnt matter?""" # fixme: gotta start at possible roots (no parents)
    if isinstance(F, (list, np.ndarray)): # check if it's a parent tree format, convert
        if np.ndim(F) == 1:
            F = no_self_loops_parent_tree_to_adj_matrix(F)
    if not preprocess(F):
        return False
    if isinstance(G, np.ndarray):
        G = nx.from_numpy_array(G, create_using=nx.DiGraph)
    if isinstance(F, np.ndarray):
        F = nx.from_numpy_array(F, create_using=nx.DiGraph)
    colors = np.array([0] * len(G))
    # find all rooties, (no incoming edges)
    possible_starts = [node for node, deg in F.in_degree() if deg == 0] #np.where(np.all(adj_matrix == 0, axis=0))
    #print('poss starts', possible_starts)
    i = 0
    while i < len(possible_starts): # only if you go through all your options and none can be good, its over
        node = possible_starts[i]
        old_colors = np.copy(colors)
        if ccv(G, F, node, colors):
            if (old_colors != colors).any(): # new change, recheck
                i=-1 # will be 0
        i+=1
    #print('colors at end', colors)
    return (colors == [1]*len(G)).all()


# ---------------------------------------------------------------------------------------------------------------------
# -- Henry's O(n^4) -- #
# ---------------------------------------------------------------------------------------------------------------------
#
def parent_tree_to_adj_matrix(tree):
    size = len(tree)    # n_vertices
    M = np.zeros((size, size))
    for ix in range(size):
        M[int(tree[ix]), ix] = 1     # edge points tree[ix] to ix, bcuz parent tree
    return M
def no_self_loops_parent_tree_to_adj_matrix(tree):  # FIXME: duplicate code in validate_distributions cuz im lazy
    """now root is just any node without parent"""
    M = parent_tree_to_adj_matrix(tree)
    np.fill_diagonal(a=M, val=0)
    return M

def preprocess(adj_matrix):
    """ensure it's acyclic, nodes have at most 1 parent"""
    if isinstance(adj_matrix, nx.Graph):
        adj_matrix = nx.to_numpy_array(adj_matrix)
    n = len(adj_matrix)
    # Step 1: Count incoming edges for each node
    in_degrees = np.sum(adj_matrix, axis=0)
    # Step 2: Ensure nodes have no more than 1 parent
    if not np.all((in_degrees == 1) | (in_degrees == 0)):
        return False  # Each node must have either 0 (root) or 1 incoming edge
    # Step 3: Ensure acyclic
    F = nx.DiGraph(adj_matrix)

    # TODO tell John about this
    return nx.is_forest(F)


def henry(G, F):
    if isinstance(F, (list, np.ndarray)): # check if it's a parent tree format, convert
        if np.ndim(F) == 1:
            F = no_self_loops_parent_tree_to_adj_matrix(F)
    if not preprocess(F):
        return False
    if isinstance(G, np.ndarray):
        G = nx.from_numpy_array(G, create_using=nx.DiGraph)
    if isinstance(F, np.ndarray):
        F = nx.from_numpy_array(F, create_using=nx.DiGraph)
    colors = [0] * len(G)
    for i in range(len(G)):  # outer restart loop
        if not ccv(G, F, i, colors):
            #print(f'fails on node {i}')
            return False
    return True


def notgreen(kids, color):
    '''kids is a list of node ix'''
    return [kid for kid in kids if color[kid] != 1]


def down(G, source, colors):
    '''return set of all descendents of node in G, cannot pass through green nodes'''
    visited = set()
    descendants = set()

    def notgreen_dfs(node):
        if node in visited or colors[node] == 1:
            return
        visited.add(node)
        descendants.add(node)
        for neighbor in G.neighbors(node):
            notgreen_dfs(neighbor)

    notgreen_dfs(source)
    descendants.discard(source)  # Exclude the source itself. If source not present (green at start) does nothing
    return descendants

def notgreen_treedown(G, source, colors): # FIXME: test this bb
    '''return set of all descendents of node in G''' # testing this in tree to deal with bug? PROBLEM, accepts disconnected bois
    visited = set()
    descendants = set()

    def minidfs(node):
        if node in visited or colors[node]==1:
            return
        visited.add(node)
        descendants.add(node)
        for neighbor in G.neighbors(node):
            minidfs(neighbor)

    minidfs(source)
    descendants.discard(source)  # Exclude the source itself.
    return descendants

def treedown(G, source): # FIXME: what goes wrong if we exclude green nodes? SEEMS BETTER FOR agnostic, at least || and fine for henry??
    '''return set of all descendents of node in G''' # testing this in tree to deal with bug? PROBLEM, accepts disconnected bois
    visited = set()
    descendants = set()

    def minidfs(node):
        if node in visited:
            return
        visited.add(node)
        descendants.add(node)
        for neighbor in G.neighbors(node):
            minidfs(neighbor)

    minidfs(source)
    descendants.discard(source)  # Exclude the source itself.
    return descendants


def ccv(G, F, node, colors):
    '''check child validity: which kid can go next, like a waterslide'''
    #print('checking ', node)
    if colors[node] == 1:   # you can only be greened by your kids being valid at some point, and once you're green you stay green
        return True
    if down(G, node, colors) != notgreen_treedown(F, node, colors): #down(F, node, colors):  # base case: this node can go
        #print('Gdown', down(G, node, colors))
        #print('Tdown', notgreen_treedown(F, node, colors))
        return False
    colors[node] = 1  # passed the vibe check, green
    #print('visiting', node)
    #print(f'greening node {node}')
    #kids = G.neighbors(node)
    kids = F.neighbors(node)
    kids = notgreen(kids, colors)  # green means visited, done
    i = 0
    while i < len(kids):
        kid = kids[i]
        pd = dict()
        ad = dict()
        pd[kid] = down(G, kid, colors)  # possible descendants **through non-green paths**
        ad[kid] = treedown(F, kid) #down(F, kid, colors)  # actual descendants **should this be thru non-green? it is**
        #breakpoint()
        if pd[kid] == ad[kid]:
            # this kid can go next
            if not ccv(G, F, kid, colors):  # blow-up if things are bad in the descendants
                #print('fails on descendants')
                return False
            i = 0  # otherwise, proceed to other top-level kids
            kids = notgreen(kids, colors)  # change the loop list
        else:
            i += 1  # this one cant go first. check the next top-level kid

    return notgreen(kids, colors) == []  # all kids green we gucci, bcuz all subkids green or else return false early


# ---------------------------------------------------------------------------------------------------------------------
# -- Test on Known Graphs -- #
# ---------------------------------------------------------------------------------------------------------------------


def get_unique_adjacency_matrices(graphs):
    unique_matrices = set()
    unique_graphs = []
    #breakpoint()
    for G in graphs:
        if isinstance(G, np.ndarray):
            adj_matrix = G
        else:
            # Get the adjacency matrix
            adj_matrix = nx.to_numpy_array(G)

        # Convert matrix to a tuple of tuples (hashable format)
        matrix_tuple = tuple(map(tuple, adj_matrix))

        # Check uniqueness
        if matrix_tuple not in unique_matrices:
            unique_matrices.add(matrix_tuple)
            unique_graphs.append(G)

    return unique_graphs


def graphtest(G, verifier):
    false_approves = []
    false_rejects = []
    go, good_forests = prolly_unique_dfs(G)
    bad_graphs = random_subgraphs(G, p=0.5, n=NUM_TO_TEST)
    #breakpoint()
    #bad_forests = random_forestish(G)
    for g in good_forests.values():
        if not verifier(G,g):
            false_rejects.append(g)
    for b in bad_graphs:
        if verifier(G,b):
            false_approves.append(b)
    return get_unique_adjacency_matrices(false_approves), get_unique_adjacency_matrices(false_rejects), len(go)


def dfs_many_times(G, n_runs):
    traversal_orders = []
    trees = {}
    for i in range(n_runs):
        order = tuple(dfs(G))
        traversal_orders.append(order)
        trees[order] = forest_from_traversal(G, order)
    return traversal_orders, trees

def prolly_unique_dfs(G):
    #print('very approximate: the unique dfs trees of G. ok for small graphs (<= 6 nodes)')
    orders, trees = dfs_many_times(G, NUM_TO_TEST)
    unique_orders = set(orders)
    shortlist_trees = {order: trees[order] for order in unique_orders}
    return unique_orders, shortlist_trees


def draw(G):
    if isinstance(G, np.ndarray):
        G = nx.from_numpy_array(G, create_using=nx.DiGraph)
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)
    plt.show()


def manual_sanity_check(graphsizes, verifier_algorithm):
    """
    for each size in graphsizes,
        1. make a random graph,
        2. run graphtest(G,verfier), which tests 1000 good trees and 1000 bad trees
        3. assert no false rejects (all good trees are marked good)
        4. store `probably false accepts` for manual inspection (bad trees are just random trees, they might be correct)
    """
    outcomes = []
    for size in graphsizes:
        for i in range(10):
            G = er_graph(n=size, p=0.6) # slightly dense
            fa, fr, nu = graphtest(G, verifier_algorithm)
            if len(fr) != 0:
                print(f'Problems with verifier on size {size}')
            outcomes.append((G, fa, fr))
    return outcomes


def automatic_sanity_check(n=64, verifier_algorithm=henry): # you should expect to randomly generate a correct tree on n=6 with 1000 random trees
    """
    1. make 10 random graphs of size n,
    2. create 1000 good trees and 1000 probably bad trees,
    3. assert no false rejects
    4. flag false accepts (should be very unlikely for sufficiently large n)
    """
    outcomes = []
    nums_unique_trees_found = []
    for i in range(10):
        G = er_graph(n=n, p=0.5)
        false_accepts, false_rejects, num_unique_true_trees_found = graphtest(G, verifier_algorithm)
        if len(false_rejects) != 0:
            print(f'Problems with verifier on Graph {i}')
        #if len(false_accepts) != 0:
        #    print(f'Manually inspect `false accept` on Graph {i}')
        outcomes.append((G, false_accepts, false_rejects))
        nums_unique_trees_found.append(num_unique_true_trees_found)
    how_many_fa(outcomes, nums_unique_trees_found)
    return outcomes

def how_many_fa(outcomes, nus):
    ix=0
    for triple in outcomes:
        nu = nus[ix]
        G = triple[0]
        false_accepts = triple[1]
        false_rejects = triple[2]
        print(f'Graph {ix} has {len(false_accepts)} possibly false accepts and {len(false_rejects)} false rejects || for-context, we saw {nu} unique trees')
        ix+=1

#o = automatic_sanity_check(5, henry)


##################################################################################################################
# BELLMAN-FORD CHECKER
##################################################################################################################
# Test whether model's pi represents valid bellman-ford



def bellman_ford_cost(A, s):
  """Bellman-Ford's single-source shortest path (Bellman, 1958).
  This has been taken and adapted from the original CLRS Bellman-Ford implementation"""
  #chex.assert_rank(A, 2)

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
            # print('S', s)
            # print('A \n', A)
            # print('ppi', parentpath[i])
            # print('i', i)
            # print('A[ppi, i]', A[parentpath[i],i])
            # print('A[ppi, i] == 0', A[parentpath[i],i] == 0)
            # print('bf_dfsv.py self parent exclusion')
            return False
        else:
            BF_tree_adj[parentpath[i],i] = A[parentpath[i],i]

    model_costs = bellman_ford_cost(BF_tree_adj, s)
    #breakpoint()

    if (true_costs == model_costs).all():
        return True
    else:
        return False