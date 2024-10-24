import networkx as nx
from itertools import permutations, product
import matplotlib.pyplot as plt
import numpy as np


def brute_force_dfs_validity(A, pi):
    """takes array, makes all valid trees, tests whether pi is a member of valid trees"""
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
    valid_trees = generate_all_dfs_trees_from_start(graph, 0)
    pi_as_edge_rep = parent_rep_to_edges(pi)
    return pi_as_edge_rep in valid_trees

def parent_rep_to_edges(pi):
    edges = []
    for i in range(len(pi)):
        edges.append((pi[i], i))
    return set(edges)

pi = [0,0,1,1]
e = parent_rep_to_edges(pi)

# Function to perform DFS with a specific neighbor visiting order and build the DFS tree
def dfs_tree_with_order(start_node, neighbor_order):
    tree_edges = [(0, 0)]  # 0 always its own parent
    visited = set()

    def dfs(node):
        visited.add(node)
        neighbors = neighbor_order[node]
        # breakpoint()
        for neighbor in neighbors:
            if neighbor not in visited:
                tree_edges.append((node, neighbor))
                dfs(neighbor)

    dfs(start_node)
    return set(tree_edges)  # return as set of edges for comparison


# Function to generate all possible DFS trees from a single start node with different neighbor visit orders
def generate_all_dfs_trees_from_start(graph, start_node):
    dfs_trees = []
    nodes = list(graph.nodes)

    # Generate all possible permutations of neighbors for each node
    neighbor_orders = {}
    for node in nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_orders[node] = list(permutations(neighbors))
    # Generate all possible neighbor orders
    orderings = product(*neighbor_orders.values())

    for ordering in orderings:
        # Perform DFS and collect the tree
        tree = dfs_tree_with_order(start_node, ordering)
        dfs_trees.append(tree)

    #breakpoint()
    # remove duplicates by temporarily changing lists to tuples
    unique_tuples = set(tuple(sublist) for sublist in dfs_trees)
    unique_trees = [set(t) for t in unique_tuples]
    return unique_trees


# Second Example Graph
G2 = nx.DiGraph()
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
G2.add_edges_from(edges)
A2 = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
    ]
)

# Example graph
G = nx.Graph()
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
G.add_edges_from(edges)
A = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
    ]
)

# Generate all DFS trees from a single start node (e.g., node 0)
start_node = 0
dfs_trees = generate_all_dfs_trees_from_start(G, start_node)
dfs_trees2 = generate_all_dfs_trees_from_start(G2, start_node)

# Sanity check the brute-force method
assert brute_force_dfs_validity(A, pi)
assert not brute_force_dfs_validity(A2, pi)


# ------------------------------------------------------------


# Print and visualize the DFS trees
def display_trees(tree_list):
    for i, tree in enumerate(tree_list):
        print(f"DFS Tree {i + 1}: {tree}")
        nx.draw(nx.DiGraph(tree), with_labels=True)
        plt.show()