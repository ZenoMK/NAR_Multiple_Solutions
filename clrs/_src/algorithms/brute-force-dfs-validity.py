import networkx as nx
from itertools import permutations, product
import matplotlib.pyplot as plt
import numpy as np
import ast

# ------------------------------------------------------------
# Reading from a results .csv
# ------------------------------------------------------------
import pandas as pd

def string_to_array(colname, df, dims=None):
    strings = df[colname]
    As = []
    for mat in strings:
        if mat.find('.') != -1:  # upwards saved format
            mat = mat.replace('.', ',')
        elif mat.find(',') == -1:  # argmax saved format
            mat = mat.replace(' ', ',')
        mat = np.array(ast.literal_eval(mat))
        if dims is not None:
            mat = mat.reshape(dims)
        As.append(mat)
    df[colname] = As
    return df

def ingest_resultfile(fpath='results/first5_dfs.csv'):  # fpath = 'results/first5_dfs.csv'
    df = pd.read_csv(fpath) # pd.read_csv('results/first5_dfs.csv') if run from main
    # eat adjacencies
    string_to_array('As', df, (5, 5))
    # eat other columns
    ARRAY_COLNAMES = ['Argmax_Model_Trees', 'Argmax_True_Trees',
                      'Random_Model_Trees', 'Random_True_Trees',
                      'Upwards_Model_Trees', 'Upwards_True_Trees',
                      'altUpwards_Model_Trees', 'altUpwards_True_Trees']
    for colname in ARRAY_COLNAMES:
        string_to_array(colname, df)
    return df

df = ingest_resultfile('results/first5_dfs.csv')

def brute_force_column_validity(df, As_name, Pis_name):
    """assumes df with As and Pis"""
    mask = []
    for row_num in range(len(df)):
        A = df[As_name].iloc[row_num]
        pi = df[Pis_name].iloc[row_num]
        mask.append(brute_force_dfs_validity(A, pi))
    return mask

#mask = brute_force_column_validity(df, As_name='As', Pis_name='Upwards_Model_Trees')

# ------------------------------------------------------------
# meat and potatoes methods
# ------------------------------------------------------------
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



# ------------------------------------------------------------
# test functions
# ------------------------------------------------------------
mask = brute_force_column_validity(df, As_name='As', Pis_name='Upwards_Model_Trees')


# to do n=5 for each method, you'd call bfcv on each name in ARRAY_COLNAMES, divide by 10, and report...
# to see discrepency, you'd compare bfcv mask to df['mask'] column, throw error if difference

def discrepency_finder(df, tree_colnames, mask_colnames):
    """computes bfcv, and """

    # newdf = pd.DataFrame({
    #     'Argmax_Model_Old': [],
    #     'Argmax_True_Old': [],
    #     'Random_Model_Old': [],
    #     'Random_True_Old': [],
    #     'Upwards_Model_Old': [],
    #     'Upwards_True_Old': [],
    #     'altUpwards_Model_Old': [],
    #     'altUpwards_True_Old': [],
    #     'Argmax_Model_New': [],
    #     'Argmax_True_New': [],
    #     'Random_Model_New': [],
    #     'Random_True_New': [],
    #     'Upwards_Model_New': [],
    #     'Upwards_True_New': [],
    #     'altUpwards_Model_New': [],
    #     'altUpwards_True_New': []
    # })
    interleaved_names = [elem for pair in zip(tree_colnames, mask_colnames) for elem in pair]

    newdf = pd.DataFrame(columns=interleaved_names)


    for i in range(len(tree_colnames)):
        #breakpoint()
        tree_colname = tree_colnames[i]
        mask_colname = mask_colnames[i]
        new_mask = brute_force_column_validity(df, As_name='As', Pis_name=tree_colname)
        old_mask = df[mask_colname].tolist()
        newdf[tree_colname] = new_mask
        newdf[mask_colname] = old_mask

        prefix = tree_colname.split('_')[0] + '_' +  tree_colname.split('_')[1]
        #print(prefix)
        name = prefix + '_XOR'
        newdf[name] = newdf[tree_colname] ^ newdf[mask_colname]



    return newdf



ARRAY_COLNAMES = ['Argmax_Model_Trees', 'Argmax_True_Trees',
                      'Random_Model_Trees', 'Random_True_Trees',
                      'Upwards_Model_Trees', 'Upwards_True_Trees',
                      'altUpwards_Model_Trees', 'altUpwards_True_Trees']

MASK_COLNAMES = ['Argmax_Model_Mask', 'Argmax_True_Mask',
                 'Random_Model_Mask', 'Random_True_Mask',
                 'Upwards_Model_Mask', 'Upwards_True_Mask',
                 'altUpwards_Model_Mask', 'altUpwards_True_Mask']


adf = discrepency_finder(df, ARRAY_COLNAMES, MASK_COLNAMES)
xors =  adf.iloc[:, -8:]
xor_sums = xors.sum()
print(xor_sums)

# want also sums as fraction of Mask
olds = adf.iloc[:, 1:-8:2]
old_sums = olds.sum()
print(old_sums)

# fraction of error... WOOOF
frac_data = [np.array(xor_sums.tolist()) / np.array(old_sums.tolist())]  # nan means 0/0, 0 means 0/nonzero (which is ideal),
fracs = pd.DataFrame(data=frac_data, columns=olds.columns)


# df.iloc[6, :]['Upwards_True_Trees']
# array([0, 1, 4, 0, 1])
# df.iloc[6, :]['As']
# array([[1, 0, 0, 1, 0],
#        [0, 0, 1, 1, 1],
#        [1, 0, 0, 1, 0],
#        [1, 0, 0, 1, 0],
#        [0, 0, 1, 0, 0]])

