import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from bf_dfs_verifiers import agnostic_henry


def draw_graph_with_highlights(adj_matrix, highlight_edges_indices):
    # Create a directed or undirected graph depending on your use case
    G = nx.DiGraph()  # or use nx.Graph() if the graph is undirected
    n = len(adj_matrix)

    # Add all edges from the adjacency matrix
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j]:
                G.add_edge(i, j)

    # Define highlighted edges based on the array
    highlight_edges = [(j, idx) for idx, j in enumerate(highlight_edges_indices)]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # use a consistent layout
    plt.figure(figsize=(6, 6))

    # Draw normal edges
    normal_edges = [e for e in G.edges() if e not in highlight_edges]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1, edge_color='gray')

    # Draw highlighted (bold) edges
    nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=3, edge_color='red')

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=600)
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='black')

    plt.title("Graph with Highlighted Edges")
    plt.axis('off')
    plt.show()

# Example usage:
# adj = np.array([
#     [1, 1, 0, 1],
#     [0, 1, 0, 0],
#     [1, 0, 1, 0],
#     [1, 1, 0, 1]
# ])
# highlight_array = [3, 3, 2, 3]
# draw_graph_with_highlights(adj, highlight_array)


false_neg_oga = np.array(
    [[1, 1, 0, 1],
     [0, 1, 0, 0],
     [1, 0, 1, 0],
     [1, 1, 0, 1]]
)

# As =
# [[1 1 0 1]
#  [0 1 0 0]
#  [0 0 1 1]
#  [1 1 0 1]]

Perms = [3, 1, 2, 0]
actual = [0, 0, 2, 0]
false_neg_og = [3, 3, 2, 3]


#print(agnostic_henry(false_neg_oga, false_neg_og)) # TRUE & fixed

# ----------------------------------------------------------------------------
true_neg_ogA = np.array(
[[1, 1, 0, 0],
 [1, 1, 0, 1],
 [1, 0, 0, 0],
 [1, 1, 1, 0]]
)

false_pos_A = np.array(
    [[1, 1, 0, 1],
     [1, 0, 1, 1],
     [0, 0, 0, 1],
     [1, 0, 0, 1]]
)
Perms2 = [3, 0, 2, 1]
false_pos_actual = [0, 0, 0, 1]
true_neg_og = [3, 1, 1, 1]
# ---- both should be false
# OG: False
# actual True

#draw_graph_with_highlights(false_pos_A, false_pos_actual) # should be false bcuz 1 should discover 2, but rn 0->2
#print(agnostic_henry(false_pos_A, false_pos_actual)) # should be false bcuz 1 should discover 2

#draw_graph_with_highlights(true_neg_ogA, true_neg_og)
#agnostic_henry(true_neg_ogA, true_neg_og) # NO we want alll  highlighted edges to be tree edges


