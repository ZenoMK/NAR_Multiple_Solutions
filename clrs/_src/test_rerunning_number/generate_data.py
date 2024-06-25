import numpy as np
from clrs._src.samplers import DfsSampler
from clrs._src.test_rerunning_number.adapted_algorithms import dfs, bellman_ford
import torch
_rng = np.random.RandomState()

# generate array of mean and std of KL_Div between using 20 and 40 iterations for DFS

def generate_data(algorithm = 'dfs', num_solutions = [20,40], min_size = 5, max_size = 64, graphs_per_size = 100):
    sizes = [i for i in range(min_size, max_size + 1)]
    means = np.zeros(len(sizes))
    std = np.zeros(len(sizes))
    for i in range(len(sizes)):
        size = sizes[i]
        if algorithm == 'dfs':
            graphs = [random_er_graph(nb_nodes=size, directed=True) for k in range(graphs_per_size)]
            distributions1 = [dfs(graph, num_solutions[0])[0] for graph in graphs]
            distributions2 = [dfs(graph, num_solutions[1])[0] for graph in graphs]
        else:
            graphs = [random_er_graph(nb_nodes=size, directed=True, weighted=True) for k in range(graphs_per_size)]
            distributions1 = [bellman_ford(graph,0,num_solutions[0])[0] for graph in graphs]
            distributions2 = [bellman_ford(graph,0, num_solutions[1])[0] for graph in graphs]
        divergences = [torch.nn.functional.kl_div(torch.Tensor(distributions1[j]), torch.Tensor(distributions2[j])) for j in range(len(distributions1))]
        means[i] = np.mean(divergences)
        std[i] = np.std(divergences)

    return means, std

  # Function taken from the original CLRS-30 repo
def random_er_graph(nb_nodes, p=0.5, directed=False, acyclic=False,
                   weighted=False, low=0.0, high=1.0):

    """Random Erdos-Renyi graph."""
    mat = _rng.binomial(1, p, size=(nb_nodes, nb_nodes))
    if not directed:
      mat *= np.transpose(mat)
    elif acyclic:
      mat = np.triu(mat, k=1)
      p = _rng.permutation(nb_nodes)  # To allow nontrivial solutions
      mat = mat[p, :][:, p]
    if weighted:
      weights = _rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
      if not directed:
        weights *= np.transpose(weights)
        weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
      mat = mat.astype(float) * weights
    return mat



