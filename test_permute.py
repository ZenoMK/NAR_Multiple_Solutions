import itertools
import numpy as np


from bf_dfs_verifiers import check_valid_BFpaths


def permute_node(node, permutation):
    return permutation[node]

def invert_permutation(perm):
    """so that p_inv(p(i)) = i, and p(p_inv(i)) = i"""
    return np.argsort(perm)

def permute_adjacency_matrix(adj_matrix, permutation):
    """use the inv_perm so that adj[i,j] = pi_adj[pi[i],pi[j]]"""
    inv_perm = invert_permutation(permutation)
    return adj_matrix[np.ix_(inv_perm, inv_perm)]

def permute_parentpath(pp, permutation):
    """want pi(pp[i])->pi(i) == pp[i]->i. Do the algebra and you get this nasty beast"""
    inv_perm = invert_permutation(permutation)
    return [permutation[pp[inv_perm[y]]] for y in range(len(pp))]

def permute_mask(mask, permutation): # fixme: untested
    return mask[cur_perm]

def test_adjacency_permutation():
    # Original graph: 4 nodes
    adj = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])

    permutation = [2, 0, 3, 1]  # π(i) = permutation[i]
    start_node = 1
    node_list = [0,1,2,3]


    # Step 1: Apply permutation to adjacency matrix
    permuted_adj = permute_adjacency_matrix(adj, permutation)
    print("Original adjacency matrix:\n", adj)
    print('--\nPermutation', permutation, '\n--')
    print("Permuted adjacency matrix:\n", permuted_adj)

    # Step 2: Permute the start node
    permuted_start = permute_node(start_node, permutation)
    print(f"Original start node: {start_node}, Permuted: {permuted_start}")

    # Step 3: Permute the parentpaths
    ppp = permute_parentpath(node_list, permutation)
    print(f"Original pps: {node_list}, ppps: {ppp}")
    test_parent_paths(adj, node_list, permuted_adj, ppp, permutation)

    # Step 4: Verify that adj[i][j] == permuted_adj[π(i)][π(j)]
    print("Verifying adjacency consistency under permutation...")
    passed = True
    for i in range(len(adj)):
        for j in range(len(adj)):
            pi_i, pi_j = permutation[i], permutation[j]
            if adj[i][j] != permuted_adj[pi_i][pi_j]:
                print(f"Mismatch at ({i},{j}): {adj[i][j]} != {permuted_adj[pi_i][pi_j]}")
                passed = False
    if passed:
        print("✅ Permutation preserves adjacency structure.")

    # Step 5
    print("Verifying adjacency unpermutation...")
    passed = True
    p_inv = invert_permutation(permutation)
    unperm_adj = permute_adjacency_matrix(permuted_adj, p_inv)
    unperm_s = p_inv[permuted_start]
    print(f"Original start node: {start_node}, Permuted: {permuted_start}, unperm_s: {unperm_s}")
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] != unperm_adj[i][j]:
                print(f"Mismatch at ({i},{j}): {adj[i][j]} != {unperm_adj[i][j]}")
                passed = False
    if passed:
        print("✅ UnPermutation preserves adjacency structure.")



def test_adj(adj, permuted_adj, permutation):
    print("Verifying adjacency consistency under permutation...")
    passed = True
    for i in range(len(adj)):
        for j in range(len(adj)):
            pi_i, pi_j = permutation[i], permutation[j]
            if adj[i][j] != permuted_adj[pi_i][pi_j]:
                print(f"Mismatch at ({i},{j}): {adj[i][j]} != {permuted_adj[pi_i][pi_j]}")
                passed = False
    if passed:
        print("✅ Permutation preserves adjacency structure.")
    else:
        raise AssertionError

def test_parent_paths(adj, parentpath, permuted_adj, permuted_pp, perm):
    passed = True
    for i in range(len(parentpath)):
        #if adj[parentpath[i], i] != permuted_adj[permuted_pp[i], i]: # fixme: this condition impossible, since og node i not perm node i, so why would it have same incoming edge
        #if adj[parentpath[i], i] != permuted_adj[permuted_pp[i], perm[i]]: # fixme: this condition means you shouldnt be reading ppp[i] as parent of i, but instead parent of perm[i]
        # TFAE & CORRECT
        #inv_perm = inverse_permutation(perm)
        #j = np.where(inv_perm == i)[0][0]
        #if adj[parentpath[i],i] != permuted_adj[permuted_pp[j], j]: # therefore, restructure ppp, and compare adj[parentpath[i], i] to adj[ppp[perm[i]],perm[i]]
        if  adj[parentpath[i],i] != permuted_adj[permuted_pp[perm[i]], perm[i]]:
            print(f"Mismatch at index ({i}): {adj[parentpath[i],i]} != {permuted_adj[permuted_pp[perm[i]], perm[i]]}")
            passed = False
    if passed:
        print('all happy funtimes')
        return True
    else:
        raise AssertionError

def test_perm_and_inverse(perm):
    p_inv = invert_permutation(perm)
    for i in range(len(perm)):
        if p_inv[perm[i]] != i or perm[p_inv[i]] != i:
            print(f'problems at index {i} with p_inv(perm(i)) = {p_inv[perm[i]]} and perm[p_inv[i]] = {perm[p_inv[i]]}')
            raise AssertionError


def test_unpermuting(p_adj, ppp, perm):
    """given p_adj, ppp, and perm that produced it, recover adj and pp"""
    # recover the goods
    inv_perm = invert_permutation(perm)
    adj = permute_adjacency_matrix(p_adj, inv_perm) # funnily this will wind-up recomputing and using perm internally
    pp = permute_parentpath(ppp, inv_perm) # fixme: will this work?

    # verify that it recovered properly
    test_adj(adj, p_adj, perm)
    test_parent_paths(adj, pp, p_adj, ppp, perm)


if __name__ == '__main__':
    #test_adjacency_permutation()
    # OK so the confusing bit is:
    # given permutation [2,0,1]
    # apply permutation to node with permutation[node] (meaning node 0 gets label 2, node 1 gets label 0, node 2 gets label 1)
    # p_inv = [1,2,0] (node 0 gets label 1, node 1 gets label 2, node 2 gets label 0)
    # apply permutation to adj with adj[np.ix_(p_inv, p_inv)], bcuz A = adj[np.ix_(f,f)] means A[i,j] = adj[fi,fj], so A = adj[np.ix_(p_inv, p_inv)] means A[pi(i), pi(j)] gonna be adj[i,j].
    # apply permutation to parent path with [permutation[pp[inv_perm[y]]] for y in range(len(pp))], which only makes sense when you follow the algebra, or believe the test

    adj = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    s = np.array([1,0,0,0]) # permutation (1, 2, 0, 3) it goes wrong
    start_node = 0
    parent_path = [3, 0, 1, 2]

    # adj = np.array([ # 0->1->3->2
    #     [0, 1, 4, 5],
    #     [0, 0, 6, 1],
    #     [8, 9, 10, 11],
    #     [12, 13, 2, 15]
    # ])
    #
    # start_node = 0
    # parent_path = [0,0,3,1]

    for permutation in itertools.permutations(np.arange(len(adj))):
        permuted_adj = permute_adjacency_matrix(adj, permutation)
        #breakpoint()
        permuted_pp = permute_parentpath(parent_path, permutation)
        permuted_s = permutation[start_node]
        ps = np.array(permutation)[s] #permutation[np.argmax(s)] #s[list(permutation)]
        if ps != permuted_s:
            breakpoint()

        test_perm_and_inverse(perm=permutation)
        test_adj(adj=adj, permuted_adj=permuted_adj, permutation=permutation)
        test_parent_paths(adj=adj, parentpath=parent_path, permuted_adj=permuted_adj, permuted_pp=permuted_pp,
                      perm=permutation)

        print('unpermuting---')
        test_unpermuting(permuted_adj, permuted_pp, permutation)

        if check_valid_BFpaths(adj, start_node, parentpath=parent_path) != check_valid_BFpaths(permuted_adj, permuted_s, parentpath=permuted_pp):
            breakpoint()
        else:
            print('validity was, ', check_valid_BFpaths(adj, start_node, parentpath=parent_path))
    # # BAH HUMBUG. No possible pp can satisfy test_parent_paths :(.
    # import itertools
    # for perm in itertools.permutations([0,1,2,3]):
    #     if test_parent_paths(adj, parent_path, permuted_adj, perm):
    #         print(perm, 'worked')
    #     else:
    #         print(perm, 'failed')

    # def find_new_pp(adj, pp, perm_adj):
    #     for i in range(len(pp)):
    #         edge = adj[pp[i]][i]
    #         ix = np.where(perm_adj == edge)
    #         print(f'ix matching {pp[i]} to {i}: {ix}')

    # import numpy as np
    #
    # # Original adjacency matrix
    # adj = np.array([
    #     [0, 1, 2],
    #     [3, 4, 5],
    #     [6, 7, 8]
    # ])
    #
    # import itertools
    #
    # # perm = np.array([1,2,0])
    # # perm = np.array([2, 0, 1])
    # for perm in itertools.permutations([0, 1, 2]):
    #     inv_perm = np.argsort(perm)
    #
    #     # Permute the adjacency matrix
    #     perm_adj = adj[np.ix_(inv_perm, inv_perm)]
    #
    #     # Edges of interest
    #     pp = np.array([1, 2, 0])  # So adj[1,0], adj[2,1], adj[0,2]
    #
    #     # Construct ppp
    #     n = len(pp)
    #     ppp = np.empty_like(pp)
    #     for i in range(n):
    #         j = np.where(inv_perm == i)[0][0]
    #         ppp[j] = np.where(inv_perm == pp[i])[0][0]
    #
    #     ppp2 = [perm[pp[inv_perm[y]]] for y in range(len(pp))]
    #     assert (ppp == ppp2).all()
    #
    #     # Output
    #     print("adj:")
    #     print(adj)
    #     print("\nperm_adj:")
    #     print(perm_adj)
    #     print("\npp:", pp)
    #     print("ppp:", ppp)
    #
    #     # Verification
    #     print("\nVerifying that adj[pp[i], i] == perm_adj[ppp[j], j] for j = where(inv_perm == i):")
    #     for i in range(n):
    #         j = np.where(inv_perm == i)[0][0]
    #         a = adj[pp[i], i]
    #         b = perm_adj[ppp[j], j]
    #         print(f"i={i}, j={j}, adj[{pp[i]}, {i}] = {a}, perm_adj[{ppp[j]}, {j}] = {b}")
    #         assert a == b
