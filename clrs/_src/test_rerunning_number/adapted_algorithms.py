# adapt algorithms to make the rerunning number an input
import chex
from typing import Tuple
import numpy as np
from clrs._src import probing
from clrs._src import specs
_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]
_OutputClass = specs.OutputClass
from clrs._src import probing, specs


def dfs(A: _Array, num_solutions: int) -> _Out:
  """Depth-first search (Moore, 1959)."""

  chex.assert_rank(A, 2)
  probeslist = []
  pies = []

  NUM_SOLUTIONS = num_solutions

  for i in range(NUM_SOLUTIONS):

      probes = probing.initialize(specs.SPECS['dfs'])

      A_pos = np.arange(A.shape[0])

      probing.push(
          probes,
          specs.Stage.INPUT,
          next_probe={
              'pos': np.copy(A_pos) * 1.0 / A.shape[0],
              'A': np.copy(A),
              'adj': probing.graph(np.copy(A))
          })

      color = np.zeros(A.shape[0], dtype=np.int32)
      pi = np.arange(A.shape[0])
      d = np.zeros(A.shape[0])
      f = np.zeros(A.shape[0])
      s_prev = np.arange(A.shape[0])
      time = 0
      shuffled = np.arange(A.shape[0])
      np.random.shuffle(shuffled)
      for s in range(A.shape[0]):
        if color[s] == 0:
          s_last = s
          u = s
          v = s
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })
          while True:
            if color[u] == 0 or d[u] == 0.0:
              time += 0.01
              d[u] = time
              color[u] = 1
              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })

            for v in shuffled:
              if A[u, v] != 0:
                if color[v] == 0:
                  pi[v] = u
                  color[v] = 1
                  s_prev[v] = s_last
                  s_last = v

                  probing.push(
                      probes,
                      specs.Stage.HINT,
                      next_probe={
                          'pi_h': np.copy(pi),
                          'color': probing.array_cat(color, 3),
                          'd': np.copy(d),
                          'f': np.copy(f),
                          's_prev': np.copy(s_prev),
                          's': probing.mask_one(s, A.shape[0]),
                          'u': probing.mask_one(u, A.shape[0]),
                          'v': probing.mask_one(v, A.shape[0]),
                          's_last': probing.mask_one(s_last, A.shape[0]),
                          'time': time
                      })
                  break

            if s_last == u:
              color[u] = 2
              time += 0.01
              f[u] = time

              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })

              if s_prev[u] == u:
                assert s_prev[s_last] == s_last
                break
              pr = s_prev[s_last]
              s_prev[s_last] = s_last
              s_last = pr

            u = s_last

      probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
      probing.finalize(probes)

      pies.append(pi)
      probeslist.append(probes)
      # only take the time hint, to figure out trajectory length
      # run code in no-hint mode
      # for every probing.push, +1 iteration of the GNN


  adjs = []
  for i in range(NUM_SOLUTIONS):
    adj = np.zeros(A.shape)
    for j in range(len(pies[0])):
        adj[j, pies[i][j]] = 1
    adjs.append(adj)
  parent_dist = sum(adjs) / NUM_SOLUTIONS
  #parent_dist = sp.special.logit(parent_dist)
  #print(probes)
  #print(parent_dist)
  probeslist[0]['output']['node']['pi']['data'] = parent_dist
  #breakpoint()
  return parent_dist, probeslist[0]



def bellman_ford(A: _Array, s: int, num_solutions: int) -> _Out:
  """Bellman-Ford's single-source shortest path (Bellman, 1958)."""

  chex.assert_rank(A, 2)

  A_pos = np.arange(A.shape[0])

  # run many, make distribution
  probeslist = []
  pies = []
  NUM_SOLUTIONS = num_solutions

  for i in range(NUM_SOLUTIONS):
      probes = probing.initialize(specs.SPECS['bellman_ford'])

      probing.push(
          probes,
          specs.Stage.INPUT,
          next_probe={
              'pos': np.copy(A_pos) * 1.0 / A.shape[0],
              's': probing.mask_one(s, A.shape[0]),
              'A': np.copy(A),
              'adj': probing.graph(np.copy(A))
          })

      d = np.zeros(A.shape[0])
      pi = np.arange(A.shape[0])
      msk = np.zeros(A.shape[0])
      d[s] = 0
      msk[s] = 1

      shuffled1 = np.arange(1, A.shape[0])
      np.random.shuffle(shuffled1)
      shuffled1 = np.concatenate(([0],shuffled1))

      # shuffled for inner loop
      shuffled2 = np.arange(A.shape[0])
      np.random.shuffle(shuffled2)
      while True:
        prev_d = np.copy(d)
        prev_msk = np.copy(msk)
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pi_h': np.copy(pi),
                'd': np.copy(prev_d),
                'msk': np.copy(prev_msk)
            })
        for u in shuffled1:
          for v in shuffled2:
            if prev_msk[u] == 1 and A[u, v] != 0:
              if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
                d[v] = prev_d[u] + A[u, v]
                pi[v] = u
              msk[v] = 1
        if np.all(d == prev_d):
          break

      probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
      probing.finalize(probes)

      pies.append(pi)
      probeslist.append(probes)

    # CHeck indent



  ### copied from DFS CODE
  # build adj matrix of "is i a parent of j in any pi", sums and divides.
  adjs = []
  for i in range(NUM_SOLUTIONS):
      adj = np.zeros(A.shape)
      for j in range(len(pies[0])):  # what's the parent of j?
          #breakpoint()
          adj[j, pies[i][j]] = 1  # at row j, put 1 in the column corresponding to parent
      adjs.append(adj)
  parent_dist = sum(adjs) / NUM_SOLUTIONS
  # parent_dist = sp.special.logit(parent_dist)
  # print(probes)
  # print(parent_dist)
  probeslist[0]['output']['node']['pi']['data'] = parent_dist

  #breakpoint()
  ## CHECK THE PUSHING!
  return parent_dist, probeslist[0]