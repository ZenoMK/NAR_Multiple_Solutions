# THIS IS MOSTLY A COPY OF vanilla_eval, relying on dummy_eval.py to load model, eval_permute_stats.py for compute stats
import pandas as pd
import time
import clrs
import jax

from clrs.examples.run import create_samplers, make_multi_sampler, permute_eval_and_record
from clrs._src.probing import mask_one

import json
import numpy as np
from types import SimpleNamespace # THIS IS SUSSY but bcuz i dont care about flags being absl flags, and just want to use the same values
import functools

from eval_permute_stats import compute_bf_stats, compute_dfs_stats
from dummy_eval import load_model, make_test_sampler
from test_permute import invert_permutation, permute_adjacency_matrix, permute_parentpath # Fixme: fragile to dir of run command
from clrs.examples.run import _concat
# -----------------------------------------------------------------------------------------------------------------
# DEFINE NEW COLLECT_AND_EVAL FUNCTION
# -----------------------------------------------------------------------------------------------------------------
def permute_everything_then_predict(ffs, predict_fn, num_perms, graph_num, new_rng_key, extras, dont_permute):
  """returns GraphIDs, permuted As, permuted Ss || but crucially also permuted pos and adj"""
  IDs = []
  As = []
  preds = []
  perms = []
  Ss = []
  # breakpoint()

  # Save originals so you can apply multiple permutations to the same stuff
  og_pos = np.copy(ffs.inputs[0].data)
  if extras['algorithm'] == 'dfs':
    ogAs = np.copy(ffs.inputs[1].data)
    og_adj = np.copy(ffs.inputs[2].data)
    ogSs = None #fixme: dummy, not needed or used by alg
  elif extras['algorithm'] == 'bellman_ford':
    ogSs = np.copy(ffs.inputs[1].data)
    ogAs = np.copy(ffs.inputs[2].data)
    og_adj = np.copy(ffs.inputs[3].data)
  else:
    raise NotImplementedError

  # breakpoint()
  # permute multiple times
  batch_size = ffs.inputs[0].data.shape[0]
  for i in range(num_perms):
    # prepare metadata... do you need these? or can you refer to ogs in list comp
    cur_pos = og_pos
    cur_As = ogAs
    cur_adj = og_adj
    cur_Ss = ogSs
    graph_size = len(cur_As[0])
    if dont_permute:
      cur_perm = np.arange(graph_size)
    else:
      cur_perm = np.random.permutation(graph_size)  # np.arange(graph_size)
    inv_perm = invert_permutation(cur_perm)  # used in adj matrix so that A[i,j] = PI(A)[PI(i),PI(j)]
    cur_perms = [cur_perm] * batch_size
    cur_IDs = [i for i in range(graph_num, graph_num + batch_size)]
    #breakpoint()
    # shuffle the node indices
    ffs.inputs[0].data = np.array([arr[inv_perm] for arr in cur_pos])
    if extras['algorithm'] == 'dfs':
      ffs.inputs[1].data = np.array([A[np.ix_(inv_perm, inv_perm)] for A in cur_As])  # use inv so that A[i,j] = PI(A)[PI(i),PI(j)]
      ffs.inputs[2].data = np.array([A[np.ix_(inv_perm, inv_perm)] for A in cur_adj])
      cur_Ss = [0] * batch_size  # DFSO always 0 starts
    elif extras['algorithm'] == 'bellman_ford':
      ffs.inputs[3].data = np.array([A[np.ix_(inv_perm, inv_perm)] for A in cur_adj])
      ffs.inputs[2].data = np.array([A[np.ix_(inv_perm, inv_perm)] for A in cur_As])
      ffs.inputs[1].data = np.array([mask_one(i=cur_perm[np.argmax(arr)], n=len(arr)) for arr in cur_Ss]) # permuting S || arr[inv_perm]
      #breakpoint()
      cur_Ss = [np.argmax(mask) for mask in ffs.inputs[1].data]
      #[np.argmax(mask) for mask in ogSs]
    else:
      raise NotImplementedError
    #breakpoint()
    # print('ogA \n', ogAs[0])
    # print('s: ', temp_s)
    # print('perm:', cur_perm)
    # print('permA \n', ffs.inputs[2].data[0])
    # print('permS \n', ffs.inputs[1].data[0])
    # print('run.py, ffs.inputs[1].data[0]:', ffs.inputs[1].data[0])
    # TESTING WHAT CAN BE SHUFFLED: BF
    # ffs.inputs[0].data = np.array([row[::-1] for row in ffs.inputs[0].data]) # reverse pos, doenst matter
    # ffs.inputs[1].data = np.array([row[::-1] for row in ffs.inputs[1].data]) # reverse s, matters
    # ffs.inputs[2].data = np.array([A[np.ix_([3,2,1,0], [3,2,1,0])] for A in ffs.inputs[2].data]) # reverse A, doesnt matter??
    # ffs.inputs[3].data = np.array([A[np.ix_([3,2,1,0], [3,2,1,0])] for A in ffs.inputs[3].data]) # reverse adj *AFFECTS*
    # ------ ffs.hints, they're all shape(4,32,4)
    # ffs.hints[0].data = ffs.hints[0].data[..., ::-1] # reverse pi_h, doesnt matter
    # ffs.hints[1].data = ffs.hints[1].data[..., ::-1] # reverse d, doesnt matter
    # ffs.hints[2].data = ffs.hints[2].data[..., ::-1] # reverse msk, doesnt matter
    # ffs[2] = np.array(ffs[2][::-1]) # cant change :( due to Features type

    # TESTING WHAT CAN BE SHUFFLED: DFS
    # ffs.inputs[0].data = np.array([row[::-1] for row in ffs.inputs[0].data])  # reverse pos, CHANGES A LOT
    # ffs.inputs[1].data = np.array([A[np.ix_([3,2,1,0], [3,2,1,0])] for A in ffs.inputs[1].data])  # reverse A, CHANGES A BIT
    # ffs.inputs[2].data = np.array([A[np.ix_([3,2,1,0], [3,2,1,0])] for A in ffs.inputs[2].data])  # reverse adj, CHANGES A BIT
    # ------ ffs.hints, they're all shape(12,32,4)
    # ffs.hints[0].data = ffs.hints[0].data[..., ::-1] all 10 hints dont matter with hint_mode --none, can matter if you reverse all with hints encoded_decoded

    # ffs[2] = np.array(ffs[2][::-1]) # cant change :( due to Features type

    t1 = time.time()
    cur_preds, _ = predict_fn(new_rng_key, ffs)
    t2 = time.time()
    print(
      f"n={graph_size} the prediction line  itself took {t2 - t1} seconds")  # size 64 graphs take 30sec to predict each time on my laptop. thats no fun
    # print('cur_preds[0], ', cur_preds['pi'].data[0])
    # breakpoint()
    # STORE SHIT
    IDs.extend(cur_IDs)
    if extras['algorithm'] == 'dfs':
      As.extend(ffs.inputs[1].data)
    elif extras['algorithm'] == 'bellman_ford':
      As.extend(ffs.inputs[2].data)
    else:
      raise NotImplementedError
    preds.extend(cur_preds['pi'].data)
    perms.extend(cur_perms)
    Ss.extend(cur_Ss)

  return IDs, As, perms, preds, Ss


def permute_and_eval(sampler, predict_fn, sample_count, rng_key, extras, num_perms=1, dont_permute=False):
  """Collect batches of output and hint preds and evaluate them."""
  # 32 graphs in a batch?
  # Do 5 permutations per graph
  # compute accuracy and write to csv
  processed_samples = 0
  preds = []
  As = []
  IDs = []
  perms = []
  Ss = []
  #outputs = []  # fixme: just for debug
  while processed_samples < sample_count:  # do another batch
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    new_rng_key, rng_key = jax.random.split(rng_key)
    #outputs.append(feedback.outputs) # for COMPARE DEFAULT EVAL

    t1 = time.time()
    # each list is batch x num_perms length
    batch_IDs, batch_As, batch_perms, batch_preds, batch_Ss = permute_everything_then_predict(ffs=feedback.features,
                                                                                     predict_fn=predict_fn,
                                                                                     num_perms=num_perms,
                                                                                     graph_num=processed_samples,
                                                                                     new_rng_key=new_rng_key,
                                                                                     extras=extras,
                                                                                     dont_permute=dont_permute)
    t2 = time.time()
    print(f"predicting on perm As took {t2 - t1} sec")
    processed_samples += batch_size
    # ADD
    IDs.extend(batch_IDs)
    As.extend(batch_As)
    perms.extend(batch_perms)
    preds.extend(batch_preds)
    Ss.extend(batch_Ss)
    # breakpoint()
  #outputs = _concat(outputs, axis=0)
  # compare manually
  # outputs[0].data[0]
  # preds[0]
  #breakpoint()
  #good = [i for i in range(len(batch_size)) if (outputs[0].data[i] == preds[i]).all()] # fixme: gotta permute outputs too
  #print('manual eval good length', len(good))
  # arr = outputs[0].data[0] == preds[0]
  # arr.index(False)
  #breakpoint()
  #out = clrs.evaluate(outputs, {'pi': DataPoint(name='pi', location='node', type_='pointer', data=preds)})
  #print('out:', out)

  # breakpoint()
  # preds = _concat(preds, axis=0)
  # WRITE TO FILE
  result_dict = {
    'GraphID': IDs,
    'As': As,
    'Perms': perms,
    'Preds': preds,
    'Ss': Ss
  }
  result_df = pd.DataFrame.from_dict(result_dict)
  # breakpoint()
  alg = extras['algorithm']
  size = len(As[0])
  result_df.to_pickle(path=alg + '_n=' + str(size) + '_testingpermute_' + str(
    extras['step']) + '_.pkl')  # read with pd.read_pickle('filename')

  # print('run.py permuteevalrecord')
  # if extras:
  #   out.update(extras)
  # return {k: unpack(v) for k, v in out.items()}
  return result_df





# -----------------------------------------------------------------------------------------------------------------
# RUN ME!!!
# -----------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
  start_time = time.time()
  # --- LOAD FLAG STUFF
  which = 'dfs'

  if which == 'dfs':
    flagjson = 'WHEREAMI/dfs_flags.json'
    modelname = 'best_dfs.pkl'
  else:
    flagjson = 'WHEREAMI/bellman_ford_flags.json' #'WHEREAMI/dfs_flags.json'
    modelname = 'best_bellman_ford.pkl' #'best_dfs.pkl'


  with open(flagjson, 'r') as f:
    saved_flags = json.load(f)
  FLAGS = SimpleNamespace(**saved_flags) # FIXME: warning this is not the same thing as in run.py, it's a gimmick so that i can use similar code
  algo_idx = 0

  json_time = time.time()
  print(f"json read in {json_time-start_time} seconds")

  model = load_model(modelname, FLAGS)
  load_time = time.time()
  print(f"model loaded in {load_time-json_time} seconds")


  N_RUNS = 5
  instance_stats = {'four':[], 'sixteen':[], 'sixtyfour':[]}
  variety_stats = {'four':[], 'sixteen':[], 'sixtyfour':[]}
  dedup_variety_stats = {'four':[], 'sixteen':[], 'sixtyfour':[]}
  for i in range(N_RUNS):
    # ---------- TEST SAMPLERS?
    four_sampler, test_samples, spec = make_test_sampler(size=4, FLAGS=FLAGS, seed=FLAGS.seed+i)
    time4 = time.time()
    print(f"four sampler built in {time4-load_time} seconds")
    #breakpoint()

    sixteen_sampler, ts, sc = make_test_sampler(size=16, FLAGS=FLAGS, seed=FLAGS.seed+i)
    time16 = time.time()
    print(f"sixteen sampler built in {time16-time4} seconds")

    sixtyfour_sampler, ts2, sc2 = make_test_sampler(size=64, FLAGS=FLAGS, seed=FLAGS.seed+i)
    time64 = time.time()
    print(f"64 sampler built in {time64-time16} seconds")

    common_extras = {'examples_seen': 0000, #current_train_items[algo_idx],
                     'step': 9999,
                     'algorithm': FLAGS.algorithms[algo_idx]}

    rng = np.random.RandomState(FLAGS.seed+i)
    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
    new_rng_key, rng_key = jax.random.split(rng_key)


    four_stats = permute_and_eval(
            four_sampler,
            functools.partial(model.predict, algorithm_index=algo_idx),
            test_samples,
            new_rng_key,
            extras=common_extras, num_perms=5, dont_permute=False)
   # print('n=4 algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

    sixteen_stats = permute_and_eval(
            sixteen_sampler,
            functools.partial(model.predict, algorithm_index=algo_idx),
            test_samples,
            new_rng_key,
            extras=common_extras, num_perms=5, dont_permute=False)
  #  print('n=16 algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

    sixtyfour_stats = permute_and_eval(
            sixtyfour_sampler,
            functools.partial(model.predict, algorithm_index=algo_idx),
            test_samples,
            new_rng_key,
            extras=common_extras, num_perms=5, dont_permute=False)
   # print('n=64 algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)
    time64stats = time.time()
    # -----------------------------------------------------------------------------------------------------------------
    # GIVEN STUFF, REPORT STATS?
    # -----------------------------------------------------------------------------------------------------------------

    
    print('================================================')
    if which == 'dfs':
      df, variety, deduplicated_variety, num_perms = compute_dfs_stats(four_stats)
    else:
      df, variety, deduplicated_variety, num_perms = compute_bf_stats(four_stats)
    time4eval = time.time()
    #print(f"4 stats eval in {time4eval - time64stats} seconds")
    #breakpoint()
    row = df.mean()
    instance_stats['four'].append(row)
    variety_stats['four'].append(variety/num_perms)
    dedup_variety_stats['four'].append(deduplicated_variety / num_perms)
    #breakpoint()

    print('================================================')
    if which == 'dfs':
      df, variety, deduplicated_variety, num_perms = compute_dfs_stats(sixteen_stats)
    else:
      df, variety, deduplicated_variety, num_perms = compute_bf_stats(sixteen_stats)
    time16eval = time.time()
    print(f"16 stats eval in {time16eval - time4eval} seconds")
    row = df.mean()
    instance_stats['sixteen'].append(row)
    variety_stats['sixteen'].append(variety/num_perms)
    dedup_variety_stats['sixteen'].append(deduplicated_variety / num_perms)

    print('================================================')
    if which == 'dfs':
      df, variety, deduplicated_variety, num_perms = compute_dfs_stats(sixtyfour_stats)
    else:
      df, variety, deduplicated_variety, num_perms = compute_bf_stats(sixtyfour_stats)
    time64eval = time.time()
    print(f"64 stats eval in {time64eval - time16eval} seconds")
    row = df.mean()
    instance_stats['sixtyfour'].append(row)
    variety_stats['sixtyfour'].append(variety/num_perms)
    dedup_variety_stats['sixtyfour'].append(deduplicated_variety / num_perms)

  print('================================================')
  print('OVERALL STATS', FLAGS.algorithms[algo_idx])
  print('================================================')
  print(f'num runs: {N_RUNS}\n')

  four = pd.DataFrame(instance_stats['four'])
  print(f'four mean\n----\n{four.mean()}\n----')
  print(f'four std\n----\n{four.std()}\n----')
  varfour = pd.DataFrame(variety_stats['four'])
  print(f'four variety\n----\n{varfour.mean()}\n----')
  print(f'four variety std\n----\n{varfour.std()}\n----')
  dvarfour = pd.DataFrame(dedup_variety_stats['four'])
  print(f'four dvariety\n----\n{dvarfour.mean()}\n----')
  print(f'four dvariety std\n----\n{dvarfour.std()}\n----')
  print('------------------------------')

  sixteen = pd.DataFrame(instance_stats['sixteen'])
  print(f'sixteen mean\n----\n{sixteen.mean()}\n----')
  print(f'sixteen std\n----\n{sixteen.std()}\n----')
  varsixteen = pd.DataFrame(variety_stats['sixteen'])
  print(f'sixteen variety\n----\n{varsixteen.mean()}\n----')
  print(f'sixteen variety std\n----\n{varsixteen.std()}\n----')
  dvarsixteen = pd.DataFrame(dedup_variety_stats['sixteen'])
  print(f'sixteen dvariety\n----\n{dvarsixteen.mean()}\n----')
  print(f'sixteen dvariety std\n----\n{dvarsixteen.std()}\n----')
  print('------------------------------')

  sixtyfour = pd.DataFrame(instance_stats['sixtyfour'])
  print(f'sixtyfour mean\n----\n{sixtyfour.mean()}\n----')
  print(f'sixtyfour std\n----\n{sixtyfour.std()}\n----')
  varsixtyfour = pd.DataFrame(variety_stats['sixtyfour'])
  print(f'sixtyfour variety\n----\n{varsixtyfour.mean()}\n----')
  print(f'sixtyfour variety std\n----\n{varsixtyfour.std()}\n----')
  dvarsixtyfour = pd.DataFrame(dedup_variety_stats['sixtyfour'])
  print(f'sixtyfour dvariety\n----\n{dvarsixtyfour.mean()}\n----')
  print(f'sixtyfour dvariety std\n----\n{dvarsixtyfour.std()}\n----')
  print('------------------------------')
  #breakpoint()