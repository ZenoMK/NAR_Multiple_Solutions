# Try collect_and_eval

import time
import clrs
import jax

from clrs.examples.run import create_samplers, make_multi_sampler, collect_and_eval

import json
import numpy as np
from types import SimpleNamespace # THIS IS SUSSY but bcuz i dont care about flags being absl flags, and just want to use the same values
import functools

from eval_permute_stats import compute_bf_stats, compute_dfs_stats

from dummy_eval import load_model, make_test_sampler


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

model = load_model(modelname)
load_time = time.time()
print(f"model loaded in {load_time-json_time} seconds")

# ---------- TEST SAMPLERS?
four_sampler, test_samples, spec = make_test_sampler(size=4)
time4 = time.time()
print(f"four sampler built in {time4-load_time} seconds")
sixteen_sampler, ts, sc = make_test_sampler(size=16)
time16 = time.time()
print(f"sixteen sampler built in {time16-time4} seconds")

sixtyfour_sampler, ts2, sc2 = make_test_sampler(size=64)
time64 = time.time()
print(f"64 sampler built in {time64-time16} seconds")

common_extras = {'examples_seen': 0000, #current_train_items[algo_idx],
                 'step': 9999,
                 'algorithm': FLAGS.algorithms[algo_idx]}

rng = np.random.RandomState(FLAGS.seed)
rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
new_rng_key, rng_key = jax.random.split(rng_key)


test_stats = collect_and_eval(
        four_sampler,
        functools.partial(model.predict, algorithm_index=algo_idx),
        test_samples,
        new_rng_key,
        extras=common_extras)
print('n=4 algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)


test_stats = collect_and_eval(
        sixteen_sampler,
        functools.partial(model.predict, algorithm_index=algo_idx),
        test_samples,
        new_rng_key,
        extras=common_extras)
print('n=16 algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

test_stats = collect_and_eval(
        sixtyfour_sampler,
        functools.partial(model.predict, algorithm_index=algo_idx),
        test_samples,
        new_rng_key,
        extras=common_extras)
print('n=64 algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)