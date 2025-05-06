# THIS IS MEANT TO, GIVEN A TRAINED MODEL, ALLOW YOU TO TEST IT, like the bottom of run.py
# UNFORTUNATELY, IT'S ALMOST AS BIG (1/7th) AS RUN.PY and VERY JANK
# FORTUNATELY, i did my best :')

# TODO: is it worth making this handle multiple algorithms? or is 1 at a time fine
# TODO: is it worth it making test samplers alongside val samplers bcuz common args. CAN YOU JUST MAKE TEST? and use it to init?

# need:
# spec list # fixme: this just seems like the types for all the probe stuff in hints
# dummy traj # fixme: how important is this stuff? Is it the data for eval?
# model params # fixme: these get updated during training, restore_model overwrites them again.
# ----
import time
# the advantage of this file is, you kaggle-train a model, then save it
# then you can run this to load that model,
  # shuffle in different ways
  # and report the outcomes


import clrs
import jax

from clrs.examples.run import create_samplers, make_multi_sampler, permute_eval_and_record

import json
import numpy as np
from types import SimpleNamespace # THIS IS SUSSY but bcuz i dont care about flags being absl flags, and just want to use the same values
import functools

from eval_permute_stats import compute_bf_stats, compute_dfs_stats

start_time = time.time()
# --- LOAD FLAG STUFF
flagjson = 'WHEREAMI/dfs_flags.json'
modelname = 'best_dfs.pkl'

with open(flagjson, 'r') as f:
  saved_flags = json.load(f)
FLAGS = SimpleNamespace(**saved_flags) # FIXME: warning this is not the same thing as in run.py, it's a gimmick so that i can use similar code
algo_idx = 0

json_time = time.time()
print(f"json read in {json_time-start_time} seconds")
# -----------------------------------------------------------------------------------------------------------------
# LOAD THE SCAFFOLD FOR AN EVAL MODEL (CORRECT DIMENSIONS, etc)
# -----------------------------------------------------------------------------------------------------------------
# ---- make spec list and val_samplers (necessary for dimensions of model)
def load_model(modelname):
      if FLAGS.hint_mode == 'encoded_decoded':
            encode_hints = True
            decode_hints = True
      elif FLAGS.hint_mode == 'decoded_only':
            encode_hints = False
            decode_hints = True
      elif FLAGS.hint_mode == 'none':
            encode_hints = False
            decode_hints = False
      else:
            raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

      train_lengths = [int(x) for x in FLAGS.train_lengths]

      rng = np.random.RandomState(FLAGS.seed)
      rng_key = jax.random.PRNGKey(rng.randint(2**32))

      # Create samplers
      algo_idx = 0 # fixme: ONLY LOADING ONE FOR NOW
      algorithm = FLAGS.algorithms[algo_idx]
      common_sampler_args = dict(
                algorithm=FLAGS.algorithms[algo_idx],
                rng=rng,
                enforce_pred_as_input=FLAGS.enforce_pred_as_input,
                enforce_permutations=FLAGS.enforce_permutations,
                chunk_length=FLAGS.chunk_length,
                )
      length_needle = FLAGS.length_needle
      p = tuple([0.1 + 0.1 * i for i in range(9)])
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')
      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(sizes=[np.amax(train_lengths)],
                      split='val',
                      batch_size=32,
                      multiplier=2 * mult,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler(**val_args)

      # ---- load model params via flags
      processor_factory = clrs.get_processor_factory(
            FLAGS.processor_type,
            use_ln=FLAGS.use_ln,
            nb_triplet_fts=FLAGS.nb_triplet_fts,
            nb_heads=FLAGS.nb_heads
      )
      model_params = dict(
            processor_factory=processor_factory,
            hidden_dim=FLAGS.hidden_size,
            encode_hints=encode_hints,
            decode_hints=decode_hints,
            encoder_init=FLAGS.encoder_init,
            use_lstm=FLAGS.use_lstm,
            learning_rate=FLAGS.learning_rate,
            grad_clip_max_norm=FLAGS.grad_clip_max_norm,
            checkpoint_path=FLAGS.checkpoint_path,
            freeze_processor=FLAGS.freeze_processor,
            dropout_prob=FLAGS.dropout_prob,
            hint_teacher_forcing=FLAGS.hint_teacher_forcing,
            hint_repred_mode=FLAGS.hint_repred_mode,
            nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
            )


      eval_model = clrs.models.BaselineModel(
          spec=[spec], #only need one spec for one algorithm
          dummy_trajectory=[next(val_sampler)], #similarly, only one alg at a time
          **model_params
      )
      dummy_fts = [next(val_sampler).features]
      eval_model.init(dummy_fts, FLAGS.seed +1)

      eval_model.restore_model(file_name=modelname)
      return eval_model

model = load_model(modelname)
load_time = time.time()
print(f"model loaded in {load_time-json_time} seconds")
# -----------------------------------------------------------------------------------------------------------------
# GIVEN THAT AN EVAL MODEL IS LOADED, TRY SOME PREDICTIONS, COLLECT SOME STATS???
# -----------------------------------------------------------------------------------------------------------------
# TODO: NEED A TEST SAMPLER
def make_test_sampler(size):
  algo_idx = 0
  algorithm = FLAGS.algorithms[algo_idx]
  mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
  common_sampler_args = dict(
        algorithm=FLAGS.algorithms[algo_idx],
        rng=rng,
        enforce_pred_as_input=FLAGS.enforce_pred_as_input,
        enforce_permutations=FLAGS.enforce_permutations,
        chunk_length=FLAGS.chunk_length,
  )
  test_args = dict(sizes=[size],#[16], #Fixme: this line is important || old was [-1],
                         split='test',
                         batch_size=32,
                         multiplier=2 * mult,
                         randomize_pos=False,
                         chunked=False,
                         sampler_kwargs={},
                         **common_sampler_args)
  test_sampler, test_samples, spec = make_multi_sampler(**test_args)
  return test_sampler, test_samples, spec


four_sampler, test_samples, spec = make_test_sampler(size=4)
time4 = time.time()
print(f"four sampler built in {time4-load_time} seconds")
sixteen_sampler, ts, sc = make_test_sampler(size=16)
time16 = time.time()
print(f"sixteen sampler built in {time16-time4} seconds")
sixtyfour_sampler, ts2, sc2 = make_test_sampler(size=64)
time64 = time.time()
print(f"64 sampler built in {time64-time16} seconds")

common_extras = {'examples_seen': 17291729, #current_train_items[algo_idx],
                 'step': 69420,
                 'algorithm': FLAGS.algorithms[algo_idx]}

rng = np.random.RandomState(FLAGS.seed)
rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))
new_rng_key, rng_key = jax.random.split(rng_key)
#breakpoint()


four_stats = permute_eval_and_record(
      sampler=four_sampler, #test_samplers[algo_idx],
      predict_fn=functools.partial(model.predict, algorithm_index=algo_idx),
      sample_count=test_samples, #test_sample_counts[algo_idx],
      rng_key=new_rng_key,
      extras=common_extras)
time4stats = time.time()
print(f"4 stats built in {time4stats-time64} seconds")
#compute_dfs_stats(four_stats)

sixteen_stats = permute_eval_and_record(
      sampler=sixteen_sampler, #test_samplers[algo_idx],
      predict_fn=functools.partial(model.predict, algorithm_index=algo_idx),
      sample_count=test_samples, #test_sample_counts[algo_idx],
      rng_key=new_rng_key,
      extras=common_extras)
time16stats = time.time()
print(f"16 stats built in {time16stats-time4stats} seconds")

#breakpoint()
# OOOPS THIS NEXT THING TAKES LIKE 2 HOURS ON MY MACHINE BCUZ GETTING FORWARD PASSES ON N=64 IS APPARENTLY TOO BIG || YOU WANT 2 GPU IT, save pickle, load n run henry CPU
sixtyfour_stats = permute_eval_and_record(
      sampler=sixtyfour_sampler, #test_samplers[algo_idx],
      predict_fn=functools.partial(model.predict, algorithm_index=algo_idx),
      sample_count=test_samples, #test_sample_counts[algo_idx],
      rng_key=new_rng_key,
      extras=common_extras)
time64stats = time.time()
print(f"64 stats built in {time64stats-time16stats} seconds")

# TODO: put in different permute methods for permuting in different ways?

# IF YOU WANT A DIFFERENT GRAPH SIZE, YOU NEED A DIFFERENT SAMPLER? bcuz you call next(sampler) to get the feedback.features to do the predictions

# -----------------------------------------------------------------------------------------------------------------
# GIVEN STUFF, REPORT STATS?
# -----------------------------------------------------------------------------------------------------------------

print('================================================')
compute_dfs_stats(four_stats)
time4eval = time.time()
print(f"4 stats eval in {time4eval-time16stats} seconds")

print('================================================')
compute_dfs_stats(sixteen_stats)
time16eval = time.time()
print(f"16 stats built in {time16eval-time4eval} seconds")

print('================================================')
compute_dfs_stats(sixtyfour_stats)
time64eval = time.time()
print(f"64 stats built in {time64eval-time16eval} seconds")