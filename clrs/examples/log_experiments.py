import numpy as np
import clrs     # for clrs.evaluate
import jax
import pandas as pd

import clrs._src.dfs_sampling as dfs_sampling
from clrs._src import dfs_uniqueness_check
from clrs._src.algorithms import check_graphs
from clrs._src.algorithms.BF_beamsearch import sample_beamsearch
#from clrs.examples.run import _concat, unpack  # circular import error!

###############################################################
# Methods needed, copy-pasted from run.py :(
###############################################################
def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)

def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


###############################################################
# BF pipeline
###############################################################

def BF_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras, filename='bf_accuracy'):
    """Collect batches of output and hint preds and evaluate them."""
    
    processed_samples = 0
    preds = []
    outputs = []
    As = []
    source_nodes = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        cur_preds, _ = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        processed_samples += batch_size
        As.append(feedback[0][0][2].data)
        source_nodes.append(
            np.argmax(feedback[0][0][1].data, axis=1))  # todo checkme. feedback[0][0][1] is 's', .data gives array
    outputs = _concat(outputs, axis=0)
    As = _concat(As, axis=0)  # concatenate batches
    source_nodes = _concat(source_nodes, axis=0)
    # breakpoint()
    preds = _concat(preds, axis=0)
    out = clrs.evaluate(outputs, preds)
    #breakpoint()

    ########
    # RANDOM #
    ########
    model_sample_random = dfs_sampling.sample_random_list([preds])
    true_sample_random = dfs_sampling.sample_random_list(outputs)

    model_random_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_random[i]) for i in range(len(model_sample_random))]
    correctness_model_random = sum(model_random_truthmask) / len(model_random_truthmask)

    true_random_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_random[i]) for i in range(len(true_sample_random))]
    correctness_true_random = sum(true_random_truthmask) / len(true_random_truthmask)

    #breakpoint()
    ########
    # BEAM #
    ########
    model_sample_beam = sample_beamsearch(As, source_nodes, [preds])
    true_sample_beam = sample_beamsearch(As, source_nodes, outputs)

    model_beam_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_beam[i]) for i in range(len(model_sample_beam))]
    correctness_model_beam = sum(model_beam_truthmask) / len(model_beam_truthmask)

    true_beam_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_beam[i]) for i in range(len(model_sample_random))]
    correctness_true_beam = sum(true_beam_truthmask) / len(true_beam_truthmask)
    
    #breakpoint()
    ########
    # Argmax #
    ########
    model_sample_argmax = dfs_sampling.sample_argmax([preds]) # doesn't use A or s
    true_sample_argmax = dfs_sampling.sample_argmax(outputs)

    model_argmax_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_argmax[i]) for i in
                            range(len(model_sample_argmax))]
    correctness_model_argmax = sum(model_argmax_truthmask) / len(model_argmax_truthmask)

    true_argmax_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_argmax[i]) for i in
                           range(len(model_sample_random))]
    correctness_true_argmax = sum(true_argmax_truthmask) / len(true_argmax_truthmask)

    breakpoint()
    ########
    # greedy beam #
    ########



    ### LOGGING ###
    As = [i.flatten() for i in As]
    result_dict = {"As": As,
                   #
                   "Argmax_Model_Trees": model_sample_argmax,
                   "Argmax_True_Trees": true_sample_argmax,
                   #
                   "Argmax_Model_Mask": model_argmax_truthmask,
                   "Argmax_True_Mask": true_argmax_truthmask,
                   #
                   "Argmax_Model_Accuracy": correctness_model_argmax,
                   "Argmax_True_Accuracy": correctness_true_argmax,
                   #
                   ###
                   #
                   "Random_Model_Trees": model_sample_random,
                   "Random_True_Trees": true_sample_random,
                   #
                   "Random_Model_Mask": model_random_truthmask,
                   "Random_True_Mask": true_random_truthmask,
                   #
                   "Random_Model_Accuracy": correctness_model_random,
                   "Random_True_Accuracy": correctness_true_random,
                   #
                   ###
                   #
                   "Beam_Model_Trees": model_sample_beam,
                   "Beam_True_Trees": true_sample_beam,
                   #
                   "Beam_Model_Mask": model_beam_truthmask,
                   "Beam_True_Mask": true_beam_truthmask,
                   #
                   "Beam_Model_Accuracy": correctness_model_beam,
                   "Beam_True_Accuracy": correctness_true_beam,
                   #
                   }
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv(filename + '.csv', encoding='utf-8', index=False)

    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}

"""

###############################################################
# DFS
###############################################################

def DFS_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras, filename = 'dfs_accuracy'):
    """Collect batch of output preds and evaluate them."""
    processed_samples = 0
    preds = []
    outputs = []
    As = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        cur_preds, _ = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        processed_samples += batch_size
        As.append(feedback[0][0][1].data)
    outputs = _concat(outputs, axis=0)
    As = _concat(As, axis=0)  # concatenate batches
    # breakpoint()

    ### We need preds and A. We want to
    # 1. Sample from preds a candidate tree
    # 2. run check_graphs on candidate tree (using A as groundtruth)
    # 3. Collect validity result into a dataframe.

    ##### RANDOM
    model_sample_random = dfs_sampling.sample_random_list(preds)
    true_sample_random = dfs_sampling.sample_random_list(outputs)

    model_random_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_random[i]) for i in
                              range(len(model_sample_random))]
    correctness_model_random = sum(model_random_truthmask) / len(model_random_truthmask)

    true_random_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_random[i]) for i in
                             range(len(true_sample_random))]
    correctness_true_random = sum(true_random_truthmask) / len(true_random_truthmask)

    ##### ARGMAX
    ## remember to convert from jax arrays to lists for easy subsequent methods using .tolist()
    model_sample_argmax = dfs_sampling.sample_argmax_listofdict(preds)
    true_sample_argmax = dfs_sampling.sample_argmax_listofdatapoint(outputs)

    # compute the fraction of trees sampled from model output fulfilling the necessary conditions
    model_argmax_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_argmax[i].tolist()) for i in
                              range(len(model_sample_argmax))]
    correctness_model_argmax = sum(model_argmax_truthmask) / len(model_argmax_truthmask)

    # compute the fraction of trees sampled from true distributions fulfilling the necessary conditions
    true_argmax_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_argmax[i].tolist()) for i in
                             range(len(true_sample_argmax))]
    correctness_true_argmax = sum(true_argmax_truthmask) / len(true_argmax_truthmask)

    ##### UPWARDS
    model_sample_upwards = dfs_sampling.sample_upwards(preds)
    true_sample_upwards = dfs_sampling.sample_upwards(outputs)

    model_upwards_uniques, model_upwards_valids = dfs_uniqueness_check.check_uniqueness_dfs(preds)
    true_upwards_uniques, true_upwards_valids = dfs_uniqueness_check.check_uniqueness_dfs(outputs)

    model_upwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_upwards[i].astype(int)) for i in
                               range(len(model_sample_upwards))]
    correctness_model_upwards = sum(model_upwards_truthmask) / len(model_upwards_truthmask)

    true_upwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_upwards[i].astype(int)) for i in
                              range(len(true_sample_upwards))]
    correctness_true_upwards = sum(true_upwards_truthmask) / len(true_upwards_truthmask)

    ##### ALTUPWARDS
    model_sample_altUpwards = dfs_sampling.sample_altUpwards(preds)
    true_sample_altUpwards = dfs_sampling.sample_altUpwards(outputs)

    model_altUpwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_altUpwards[i].astype(int)) for i
                                  in
                                  range(len(model_sample_altUpwards))]
    correctness_model_altUpwards = sum(model_altUpwards_truthmask) / len(model_altUpwards_truthmask)

    true_altUpwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_altUpwards[i].astype(int)) for i in
                                 range(len(true_sample_altUpwards))]
    correctness_true_altUpwards = sum(true_altUpwards_truthmask) / len(true_altUpwards_truthmask)

    # breakpoint()
    As = [i.flatten() for i in As]
    result_dict = {"As": As,
                   #
                   "Argmax_Model_Trees": model_sample_argmax,
                   "Argmax_True_Trees": true_sample_argmax,
                   #
                   "Argmax_Model_Mask": model_argmax_truthmask,
                   "Argmax_True_Mask": true_argmax_truthmask,
                   #
                   "Argmax_Model_Accuracy": correctness_model_argmax,
                   "Argmax_True_Accuracy": correctness_true_argmax,
                   #
                   ###
                   #
                   "Random_Model_Trees": model_sample_random,
                   "Random_True_Trees": true_sample_random,
                   #
                   "Random_Model_Mask": model_random_truthmask,
                   "Random_True_Mask": true_random_truthmask,
                   #
                   "Random_Model_Accuracy": correctness_model_random,
                   "Random_True_Accuracy": correctness_true_random,
                   #
                   ###
                   #
                   "Upwards_Model_Trees": model_sample_upwards,
                   "Upwards_True_Trees": true_sample_upwards,
                   #
                   "Upwards_Model_Mask": model_upwards_truthmask,
                   "Upwards_True_Mask": true_upwards_truthmask,
                   #
                   "Upwards_Model_Accuracy": correctness_model_upwards,
                   "Upwards_True_Accuracy": correctness_true_upwards,

                   "Upwards_Model_Uniques": model_upwards_uniques,
                   "Upwards_Model_Valids": model_upwards_valids,

                   "Upwards_True_Uniques": true_upwards_uniques,
                   "Upwards_True_Valids": true_upwards_valids,
                   #
                   ###
                   #
                   "altUpwards_Model_Trees": model_sample_altUpwards,
                   "altUpwards_True_Trees": true_sample_altUpwards,
                   #
                   "altUpwards_Model_Mask": model_altUpwards_truthmask,
                   "altUpwards_True_Mask": true_altUpwards_truthmask,
                   #
                   "altUpwards_Model_Accuracy": correctness_model_altUpwards,
                   "altUpwards_True_Accuracy": correctness_true_altUpwards,
                   }
    # breakpoint()
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv(filename + '.csv', encoding='utf-8', index=False)

    # As[0].reshape((np.sqrt(len(lAs[0])).astype(int)), np.sqrt(len(lAs[0])).astype(int))

    preds = _concat(preds, axis=0)
    out = clrs.evaluate(outputs, preds)

    # breakpoint()
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}