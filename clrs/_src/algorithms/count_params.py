# given a jax params dictionary, such as eval_model.params in run.py
# see that dictionaries inside dictionaries (leaves are np.arrays)

train_model.params = ...

minitree = jax.tree_util.tree_map(lambda x: np.prod(x.shape), train_model.params)
params = jax.tree_util.tree_reduce(operator.add, minitree)
print(params)


# to count the number of message-passing steps, count the number of times probes are pushed