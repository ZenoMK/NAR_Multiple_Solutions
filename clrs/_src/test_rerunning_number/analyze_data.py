from generate_data import generate_data
import matplotlib.pyplot as plt
from sciplotlib import style as spstyle
import numpy as np

with plt.style.context(spstyle.get_style('nature-reviews')):
    fig, ax = plt.subplots(ncols =1, sharey = True)

min_size = 5
max_size = 64
means, std = generate_data(graphs_per_size=100)
data = [[i,means[i-min_size]] for i in range(min_size, max_size + 1)]
plt.plot([i for i in range(min_size, max_size + 1)], means)
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means + std, y2 = means - std, alpha = 0.3)
plt.savefig("result.png")
