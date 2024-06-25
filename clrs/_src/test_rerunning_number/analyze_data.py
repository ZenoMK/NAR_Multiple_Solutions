from clrs._src.test_rerunning_number.generate_data import generate_data
import matplotlib.pyplot as plt
from sciplotlib import style as spstyle
import numpy as np

with plt.style.context(spstyle.get_style('nature-reviews')):
    fig, ax = plt.subplots(ncols =1, sharey = True)

min_size = 5
max_size = 64
means20_50, std20_50 = generate_data(graphs_per_size=100, num_solutions=[20,50])
print("here1")
plt.plot([i for i in range(min_size, max_size + 1)], means20_50, label = "20 vs 50 re-runs")
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means20_50 + std20_50, y2 = means20_50 - std20_50, alpha = 0.3)

means50_100, std50_100 = generate_data(graphs_per_size=100, num_solutions=[50,100])
plt.plot([i for i in range(min_size, max_size + 1)], means50_100, label = "50 vs 100 re-runs")
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means50_100 + std50_100, y2 = means50_100 - std50_100, alpha = 0.3)
print("here2")
means20_100, std20_100 = generate_data(graphs_per_size=100, num_solutions=[20,100])
plt.plot([i for i in range(min_size, max_size + 1)], means20_100, label = "20 vs 100 re-runs")
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means20_100 + std20_100, y2 = means20_100 - std20_100, alpha = 0.3)
print("here3")
plt.legend(loc = "lower right")
plt.xlabel("Graph sizes")
plt.ylabel("KL-Divergence")
plt.title("KL-div between distributions from varying numbers of DFS re-runs")
plt.savefig("result_DFS.png", bbox_inches = 'tight')
plt.close()

with plt.style.context(spstyle.get_style('nature-reviews')):
    fig, ax = plt.subplots(ncols =1, sharey = True)

plt.tight_layout()
means20_50, std20_50 = generate_data(graphs_per_size=100, num_solutions=[20,50], algorithm='bf')
plt.plot([i for i in range(min_size, max_size + 1)], means20_50, label = "20 vs 50 re-runs")
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means20_50 + std20_50, y2 = means20_50 - std20_50, alpha = 0.3)
print("here4")
means50_100, std50_100 = generate_data(graphs_per_size=100, num_solutions=[50,100], algorithm = 'bf')
plt.plot([i for i in range(min_size, max_size + 1)], means50_100, label = "50 vs 100 re-runs")
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means50_100 + std50_100, y2 = means50_100 - std50_100, alpha = 0.3)
print("here5")
means20_100, std20_100 = generate_data(graphs_per_size=100, num_solutions=[20,100], algorithm = 'bf')
print("here6")
plt.plot([i for i in range(min_size, max_size + 1)], means20_100, label = "20 vs 100 re-runs")
plt.fill_between(x = [i for i in range(min_size, max_size + 1)], y1 = means20_100 + std20_100, y2 = means20_100 - std20_100, alpha = 0.3)
plt.legend(loc = "lower right")
plt.xlabel("Graph sizes")
plt.ylabel("KL-Divergence")
plt.title("KL-div between distributions from varying numbers of DFS re-runs")
plt.savefig("result_BF.png", bbox_inches = 'tight')
