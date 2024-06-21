from generate_data import generate_data
import matplotlib.pyplot as plt
from sciplotlib import style as spstyle

with plt.style.context(spstyle.get_style('nature-reviews')):
    fig, ax = plt.subplots(ncols =1, sharey = True)

min_size = 5
max_size = 64
means, std = generate_data(graphs_per_size=10)
sizes = [i for i in range(min_size, max_size + 1)]
plt.plot(means)
plt.fill_between(x = sizes, y1 = means + std, y2 = means - std)
plt.show()
