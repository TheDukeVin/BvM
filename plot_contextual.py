
from contextual import *
import matplotlib.pyplot as plt

font = {'size' : 20}
plt.rc('font', **font)

tv = np.loadtxt('data/contextual_tv.txt')
tv_se = np.loadtxt('data/contextual_tvse.txt')

fig, ax = plt.subplots(figsize = (8, 6))

labels = ["convex", "dominated", "duplicate"]

print("Standard Error Ratio: " + str((tv_se / tv).max()))


for i in range(3):
    ax.plot(tv[i, :], label=labels[i])
plt.legend()
ax.set_yscale('log')
ax.set_xlabel("Time")
ax.set_ylabel("BvM TV distance")
ax.set_title("Lin-UCB on Contextual Bandit")
plt.tight_layout()
plt.savefig("img/contextual")