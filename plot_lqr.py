
from lqr import *
import matplotlib.pyplot as plt

font = {'size' : 20}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize = (8, 6))

labels = ["determined", "stabilizable", "unstabilizable"]

for i in range(3):
    mean_TV = np.loadtxt(f'data/lqr_tv{i}.txt')
    ax.plot(mean_TV, label=labels[i])

ax.set_yscale('log')
ax.set_xlabel('Time')
ax.set_ylabel('BvM TV distance')
ax.set_title("NCEC on LQR")
plt.legend()
plt.tight_layout()
plt.savefig("img/lqr")