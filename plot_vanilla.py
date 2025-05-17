
from vanilla import *
import matplotlib.pyplot as plt

font = {'size' : 20}
plt.rc('font', **font)

for kind, means, priors in all_configs:
    fig, ax = plt.subplots(figsize = (8, 6))
    for i, config in enumerate(means):
        tv_est = np.loadtxt(f"data/vanilla_{kind}_{i}_TV")
        tv_se = np.loadtxt(f"data/vanilla_{kind}_{i}_TVSE")
        ax.plot(tv_est, label=config)
    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel('TV dist')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/vanilla_{kind}")