
from batched import *
import matplotlib.pyplot as plt

font = {'size' : 20}
plt.rc('font', **font)

dist_agg = np.loadtxt(f"data/batched_dist.txt")
cov_agg = np.loadtxt("data/batched_cov.txt")

fig, ax = plt.subplots(figsize = (8, 6))
ax2 = ax.twinx()

ax.plot(otherArm, dist_agg, label="BvM distance", color="blue")
ax2.plot(otherArm, cov_agg, label="Coverage", color="orange")
ax2.errorbar(otherArm, cov_agg, stats.norm.ppf(1-errorBarSig/2) * np.sqrt(cov_agg*(1-cov_agg)/numTrials), color="orange")
ax2.hlines(0.95, otherArm.min(), otherArm.max(), color="black", linestyle="dotted")
ax.set_yscale("log")
ax.set_ylabel("BvM TV distance")
ax.set_xlabel("Margin")
ax2.set_ylabel("Coverage")
ax.legend(loc="upper left")
ax2.legend(loc="lower left")
ax.set_title("Thompson Sampling on Batched Bandit")

plt.tight_layout()
plt.savefig("img/batched")