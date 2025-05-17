
import numpy as np
from scipy import stats
import time

np.random.seed(42)

GAUSSIAN_BANDIT_VAR = 1

numTrials = int(2e+05)

# Smaller scale:
# numTrials = int(2e+03)

errorBarSig = 0.05
batchSize = int(1e+04)

otherArm = np.arange(-0.3, 0.301, 0.02)

class TwoStageGaussianBandit:
    def __init__(self, means):
        self.means = means
        self.K = len(self.means)
        assert self.K == 2
        self.priorMean = np.full(self.K, 0)
        self.priorVariance = np.full(self.K, 1)
    
    def getSampleProb(self, mean, var):
        # Find probability of pulling arm 0
        return stats.norm.cdf((mean[0] - mean[1]) / np.sqrt(var[0] + var[1]))
    
    def getSamples(self, samplesPerBatch, arm0Prob):
        sizes = stats.multinomial.rvs(samplesPerBatch, np.array([arm0Prob, 1-arm0Prob]))
        sums = stats.norm.rvs(loc=self.means * sizes, scale=np.sqrt(GAUSSIAN_BANDIT_VAR * sizes))
        return sizes, sums
    
    def getPosterior(self, sizes, sums):
        posteriorVar = 1/(sizes/GAUSSIAN_BANDIT_VAR + 1/self.priorVariance)
        posteriorMean = posteriorVar * (sums/GAUSSIAN_BANDIT_VAR + self.priorMean/self.priorVariance)
        return posteriorMean, posteriorVar
    
    def runThompson(self, samplesPerBatch):
        p = self.getSampleProb(self.priorMean, self.priorVariance)
        batch1Size, batch1Sums = self.getSamples(samplesPerBatch, p)
        posteriorMean, posteriorVar = self.getPosterior(batch1Size, batch1Sums)
        p2 = self.getSampleProb(posteriorMean, posteriorVar)
        batch2Size, batch2Sums = self.getSamples(samplesPerBatch, p2)

        totalSize = batch1Size + batch2Size
        totalSums = batch1Sums + batch2Sums
        posteriorMean, posteriorVar = self.getPosterior(totalSize, totalSums)

        # Get BvM distance:

        numBvMsamples = 5
        BvMmean = totalSums / totalSize
        BvMvar = GAUSSIAN_BANDIT_VAR / totalSize

        samples = stats.norm.rvs(loc=BvMmean, scale=np.sqrt(BvMvar), size=(numBvMsamples, self.K))
        BvMpdf = np.prod(stats.norm.pdf(samples, loc=BvMmean, scale=np.sqrt(BvMvar)), axis=1)
        posteriorpdf = np.prod(stats.norm.pdf(samples, loc=posteriorMean, scale=np.sqrt(posteriorVar)), axis=1)
        temp = 1 - posteriorpdf / BvMpdf
        temp[temp < 0] = 0
        BvMdist = np.mean(temp)

        # Get coverage of margin

        alpha = 0.05

        postMarginMean = posteriorMean[0] - posteriorMean[1]
        postMarginVar = posteriorVar[0] + posteriorVar[1]
        marginLeft = postMarginMean - np.sqrt(postMarginVar) * stats.norm.ppf(1-alpha/2)
        marginRight = postMarginMean + np.sqrt(postMarginVar) * stats.norm.ppf(1-alpha/2)

        trueMargin = self.means[0] - self.means[1]

        coverage = marginLeft < trueMargin and trueMargin < marginRight

        return BvMdist, coverage

def simulate(bandit, numTrials, batchSize):
    dists = []
    coverages = []
    for i in range(numTrials):
        dist, coverage = bandit.runThompson(batchSize)
        dists.append(dist)
        coverages.append(coverage)
    dists = np.array(dists)
    print("Standard Error Ratio: " + str(dists.std()/np.sqrt(numTrials) / dists.mean()))
    return dists.mean(), np.array(coverages).mean()

if __name__ == '__main__':

    start_time = time.time()
    print("Running batched bandit...")

    dist_agg = []
    cov_agg = []
    for other in otherArm:
        bandit = TwoStageGaussianBandit(np.array([0, other]))
        dist, cov = simulate(bandit, numTrials, batchSize)
        dist_agg.append(dist)
        cov_agg.append(cov)

    dist_agg = np.array(dist_agg)
    cov_agg = np.array(cov_agg)

    np.savetxt(f"data/batched_dist.txt", np.array(dist_agg))
    np.savetxt(f"data/batched_cov.txt", np.array(cov_agg))

    print(f"Finished running batched bandit, Time: {time.time() - start_time}")