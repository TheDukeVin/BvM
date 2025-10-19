
import numpy as np
import time
import scipy
import scipy.special as sc

np.random.seed(42)

GAUSSIAN_BANDIT_VAR = 1

n_rounds = 10000
n_sims = 10000

# Smaller scale:
# n_rounds = 100
# n_sims = 100

gaussian_means = [
    [0, 0],
    [0, 0.1], 
    [0, 0.2], 
    [0, 0.3], 
    [0, 0.4]
]

bernoulli_means = [
    [0.3, 0.5],
    [0.4, 0.5],
    [0.5, 0.5],
    [0.5, 0.6],
    [0.5, 0.7]
]

poisson_means = [
    [0.2, 1],
    [0.5, 1],
    [1, 1],
    [1, 2],
    [1, 5]
]

all_configs = [
    ('gaussian', gaussian_means, (np.full(2, 0), np.full(2, 1))),
    ('bernoulli', bernoulli_means, (np.full(2, 1), np.full(2, 1))),
    ('poisson', poisson_means, (np.full(2, 1), np.full(2, 1)))
]

def runBandit(armMeans, kind, alg, n_rounds, n_sims, priorParams, alpha):

    n_arms = len(armMeans)
    numPulls = np.full((n_sims, n_arms), 1)
    sumReward = np.zeros((n_sims, n_arms))

    tv_est = np.zeros(n_rounds)
    tv_se = np.zeros(n_rounds)

    coverage = np.zeros((n_rounds, n_arms))

    for t in range(n_rounds):
        if alg == 'ucb':
            chosenArms = np.argmax(sumReward / numPulls + np.sqrt(2*np.log(t+1) / numPulls), axis=1)
        if alg == 'unif':
            chosenArms = np.full(n_sims, t % n_arms)
        
        if kind == 'gaussian':
            observedReward = np.random.normal(loc=armMeans[chosenArms], scale=np.sqrt(GAUSSIAN_BANDIT_VAR))
        elif kind == 'bernoulli':
            observedReward = np.random.uniform(size=n_sims) < armMeans[chosenArms]
        elif kind == 'poisson':
            observedReward = np.random.poisson(lam=armMeans[chosenArms])
        sumReward[np.arange(n_sims), chosenArms] += observedReward
        numPulls[np.arange(n_sims), chosenArms] += 1

        BvMmean = sumReward / numPulls
        if kind == 'gaussian':
            BvMvar = GAUSSIAN_BANDIT_VAR / numPulls
        elif kind == 'bernoulli':
            BvMvar = BvMmean * (1-BvMmean) / numPulls
        elif kind == 'poisson':
            BvMvar = BvMmean / numPulls

        BvMmean = np.nan_to_num(BvMmean, nan=0)
        BvMvar = np.nan_to_num(BvMvar, nan=1)
        BvMvar[BvMvar == 0] = 1

        samples = np.random.normal(loc=BvMmean, scale=np.sqrt(BvMvar))

        BvMpdf = np.prod(1/np.sqrt(2 * np.pi * BvMvar) * np.exp(-np.power(samples - BvMmean, 2) / (2 * BvMvar)), axis=1)

        assert not np.any(np.isnan(samples))
        assert not np.any(np.isnan(BvMpdf))

        if kind == 'gaussian':
            priorMean, priorVar = priorParams
            posteriorVar = 1 / (1 / BvMvar + 1/priorVar)
            posteriorMean = posteriorVar * (BvMmean / BvMvar + priorMean / priorVar)
        
            posteriorpdf = np.prod(1/np.sqrt(2 * np.pi * posteriorVar) * np.exp(-np.power(samples - posteriorMean, 2) / (2 * posteriorVar)), axis=1)

            q = scipy.stats.norm.ppf(1-alpha/2)
            coverage[t] = (np.abs((posteriorMean - armMeans) / np.sqrt(posteriorVar)) < q).mean(axis=0)
        
        elif kind == 'bernoulli':
            priorAlpha, priorBeta = priorParams
            posteriorAlpha = sumReward + priorAlpha[None, :]
            posteriorBeta = numPulls - sumReward + priorBeta[None, :]

            samples[samples <= 0] = 1e-05
            samples[samples >= 1] = 1-1e-05

            logpdf = np.sum(np.log(samples)*(posteriorAlpha-1) + np.log(1-samples)*(posteriorBeta-1) 
                            + sc.loggamma(posteriorAlpha + posteriorBeta) - sc.loggamma(posteriorAlpha) - sc.loggamma(posteriorBeta), axis=1)
            posteriorpdf = np.exp(logpdf)

            quantile_of_true = scipy.stats.beta.cdf(armMeans[None, :], posteriorAlpha, posteriorBeta)
            coverage[t] = ((alpha/2 < quantile_of_true) * (quantile_of_true < 1-alpha/2)).mean(axis=0)
        
        elif kind == 'poisson':
            priorAlpha, priorBeta = priorParams
            posteriorAlpha = sumReward + priorAlpha[None, :]
            posteriorBeta = numPulls + priorBeta[None, :]

            samples[samples <= 0] = 1e-05
            logpdf = np.sum(np.log(samples)*(posteriorAlpha-1) - samples*posteriorBeta 
                            + np.log(posteriorBeta)*posteriorAlpha - sc.loggamma(posteriorAlpha), axis=1)
            posteriorpdf = np.exp(logpdf)

            quantile_of_true = scipy.stats.gamma.cdf(armMeans[None, :], a=posteriorAlpha, scale=1/posteriorBeta)
            coverage[t] = ((alpha/2 < quantile_of_true) * (quantile_of_true < 1-alpha/2)).mean(axis=0)

        tmp = 1 - posteriorpdf / BvMpdf

        tmp[tmp < 0] = 0

        tv_est[t] = tmp.mean()
        tv_se[t] = tv_est.std() / n_sims

        

    print("Standard Error Ratio: " + str((tv_se/tv_est).max()))
    return tv_est, tv_se, coverage

if __name__ == '__main__':

    start_time = time.time()
    print("Running vanilla bandit...")

    for kind, means, priors in all_configs:
        for i, config in enumerate(means):
            tv_est, tv_se, coverage = runBandit(armMeans=np.array(config),
                                    kind=kind,
                                    alg='ucb',
                                    n_rounds=n_rounds,
                                    n_sims=n_sims,
                                    priorParams=priors,
                                    alpha=0.05)
            np.savetxt(f"data/vanilla_{kind}_{i}_TV", tv_est)
            np.savetxt(f"data/vanilla_{kind}_{i}_TVSE", tv_se)
            np.savetxt(f"data/vanilla_{kind}_{i}_coverage", coverage)

    print("Finished running vanilla bandit, Time: " + str(time.time() - start_time))