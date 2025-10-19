
import pandas as pd
import numpy as np
import scipy

N_dfs = 124

n_arms = 80

bvm_sample_size = 10000

calc_period = 20000

priorAlpha = np.ones(n_arms)
priorBeta = np.ones(n_arms)

def computeBvM(counts, sums):
    sums = np.tile(sums, (bvm_sample_size, 1))
    counts = np.tile(counts, (bvm_sample_size, 1))
    BvMmean = sums / counts
    BvMvar = BvMmean * (1 - BvMmean) / counts

    samples = np.random.normal(loc=BvMmean, scale=np.sqrt(BvMvar))

    BvMpdf = np.prod(1/np.sqrt(2 * np.pi * BvMvar) * np.exp(-np.power(samples - BvMmean, 2) / (2 * BvMvar)), axis=1)


    posteriorAlpha = sums + priorAlpha[None, :]
    posteriorBeta = counts - sums + priorBeta[None, :]

    samples[samples <= 0] = 1e-05
    samples[samples >= 1] = 1-1e-05

    posteriorpdf = np.prod(scipy.stats.beta.pdf(samples, a=posteriorAlpha, b=posteriorBeta), axis=1)

    tmp = 1 - posteriorpdf / BvMpdf

    tmp[tmp < 0] = 0

    tv_est = tmp.mean()
    tv_se = tmp.std() / np.sqrt(bvm_sample_size)

    return tv_est, tv_se


for i in range(1, N_dfs+1):

    df = pd.read_csv(f'states/state{i}.csv')

    bvm = []

    for j in range(0, len(df), calc_period):
        counts = df.iloc[j][[f'count{i}' for i in range(n_arms)]]
        counts += 1
        sums = df.iloc[j].iloc[n_arms:][[f'sum{i}' for i in range(n_arms)]]
        sums += 0.5
        est, se = computeBvM(np.array(counts), np.array(sums))
        bvm.append([np.sum(counts), est, se])
    
    bvm = pd.DataFrame(bvm, columns=['Time', 'BvM', 'SE'])

    bvm.to_csv(f'bvm/bvm{i}.csv')