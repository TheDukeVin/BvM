

import numpy as np
import time

np.random.seed(42)

n_sims = 1000
n_rounds = 1000

# Smaller scale:
# n_sims = 100
# n_rounds = 100


n_arms = 3
d = 2
alpha = 0.1

SAMPLING_STD = 1

theta1 = np.array([
    [1, 0],
    [0, 1],
    [-1, 0]
])

theta2 = np.array([
    [1, 0],
    [0, 0],
    [-1, 0]
])

theta3 = np.array([
    [1, 0],
    [1, 0],
    [-1, 0]
])

allTheta = [theta1, theta2, theta3]

def runContextual(n_sims, n_arms, d, n_rounds, alpha, theta):

    # Shape (n_sims, n_arms, d, d)
    A = np.eye(d)[None, None, :, :] * np.ones((n_sims, n_arms, 1, 1))
    # Shape (n_sims, n_arms, d)
    b = np.zeros((n_sims, n_arms, d))

    armCounts = np.zeros((n_sims, n_arms))

    priorMean = np.zeros((n_arms, d))
    priorVar = np.eye(d)[None, :, :] * np.ones((n_arms, d, d))

    TV_est = np.zeros(n_rounds)
    TV_se = np.zeros(n_rounds)
    minPulls = np.zeros(n_rounds)
    diff_mean_est = np.zeros(n_rounds)
    diff_var_est = np.zeros(n_rounds)

    for t in range(n_rounds):
        context = np.random.normal(loc=0, scale=1, size=(n_sims, d))

        theta_hat = np.matmul(np.linalg.inv(A), b[:, :, :, None]).reshape((n_sims, n_arms, d))

        tile_context = context[:, None, :] * np.ones((1, n_arms, 1))
        exploit = np.matmul(theta_hat.reshape(n_sims, n_arms, 1, d), tile_context.reshape(n_sims, n_arms, d, 1))
        explore = alpha * np.sqrt(
            np.matmul(tile_context.reshape((n_sims, n_arms, 1, d)),
                np.matmul(np.linalg.inv(A), tile_context.reshape((n_sims, n_arms, d, 1)))
            )
        )
        p = (exploit + explore).reshape((n_sims, n_arms))

        chosen_arms = np.argmax(p, axis=1)
        chosen_means = np.matmul(context.reshape(n_sims, 1, d), theta[chosen_arms].reshape(n_sims, d, 1)).reshape(n_sims)
        rewards = np.random.normal(loc=chosen_means, scale=SAMPLING_STD)

        A[np.arange(n_sims), chosen_arms] += context[:, :, None] * context[:, None, :]
        b[np.arange(n_sims), chosen_arms] += context * rewards[:, None]
        armCounts[np.arange(n_sims), chosen_arms] += 1

        '''Compute BvM TV distance'''

        Emp_Fisher = (A - np.eye(d)[None, None, :, :] * np.ones((n_sims, n_arms, 1, 1))) / SAMPLING_STD**2

        posteriorVar = np.linalg.inv(Emp_Fisher + np.linalg.inv(priorVar)[None, :, :, :])
        posteriorMean = np.matmul(posteriorVar, (b/SAMPLING_STD**2 + np.matmul(
            np.linalg.inv(priorVar), priorMean.reshape(n_arms, d, 1)
        ).reshape(n_arms, d)[None, :, :]).reshape(n_sims, n_arms, d, 1)).reshape(n_sims, n_arms, d)

        bvmVar = np.linalg.inv(Emp_Fisher + np.eye(d)[None, None, :, :]*np.ones((n_sims, n_arms, 1, 1))*1e-10)
        bvmMean = np.matmul(bvmVar / SAMPLING_STD**2, b.reshape(n_sims, n_arms, d, 1)).reshape(n_sims, n_arms, d)


        # Rescale Gaussians so that bvm distribution is standard normal

        L = np.linalg.cholesky(bvmVar)
        diffMean = np.matmul(np.linalg.inv(L), (posteriorMean - bvmMean).reshape(n_sims, n_arms, d, 1)).reshape(n_sims, n_arms, d)
        diffVar = np.matmul(np.linalg.inv(L), np.matmul(posteriorVar, np.transpose(np.linalg.inv(L), axes=[0, 1, 3, 2])))

        standardSamples = np.random.normal(size=(n_sims, n_arms, d))
        standardPDF = np.prod(1/np.sqrt(2*np.pi) * np.exp(-0.5 * np.power(standardSamples, 2)), axis=(1, 2))

        offset = standardSamples - diffMean
        det = np.power(np.prod(np.diagonal(np.linalg.cholesky(diffVar), axis1=2, axis2=3), axis=2), 2)
        diffPDF = np.prod(1/np.sqrt(np.power(2*np.pi, d)*det) * np.exp(-0.5 * np.matmul(offset.reshape(n_sims, n_arms, 1, d),
                    np.matmul(np.linalg.inv(diffVar), offset.reshape(n_sims, n_arms, d, 1)))).reshape(n_sims, n_arms), axis=1)

        TVsamples = 1 - diffPDF/standardPDF
        TVsamples[TVsamples < 0] = 0
        TV_est[t] = TVsamples.mean()
        TV_se[t] = TVsamples.std() / np.sqrt(n_sims)

        '''Get min arm pulls'''

        minPulls[t] = armCounts.min(axis=1).mean()
        tmp = np.power(posteriorMean - bvmMean, 2)
        tmp[tmp > 1] = 1
        diff_mean_est[t] = np.sum(tmp)
        tmp = np.power(posteriorVar - bvmVar, 2)
        tmp[tmp > 1] = 1
        diff_var_est[t] = np.sum(tmp)
    return TV_est, TV_se, minPulls, diff_mean_est / (n_sims * n_arms), diff_var_est / (n_sims * n_arms)

if __name__ == '__main__':

    start_time = time.time()
    print("Running contextual bandit...")

    tv = np.zeros((3, n_rounds))
    tv_se = np.zeros((3, n_rounds))
    minPulls = np.zeros((3, n_rounds))
    diff_mean = np.zeros((3, n_rounds))
    diff_var = np.zeros((3, n_rounds))

    for i, theta in enumerate(allTheta):
        tv[i, :], tv_se[i, :], minPulls[i, :], diff_mean[i, :], diff_var[i, :] = runContextual(n_sims, n_arms, d, n_rounds, alpha, theta)

    print("Finished running contextual bandit, Time: " + str(time.time() - start_time))

    np.savetxt('data/contextual_tv.txt', tv)
    np.savetxt('data/contextual_tvse.txt', tv_se)
