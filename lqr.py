
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy import stats
import time

T=1000
numSim = 1000

# Smaller scale:
# T=100
# numSim = 100

Cx=100
CK=100
tau2=1
sigma2=1
beta=0.5
alpha=0

config0 = {
    'x0': np.array([0, 0]),
    'A_true': np.array([
        [1, 0],
        [0, 1]
    ]),
    'B_true': np.array([
        [1, 0],
        [0, 1]
    ]),
    'Q': np.array([
        [1, 0],
        [0, 1]
    ]),
    'R': np.array([
        [0.5, 0],
        [0, 0.5]
    ]),
    'K0': np.array([
        [0, 0],
        [0, 0]
    ])
}

config1 = {
    'x0': np.array([0, 0]),
    'A_true': np.array([
        [0, -1],
        [1, 0]
    ]),
    'B_true': np.array([
        [1],
        [0]
    ]),
    'Q': np.array([
        [1, 0],
        [0, 1]
    ]),
    'R': np.array([
        [0.5]
    ]),
    'K0': np.array([
        [0, 0]
    ])
}

config2 = {
    'x0': np.array([0, 0]),
    'A_true': np.array([
        [1, 0],
        [0, 1]
    ]),
    'B_true': np.array([
        [1],
        [0]
    ]),
    'Q': np.array([
        [1, 0],
        [0, 1]
    ]),
    'R': np.array([
        [0.5]
    ]),
    'K0': np.array([
        [0, 0]
    ])
}

configs = [config0, config1, config2]

def lqr_gain(A, B, Q, R):
    """Compute the LQR gain matrix K using the discrete algebraic Riccati equation."""
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

def least_squares_estimation(X, U, X_next):
    """Perform least squares estimation to find A_hat and B_hat."""
    Z = np.vstack((X, U))
    Theta_hat = X_next @ np.linalg.pinv(Z)
    n = X.shape[0]
    A_hat = Theta_hat[:, :n]
    B_hat = Theta_hat[:, n:]
    return A_hat, B_hat

def stepwise_noisy_cec(x0, A_true, B_true, Q, R, 
                        K0, Cx, CK, tau2, sigma2, beta, alpha, T):
    """Algorithm 1: Stepwise Noisy Certainty Equivalent Control."""
    n, d = B_true.shape
    x = x0.reshape(n, 1)
    
    # History lists
    X = [x]
    U = []
    X_next = []
    K_hist = []

    priorMean = np.zeros((n, n+d))
    priorVar = np.eye(n+d)

    TV_dist_est = np.zeros(T)
    TV_dist_est[0] = TV_dist_est[1] = 1

     # Initial actions with stabilizing controller and noise
    for _ in range(2):
        w = np.random.randn(d, 1)
        u = K0 @ x + np.sqrt(tau2) * w
        u = u.reshape(d, 1)
        x_next = A_true @ x + B_true @ u + np.random.normal(loc=0, scale=sigma2**0.5, size=(n, 1))
        
        U.append(u)
        X_next.append(x_next)
        X.append(x_next)
        x = x_next

    for t in range(2, T):
        X_mat = np.hstack(X[:-1])
        U_mat = np.hstack(U)
        X_next_mat = np.hstack(X_next)

        A_hat, B_hat = least_squares_estimation(X_mat, U_mat, X_next_mat)

        # Check if stabilizable (simple spectral radius check)
        try:
            K_hat = lqr_gain(A_hat, B_hat, Q, R)
        except:
            K_hat = K0

        # Enforce safeguard if state or controller norm is too large
        if np.linalg.norm(x) > Cx * np.log(t) or np.linalg.norm(K_hat) > CK:
            K_hat = K0
        
        K_hist.append(K_hat)

        # Generate control input with vanishing exploration noise
        w = np.random.randn(d, 1)
        eta = np.sqrt(tau2 * t**(-(1 - beta)) * np.log(t)**alpha) * w
        u = K_hat @ x + eta
        
        u = u.reshape(d, 1)

        # System update
        x_next = A_true @ x + B_true @ u + np.random.normal(loc=0, scale=sigma2**0.5, size=(n, 1))

        U.append(u)
        X_next.append(x_next)
        X.append(x_next)
        x = x_next

        # Calculate BvM, where parameter distributions are calculated separately for each row of Theta.

        Z = np.vstack((X_mat, U_mat))
        BvMmean = X_next_mat @ np.linalg.pinv(Z)
        BvMvar = sigma2 * np.linalg.inv(Z @ Z.T + np.eye(n+d)*1e-03)

        posteriorVar = np.linalg.inv(Z @ Z.T / sigma2 + priorVar + np.eye(n+d)*1e-03)
        posteriorMean = (X_next_mat @ Z.T / sigma2 + priorMean @ np.linalg.inv(priorVar)) @ posteriorVar.T

        # Draw samples from BvM distribution row by row and calculate their pdf

        BvMpdf = np.zeros(n)
        posteriorpdf = np.zeros(n)
        for i in range(n):
            sample = np.random.multivariate_normal(mean=BvMmean[i, :], cov=BvMvar)
            BvMpdf[i] = stats.multivariate_normal.pdf(sample, mean=BvMmean[i, :], cov=BvMvar)
            posteriorpdf[i] = stats.multivariate_normal.pdf(sample, mean=posteriorMean[i, :], cov=posteriorVar)
        TV_dist_est[t] = max(0, 1 - posteriorpdf.prod() / BvMpdf.prod())

    return TV_dist_est

if __name__ == '__main__':

    start_time = time.time()
    print("Running LQR...")

    for config in configs:
        try:
            print(lqr_gain(config['A_true'], config['B_true'], config['Q'], config['R']))
        except:
            print("Not stabilizable")

    for configID, config in enumerate(configs):

        agg = np.zeros((numSim, T))
        for i in range(numSim):
            with open("data/progress.txt", 'w') as f:
                f.write(f"{i} of {numSim}")
            agg[i, :] = stepwise_noisy_cec(config['x0'], config['A_true'], config['B_true'], config['Q'], config['R'], config['K0'],
                                            Cx, CK, tau2, sigma2, beta, alpha, T)

        mean_TV = agg.mean(axis=0)
        tv_se = agg.std(axis=0) / np.sqrt(numSim)
        print("Standard Error Ratio: " + str(np.nan_to_num(tv_se / mean_TV).max()))
        np.savetxt(f'data/lqr_tv{configID}.txt', mean_TV)

    print("Finished running LQR, Time: " + str(time.time() - start_time))
