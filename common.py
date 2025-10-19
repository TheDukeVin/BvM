
import scipy
import numpy as np

# Getting confidence ellipse cutoff

def A(n, c):
    if n == 0:
        return np.sqrt(2*np.pi) * (scipy.stats.norm.cdf(c) - 0.5)
    if n == 1:
        return 1 - np.exp(-c**2 / 2)
    return (n-1) * A(n-2, c) - np.exp(-c**2/2) * c**(n-1)

# n = degrees of freedom of sphere = dim-1

def getEllipseCutoff(alpha, n):
    M = A(n, 1e+10)
    left = 0
    right = 1e+3
    while right - left > 1e-10:
        mid = (left + right) / 2
        if A(n, mid) / M < 1-alpha:
            left = mid
        else:
            right = mid
    return left