from scipy.spatial.distance import cdist
import numpy as np 
from numba import njit, prange

## Choosing a kernel

# def squared_exp_kernel(X1, X2, l=1.0, sigma_f=1.0):
#     '''
#     Isotropic squared exponential kernel. Computes 
#     a covariance matrix from points in X1 and X2.
    
#     Args:
#     X1: Array of m points (m x d).
#     X2: Array of n points (n x d).
    
#     Returns:
#     Covariance matrix (m x n).
#     '''

#     sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
#     return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


@njit(parallel=True)
def squared_exp_kernel(X1, X2, l=1.0, sigma_f=1.0):
    m, d = X1.shape
    n = X2.shape[0]
    K = np.empty((m, n))
    for i in prange(m):
        for j in range(n):
            sqdist = 0.0
            for k in range(d):
                diff = X1[i, k] - X2[j, k]
                sqdist += diff * diff
            K[i, j] = sigma_f ** 2 * np.exp(-0.5 * sqdist / (l ** 2))
    return K
    

@njit(parallel=True)
def gaussian_kernel(X1, X2, sigma=0.1):
    L, d = X1.shape
    N = X2.shape[0]
    K = np.empty((L, N))
    for i in prange(L):
        for j in range(N):
            sq_dist = 0.0
            for k in range(d):
                diff = X1[i, k] - X2[j, k]
                sq_dist += diff * diff
            K[i, j] = np.exp(-sq_dist / (2 * sigma * sigma))
    return K

# def gaussian_kernel(X1, X2, sigma=0.1):
#     """
#     Compute the Gaussian kernel matrix K(X1, X2).

#     Parameters:
#     X1: np.ndarray of shape (L, d)  # L points in d dimensions
#     X2: np.ndarray of shape (N, d)  # N points in d dimensions
#     sigma: float, kernel bandwidth

#     Returns:
#     K: np.ndarray of shape (L, N)  # Kernel matrix
#     """
#     pairwise_sq_dists = cdist(X1, X2, metric='sqeuclidean')  # (L, N) matrix
#     K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
#     return K