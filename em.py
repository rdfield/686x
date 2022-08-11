"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    mu, var, pi = mixture
    K = mu.shape[0]
    delta = X > 0
    f = np.empty((X.shape[0],K),dtype=np.float64)
    mype = np.empty((X.shape[0],K),dtype=np.float64)
    for i in range(K):
       z=np.sum((((X-mu[i])*delta )**2 )/(var[i]*2),axis=1) 
       z=z[..., None]
       f[:, i] = z[:,0]
       zpe = (var[i] * 2 * np.pi)
       zpe=zpe[..., None]
       mype[:, i] = np.log(zpe)
    pe_1_1 = (-np.sum(delta, axis=1))/2.0
    pe_1_1=pe_1_1[..., None]
    mype = mype*pe_1_1
    f = mype - f
    f = f + np.log(pi + 1e-16)
    logsums = logsumexp(f, axis=1)
    logsums=logsums[..., None]
    log_posts = f - logsums
    return np.exp(log_posts),np.sum(logsums, axis=0).item()


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    old_mu, old_var, old_post = mixture
    delta = X > 0
    K = post.shape[1]
    sum_post = post.sum(axis=0)
    sum_post=sum_post[..., None]
    pi = sum_post / n 
    mu_new = np.dot(post.T, X)
    mu_divisors = post.T @ delta
    mu = np.where(mu_divisors > 1, np.divide(mu_new , mu_divisors),old_mu )
    sigma = np.zeros((n,K));
    for i in range(n):
       Cu = X[i,:] > 0
       diff = X[i, Cu] - mu[:,Cu]
       sigma[i,:] = np.sum(diff**2, axis=1)
    sum_Cu = np.sum(post*np.sum(delta, axis=1).reshape(-1,1), axis=0)
    sigma = np.sum(post*sigma, axis=0)/sum_Cu
    sigma = np.maximum(sigma,min_variance)
    return GaussianMixture(mu, sigma, pi.T[0])



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_lh = None
    new_log_lh = None  # Keep track of log likelihood to check convergence

    # Start the main loop
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):

        old_log_lh = new_log_lh

        # E-step
        post, new_log_lh = estep(X, mixture)

        # M-step
        mixture = mstep(X, post, mixture)

    return mixture, post, new_log_lh


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X.copy()
    mu, _, _ = mixture

    n, d = X.shape
    mu, var, pi = mixture
    print("fill_matrix from mu",mu)
    K = mu.shape[0]
    delta = X > 0
    f = np.empty((X.shape[0],K),dtype=np.float64)
    mype = np.empty((X.shape[0],K),dtype=np.float64)
    for i in range(K):
       z=np.sum((((X-mu[i])*delta )**2 )/(var[i]*2),axis=1)
       z=z[..., None]
       f[:, i] = z[:,0]
       zpe = (var[i] * 2 * np.pi)
       zpe=zpe[..., None]
       mype[:, i] = np.log(zpe)
    pe_1_1 = (-np.sum(delta, axis=1))/2.0
    pe_1_1=pe_1_1[..., None]
    mype = mype*pe_1_1
    f = mype - f
    f = f + np.log(pi + 1e-16)
    logsums = logsumexp(f, axis=1)
    logsums=logsums[..., None]
    log_posts = f - logsums

    post = np.exp(log_posts)

    # Missing entries to be filled
    miss_indices = np.where(X == 0)
    X_pred[miss_indices] = (post @ mu)[miss_indices]

    return X_pred

