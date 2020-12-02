import numpy as np


def pca(X):
    """
    pca Run principal component analysis on the dataset X
        pca(X) computes eigenvectors of the covariance matrix of X
        Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.

    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).

    # ============================================================
    return U, S
