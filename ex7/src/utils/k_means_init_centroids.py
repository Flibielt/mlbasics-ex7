import numpy as np


def k_means_init_centroids(X, K):
    """
    k_means_init_centroids This function initializes K centroids that are to be
    used in K-Means on the dataset X
        centroids = k_means_init_centroids(X, K) returns K initial centroids to be
        used with the K-Means on the dataset X
    """

    m, n = X.shape

    # You should return this values correctly
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X

    # =============================================================
    return centroids
