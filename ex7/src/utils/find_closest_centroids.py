import numpy as np


def find_closest_centroids(X, centroids):
    """
    find_closest_centroids computes the centroid memberships for every example
        idx = find_closest_centroids (X, centroids) returns the closest centroids
        in idx for a dataset X where each row is a single example. idx = m x 1
        vector of centroid assignments (i.e. each entry in range [1..K])
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the
    #               range 1..K

    for i in np.arange(idx.size):
        J = np.sqrt(np.sum(np.square(X[i] - centroids), axis=1))

        idx[i] = np.argmin(J)

    # =============================================================
    return idx
