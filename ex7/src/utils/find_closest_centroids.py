def find_closest_centroids(X, centroids):
    """
    find_closest_centroids computes the centroid memberships for every example
        idx = find_closest_centroids (X, centroids) returns the closest centroids
        in idx for a dataset X where each row is a single example. idx = m x 1
        vector of centroid assignments (i.e. each entry in range [1..K])
    """