def plot_progress_k_means(X, centroids, previous, idx, K, i):
    """
    plot_progress_k_means is a helper function that displays the progress of
    k-Means as it is running. It is intended for use only with 2D data.
        plot_progress_k_means(X, centroids, previous, idx, K, i) plots the data
        points with colors assigned to each centroid. With the previous
        centroids, it also plots a line between the previous locations and
        current locations of the centroids.
    """