def run_k_means(X, initial_centroids, max_iters, plot_progress):
    """
    run_k_means runs the K-Means algorithm on data matrix X, where each row of X is a single example
        run_k_means(X, initial_centroids, max_iters, plot_progress)
        runs the K-Means algorithm on data matrix X, where each
        row of X is a single example. It uses initial_centroids used as the
        initial centroids. max_iters specifies the total number of interactions
        of K-Means to execute. plot_progress is a true/false flag that
        indicates if the function should also plot its progress as the
        learning happens. This is set to false by default. runkMeans returns
        centroids, a Kxn matrix of the computed centroids and idx, a m x 1
        vector of centroid assignments (i.e. each entry in range [1..K])
    """