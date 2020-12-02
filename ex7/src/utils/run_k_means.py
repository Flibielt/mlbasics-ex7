from matplotlib.animation import FuncAnimation
from matplotlib import pyplot

from .plot_progress_k_means import plot_progress_k_means


def run_k_means(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):
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

    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)

        centroids = computeCentroids(X, idx, K)

    if plot_progress:
        fig = pyplot.figure()
        anim = FuncAnimation(fig, plot_progress_k_means,
                             frames=max_iters,
                             interval=500,
                             repeat_delay=2,
                             fargs=(X, centroid_history, idx_history))
        return centroids, idx, anim

    return centroids, idx
