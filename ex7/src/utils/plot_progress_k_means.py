from matplotlib import pyplot
import matplotlib as mpl
import numpy as np


def plot_progress_k_means(i, X, centroid_history, idx_history):
    """
    plot_progress_k_means is a helper function that displays the progress of
    k-Means as it is running. It is intended for use only with 2D data.
        plot_progress_k_means(X, centroids, previous, idx, K, i) plots the data
        points with colors assigned to each centroid. With the previous
        centroids, it also plots a line between the previous locations and
        current locations of the centroids.
    """

    K = centroid_history[0].shape[0]
    pyplot.gcf().clf()
    cmap = pyplot.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    for k in range(K):
        current = np.stack([c[k, :] for c in centroid_history[:i + 1]], axis=0)
        pyplot.plot(current[:, 0], current[:, 1],
                    '-Xk',
                    mec='k',
                    lw=2,
                    ms=10,
                    mfc=cmap(norm(k)),
                    mew=2)

        pyplot.scatter(X[:, 0], X[:, 1],
                       c=idx_history[i],
                       cmap=cmap,
                       marker='o',
                       s=8 ** 2,
                       linewidths=1, )
    pyplot.grid(False)
    pyplot.title('Iteration number %d' % (i + 1))
