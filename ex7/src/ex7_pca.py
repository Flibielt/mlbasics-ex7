from matplotlib import pyplot
from scipy.io import loadmat
import matplotlib as mpl
import numpy as np
import os

from .utils import pca, project_data, recover_data, compute_centroids, find_closest_centroids, k_means_init_centroids, \
    feature_normalize, display_data, run_k_means


def ex7_pca():
    """
    Exercise 7 | Principle Component Analysis and K-Means Clustering

    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

        pca.py
        project_data.py
        recover_data.py
        compute_centroids.py
        find_closest_centroids.py
        k_means_init_centroids.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    """
    ================== Part 1: Load Example Dataset  ===================
    We start this exercise by using a small dataset that is easily to
    visualize
    """
    print('Visualizing example dataset for PCA.\n\n')

    # Load the dataset into the variable X
    data1_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex7data1.mat'
    data1_path = data1_path.replace('\\', '/')
    data = loadmat(data1_path)
    X = data['X']

    #  Visualize the example dataset
    pyplot.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=1)
    pyplot.axis([0.5, 6.5, 2, 8])
    pyplot.gca().set_aspect('equal')
    pyplot.grid(False)

    input('Program paused. Press enter to continue.\n')

    """
    =============== Part 2: Principal Component Analysis ===============
    You should now implement PCA, a dimension reduction technique. You
    should complete the code in pca.py
    """
    print('\nRunning PCA on example dataset.\n\n')

    #  Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = feature_normalize(X)

    #  Run PCA
    U, S = pca(X_norm)

    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    fig, ax = pyplot.subplots()
    ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)

    for i in range(2):
        ax.arrow(mu[0], mu[1], 1.5 * S[i] * U[0, i], 1.5 * S[i] * U[1, i],
                 head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

    ax.axis([0.5, 6.5, 2, 8])
    ax.set_aspect('equal')
    ax.grid(False)

    print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
    print(' (you should expect to see [-0.707107 -0.707107])')

    input('Program paused. Press enter to continue.\n')

    """
    =================== Part 3: Dimension Reduction ===================
    You should now implement the projection step to map the data onto the 
    first k eigenvectors. The code will then plot the data in this reduced 
    dimensional space.  This will show you what the data looks like when 
    using only the corresponding eigenvectors to reconstruct it.
    
    You should complete the code in project_data.py
    """
    print('\nDimension reduction on example dataset.\n\n')

    #  Plot the normalized dataset (returned from pca)
    fig = pyplot.subplots()
    pyplot.plot(X_norm[:, 1], X_norm[:, 2], 'bo')

    #  Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
    print('(this value should be about    : 1.481274)')

    X_rec = recover_data(Z, U, K)
    print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
    print('       (this value should be about  [-1.047419 -1.047419])')

    #  Plot the normalized dataset (returned from featureNormalize)
    fig, ax = pyplot.subplots(figsize=(5, 5))
    ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
    ax.set_aspect('equal')
    ax.grid(False)
    pyplot.axis([-3, 2.75, -3, 2.75])

    # Draw lines connecting the projected points to the original points
    ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
    for xnorm, xrec in zip(X_norm, X_rec):
        ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)

    input('Program paused. Press enter to continue.\n')

    """
    =============== Part 4: Loading and Visualizing Face Data =============
    We start the exercise by first loading and visualizing the dataset.
    The following code will load the dataset into your environment
    """
    print('\nLoading face dataset.\n\n')

    #  Load Face dataset
    faces_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex7faces.mat'
    faces_path = faces_path.replace('\\', '/')
    data = loadmat(faces_path)
    X = data['X']

    #  Display the first 100 faces in the dataset
    display_data(X[:100, :], figsize=(8, 8))

    input('Program paused. Press enter to continue.\n')

    """
    =========== Part 5: PCA on Face Data: Eigenfaces  ===================
    Run PCA and visualize the eigenvectors which are in this case eigenfaces
    We display the first 36 eigenfaces.
    """
    print('\nRunning PCA on face dataset.\n (this might take a minute or two ...)\n\n')

    #  Before running PCA, it is important to first normalize X by subtracting
    #  the mean value from each feature
    X_norm, mu, sigma = feature_normalize(X)

    #  Run PCA
    U, S = pca(X_norm)

    #  Visualize the top 36 eigenvectors found
    display_data(U[:, :36].T, figsize=(8, 8))

    input('Program paused. Press enter to continue.\n')

    """
    ============= Part 6: Dimension Reduction for Faces =================
    Project images to the eigen space using the top k eigenvectors 
    If you are applying a machine learning algorithm 
    """
    print('\nDimension reduction for face dataset.\n\n')

    #  Project images to the eigen space using the top k eigenvectors
    #  If you are applying a machine learning algorithm
    K = 100
    Z = project_data(X_norm, U, K)

    print('The projected data Z has a shape of: ', Z.shape)

    input('\nProgram paused. Press enter to continue.\n')

    """
    ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
    Project images to the eigen space using the top K eigen vectors and 
    visualize only using those K dimensions
    Compare to the original input, which is also displayed
    """
    print('\nVisualizing the projected (reduced dimension) faces.\n\n')

    #  Project images to the eigen space using the top K eigen vectors and
    #  visualize only using those K dimensions
    #  Compare to the original input, which is also displayed
    K = 100
    X_rec = recover_data(Z, U, K)

    # Display normalized data
    display_data(X_norm[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Original faces')

    # Display reconstructed data from only k eigenfaces
    display_data(X_rec[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Recovered faces')
    pass

    input('Program paused. Press enter to continue.\n')

    """
    === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
    One useful application of PCA is to use it to visualize high-dimensional
    data. In the last K-Means exercise you ran K-Means on 3-dimensional 
    pixel colors of an image. We first visualize this output in 3D, and then
    apply PCA to obtain a visualization in 2D.
    """

    # Reload the image from the previous exercise and run K-Means on it
    # For this to work, you need to complete the K-Means assignment first

    bird_small_path = os.path.dirname(os.path.realpath(__file__)) + '/data/bird_small.png'
    bird_small_path = bird_small_path.replace('\\', '/')
    A = mpl.image.imread(bird_small_path)
    A /= 255
    X = A.reshape(-1, 3)

    # perform the K-means clustering again here
    K = 16
    max_iters = 10
    initial_centroids = k_means_init_centroids(X, K)
    centroids, idx = run_k_means(X, initial_centroids,
                                 find_closest_centroids,
                                 compute_centroids, max_iters)

    #  Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    sel = np.random.choice(X.shape[0], size=1000)

    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=8 ** 2)
    ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')
    pass

    """
    === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
    Use PCA to project this cloud to 2D for visualization
    """
    # Subtract the mean to use PCA
    X_norm, mu, sigma = feature_normalize(X)

    # PCA and project the data to 2D
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2)

    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=64)
    ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    ax.grid(False)
    pass

