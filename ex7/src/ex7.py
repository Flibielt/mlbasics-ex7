from matplotlib import pyplot
from scipy.io import loadmat
import matplotlib as mpl
import numpy as np
import os

from .utils import find_closest_centroids, compute_centroids, run_k_means, k_means_init_centroids


def ex7():
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
    ================= Part 1: Find Closest Centroids ====================
    To help you implement K-Means, we have divided the learning algorithm 
    into two functions -- find_closest_centroids and compute_centroids. In this
    part, you should complete the code in the find_closest_centroids function. 
    """
    print('Finding closest centroids.\n\n')

    # Load an example dataset that we will be using
    data2_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex7data2.mat'
    data2_path = data2_path.replace('\\', '/')
    data = loadmat(data2_path)
    X = data['X']

    # Select an initial set of centroids
    K = 3  # 3 Centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Find the closest centroids for the examples using the initial_centroids
    idx = find_closest_centroids(X, initial_centroids)

    print('Closest centroids for the first 3 examples:')
    print(idx[:3])
    print('(the closest centroids should be 0, 2, 1 respectively)')

    input('Program paused. Press enter to continue.\n')

    """
    ===================== Part 2: Compute Means =========================
    After implementing the closest centroids function, you should now
    complete the compute_centroids function.
    """
    print('\nComputing centroids means.\n\n')

    # Compute means based on the closest centroids found in the previous part.
    centroids = compute_centroids(X, idx, K)

    print('Centroids computed after initial finding of closest centroids:')
    print(centroids)
    print('\nThe centroids should be')
    print('   [ 2.428301 3.157924 ]')
    print('   [ 5.813503 2.633656 ]')
    print('   [ 7.119387 3.616684 ]')

    input('Program paused. Press enter to continue.\n')

    """
    =================== Part 3: K-Means Clustering ======================
    After you have completed the two functions computeCentroids and
    find_closest_centroids, you have all the necessary pieces to run the
    kMeans algorithm. In this part, you will run the K-Means algorithm on
    the example dataset we have provided. 
    """
    print('\nRunning K-Means clustering on example dataset.\n\n')

    # Load an example dataset
    data = loadmat(data2_path)

    # Settings for running K-Means
    K = 3
    max_iters = 10

    # For consistency, here we set centroids to specific values
    # but in practice you want to generate them automatically, such as by
    # settings them to be random examples (as can be seen in
    # kMeansInitCentroids).
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    # the progress of K-Means
    centroids, idx, anim = run_k_means(X, initial_centroids,
                                       find_closest_centroids, compute_centroids, max_iters, True)
    anim
    pyplot.show()

    print('\nK-Means Done.\n\n')
    input('Program paused. Press enter to continue.\n')

    """
    ============= Part 4: K-Means Clustering on Pixels ===============
    In this exercise, you will use K-Means to compress an image. To do this,
    you will first run K-Means on the colors of the pixels in the image and
    then you will map each pixel onto its closest centroid.
    
    You should now complete the code in k_means_init_centroids.py
    """
    print('\nRunning K-Means clustering on pixels from an image.\n\n')

    # ======= Experiment with these parameters ================
    # You should try different values for those parameters
    K = 16
    max_iters = 10

    # Load an image of a bird
    # Change the file name and path to experiment with your own images
    bird_small_path = os.path.dirname(os.path.realpath(__file__)) + '/data/bird_small.png'
    bird_small_path = bird_small_path.replace('\\', '/')
    A = mpl.image.imread(bird_small_path)
    # ==========================================================

    # Divide by 255 so that all values are in the range 0 - 1
    A /= 255

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = A.reshape(-1, 3)

    # When using K-Means, it is important to randomly initialize centroids
    # You should complete the code in kMeansInitCentroids above before proceeding
    initial_centroids = k_means_init_centroids(X, K)

    # Run K-Means
    centroids, idx = run_k_means(X, initial_centroids,
                                 find_closest_centroids,
                                 compute_centroids,
                                 max_iters)

    """
    ================= Part 5: Image Compression ======================
    In this part of the exercise, you will use the clusters of K-Means to
    compress an image. To do this, we first find the closest clusters for
    each example. After that, we 
    """
    print('\nApplying K-Means to compress an image.\n\n')

    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by its index in idx) to the centroid value
    # Reshape the recovered image into proper dimensions
    X_recovered = centroids[idx, :].reshape(A.shape)

    # Display the original image, rescale back by 255
    fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(A * 255)
    ax[0].set_title('Original')
    ax[0].grid(False)

    # Display compressed image, rescale back by 255
    ax[1].imshow(X_recovered * 255)
    ax[1].set_title('Compressed, with %d colors' % K)
    ax[1].grid(False)

    input('Program paused. Press enter to continue.\n')
