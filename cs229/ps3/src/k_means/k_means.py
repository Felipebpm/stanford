from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    #  centroids_init
    # raise NotImplementedError('init_centroids function not implemented')
    height = image.shape[0]
    width = image.shape[1]
    colors = image.shape[2]
    pixel_sample = image.reshape(height * width, colors)
    index_choices = np.random.choice([i for i in range(pixel_sample.shape[0])], size=num_clusters)
    centroids_init = np.array([pixel_sample[i] for i in index_choices])
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
    # Usually expected to converge long before `max_iter` iterations
    # Initialize `dist` vector to keep track of distance to every centroid
    height = image.shape[0]
    width = image.shape[1]
    colors = image.shape[2]
    num_clusters = centroids.shape[0]
    num_pixels = height * width
    pixel_sample = image.reshape(num_pixels, colors)
    dist = np.zeros((num_clusters, num_pixels))
    new_centroids = np.zeros(centroids.shape)
    # Loop over all centroids and store distances in `dist`
    for index in range(max_iter):
        if index != 0:
            centroids = new_centroids
        if index % print_every == 0:
            print("iteration " + str(index) + " centroids:")
            print(centroids)
        for i in range(num_clusters):
            for j in range(num_pixels):
                dist[i][j] = np.linalg.norm(centroids[i] - pixel_sample[j])
        # Find closest centroid and update `new_centroids`
        ci_count = [0 for i in range(num_clusters)]
        min_values = np.amin(dist, axis=0)
        disT = dist.T
        for i in range(num_pixels):
            cluster = list(disT[i]).index(min_values[i])
            new_centroids[cluster] = (new_centroids[cluster] * ci_count[cluster] + pixel_sample[i]) / (ci_count[cluster] + 1)
            ci_count[cluster] += 1
    # Update `new_centroids`
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    height = image.shape[0]
    width = image.shape[1]
    colors = image.shape[2]
    num_clusters = centroids.shape[0]
    num_pixels = height * width
    pixel_sample = image.reshape(num_pixels, colors)
    # Initialize `dist` vector to keep track of distance to every centroid
    dist = np.zeros((num_clusters, num_pixels))
    # Loop over all centroids and store distances in `dist`
    for i in range(num_clusters):
        for j in range(num_pixels):
            dist[i][j] = np.linalg.norm(centroids[i] - pixel_sample[j])
    # Find closest centroid and update `new_centroids`
    ci_count = [0 for i in range(num_clusters)]
    min_values = np.amin(dist, axis=0)
    disT = dist.T
    # Find closest centroid and update pixel value in `image`
    for i in range(num_pixels):
        cluster = list(disT[i]).index(min_values[i])
        pixel_sample[i] = centroids[cluster]
    image = pixel_sample.reshape(height, width, colors)
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
