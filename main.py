import numpy as np

from algorithms.readDataset import read_dataset
from algorithms.kmeansImproved import KMeansImproved
from sklearn.cluster import KMeans  # To compare our implementation with original K-means


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    # HYPERPARAMETERS
    K = 3  # Number of clusters
    epsilon = 0.001  # Stopping criterion 1
    max_iterations = 10  # Stopping criterion 2
    m = 5  # Number of different partitions (an axis is divided in 2^m regions, for m=3 -> 8 regions)
    doplots = False  # Output plots (only useful for 2-dimensional data)

    # Preprocess the data
    data = read_dataset()
    N = data.shape[0]  # number of instances
    d = data.shape[1]  # number of features

    # Call fit and predict of the algorithm (Improved Kmeans) I have implemented
    KmeansIm = KMeansImproved(n_clusters=K)
    KmeansIm.fit(data, m, max_iterations, epsilon, doplots)

    # Call fit and predict of Kmeans
    Kmeans = KMeans(n_clusters=K)
    Kmeans.fit(data[:, 0:d])

    # Compute Sum of Squared Errors (SSE) of Improved Kmeans and original Kmeans
    SSE_mod = 0
    SSE = 0
    for i in range(0, N):
        SSE_mod = SSE_mod + np.sqrt(np.sum((data[i, 0:d] - KmeansIm.centroids[KmeansIm.labels[i], :]) ** 2))
        SSE = SSE + np.sqrt(np.sum((data[i, 0:d] - Kmeans.cluster_centers_[Kmeans.labels_[i], :]) ** 2))
    print('\033[1m' + 'The total SSE of the improved Kmeans is: ' + str(SSE_mod / N) + '\033[0m')
    print('\033[1m' + 'The total SSE of the original Kmeans is: ' + str(SSE / N) + '\033[0m')


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()