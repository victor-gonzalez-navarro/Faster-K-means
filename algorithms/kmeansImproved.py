from algorithms.auxiliaryFunctions import *
import matplotlib.pyplot as plt


class KMeansImproved():
    centroids = None
    labels = None

    def __init__(self, n_clusters):
        self.K = n_clusters

    def fit(self, data, m, max_iterations, epsilon, doplots):
        d = data.shape[1]  # number of features
        N = data.shape[0]  # number of isntances
        K = self.K  # number of clusters

        if doplots:
            fig2 = plt.figure(figsize=(35, 6))

        # ***** STEP 1: Compute the set of weights and representatives of the sequence of thinner partitions backwards
        resolution = [3]
        if m > 1:
            for it in range(m - 1):
                resolution.append(resolution[it] + (2 ** (it + 1)))
        group = (resolution[m - 1] - 1) * np.ones((N, d))
        dictionary = dict()
        saved_dictionaries = [0] * m
        for midx in range(0, m):
            res = np.linspace(0, 1, resolution[m - 1 - midx])
            numgrups_axis = len(res) - 1
            numgrups = numgrups_axis ** d
            if midx == 0:
                for i in range(0, N):
                    key_num = ()
                    for j in range(0, d):
                        groupnum = int(min(np.floor(data[i, j] * (numgrups_axis)) + 1, numgrups_axis))
                        group[i, j] = groupnum
                        key_num = key_num + (groupnum,)

                    if key_num in dictionary:
                        dictionary[key_num] = (dictionary[key_num][0] + data[i, 0:d], dictionary[key_num][1] + 1)
                    else:
                        dictionary[key_num] = (data[i, 0:d], 1)
            dictionary_new = dict()
            for key, value in dictionary.items():
                norm_vector = value[0]
                norm_vector = norm_vector / value[1]
                dictionary[key] = (norm_vector, value[1])
                reduced_tup = tuple(np.ceil(ti / 2) for ti in key)
                if reduced_tup in dictionary_new:
                    dictionary_new[reduced_tup] = (
                        dictionary_new[reduced_tup][0] + norm_vector * value[1],
                        dictionary_new[reduced_tup][1] + value[1])
                else:
                    dictionary_new[reduced_tup] = (norm_vector * value[1], value[1])
            if doplots:
                ploting2(data, dictionary, res, True, fig2, midx + 1, m)
            saved_dictionaries[midx] = dictionary
            dictionary = dictionary_new

        # ***** STEP 2: Update the centroid’s set approximation
        i = 1
        # Initialze Centroids in smaller grid
        centroids = []
        for l in range(0, K):
            centroids = centroids + [list(list(saved_dictionaries[m - 1].values())[0:K][l][0])]
        centroids = np.array(centroids)
        for midx in range(0, m):
            representative = []
            weights = []
            res = np.linspace(0, 1, resolution[midx])
            for key, value in saved_dictionaries[-(midx + 1)].items():
                weights = weights + [value[1]]
                representative = representative + [list(value[0])]
            centroids, pred_labels = kmeans_algorithm(np.array(representative), K, max_iterations, np.array(weights),
                                                      centroids, res, doplots, epsilon)

        # ***** Compute Final Labels of the data points
        final_labels = []  # label of all the datapoints
        for i in range(0, N):
            close_represent = saved_dictionaries[0][tuple(group[i, :])]
            row_rep = np.where((np.array(representative) == tuple(close_represent[0])).all(axis=1))[0][0]
            final_labels = final_labels + [pred_labels[row_rep]]
        if doplots:
            fig3 = plt.figure(figsize=(6, 6))
            ploting(data[:, 0:d], K, final_labels, 1, fig3, centroids, np.linspace(0, 1, resolution[midx]), 1)

        self.centroids = centroids
        self.labels = final_labels

