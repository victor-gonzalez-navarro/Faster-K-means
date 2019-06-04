import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


# To plot each iteration of the K-means algorithm
def ploting(data,N_clusters,pred_labels,iterations,fig,centroids,res,max_iterations):
    color=iter(cm.rainbow(np.linspace(0,1,N_clusters)))
    ax = fig.add_subplot(1,max_iterations,iterations)
    for i in range(0,N_clusters):
        whe = np.argwhere(np.array(pred_labels) == i)
        c=next(color)
        ax.plot(data[whe[:,0],0],data[whe[:,0],1],'*',c=c)
        ax.plot(centroids[i,0], centroids[i,1], 'k*')
        ax.grid(b=True, which='major', axis='both', c='0.7', ls='-', linewidth=1)
        plt.xticks(np.arange(0, 1+res[1], res[1]))
        plt.yticks(np.arange(0, 1+res[1], res[1]))


# To plot the multiple partitions
def ploting2(data, dictionary, res, isdictionary, fig, iterations, max_iterations):
    color = iter(cm.rainbow(np.linspace(0, 1, 2)))
    ax = fig.add_subplot(1, max_iterations, iterations)
    for i in range(0, 2):
        c = next(color)
        if i == 0:
            ax.plot(data[:, 0], data[:, 1], '*', c=c)
        else:
            if isdictionary:
                for key, values in dictionary.items():
                    ax.plot(values[0][0], values[0][1], '*', c=c)
            else:
                for val in range(0, dictionary.shape[0]):
                    ax.plot(dictionary[val, :][0], dictionary[val, :][1], '*', c=c)
    ax.grid(b=True, which='major', axis='both', c='0.2', ls='-', linewidth=2)
    plt.xticks(np.arange(0, 1 + res[1], res[1]))
    plt.yticks(np.arange(0, 1 + res[1], res[1]))


# Wheighted K-means algorithm
def kmeans_algorithm(data, K, max_iterations, weights, centroids, res, doplots, epsilon):
    d = data.shape[1]  # number of features
    N = data.shape[0]  # number of instances
    distances = np.zeros((N, K))
    if doplots:
        fig = plt.figure(figsize=(33, 5.5))

    # --INIT ASSIGNMENT--
    new_centroids = np.zeros((K, d))
    for i in range(0, K):
        distances[:, i] = np.sum((data - centroids[i, :]) ** 2, axis=1)
    pred_labels = np.argmin(distances, axis=1)
    if doplots:
        ploting(data, K, pred_labels, 1, fig, centroids, res, max_iterations + 1)
    SSEnew = np.sum(np.min(distances, axis=1))
    SSElast = SSEnew + 10

    iterations = 0
    while (abs(SSElast-SSEnew) > epsilon) and (iterations < max_iterations):
        SSElast = SSEnew
        # --UPDATE--
        for i in range(0, K):
            indic = np.argwhere(pred_labels == i).reshape(np.argwhere(pred_labels == i).shape[0], )
            wheihtsres = weights[indic]
            info_data = data[indic, :] * wheihtsres.reshape(len(wheihtsres), 1)
            info_weights = weights[indic]
            new_centroids[i, :] = np.sum(info_data, axis=0)
            centroids[i, :] = new_centroids[i, :] / np.sum(info_weights)

        # --ASSIGNMENT--
        new_centroids = np.zeros((K, d))
        for i in range(0, K):
            distances[:, i] = np.sum((data - centroids[i, :]) ** 2, axis=1)
        SSEnew = np.sum(np.min(distances, axis=1))
        pred_labels = np.argmin(distances, axis=1)
        if doplots:
            ploting(data, K, pred_labels, iterations + 2, fig, centroids, res, max_iterations + 1)
        iterations = iterations + 1

    return centroids, pred_labels