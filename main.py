import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from readDataset import read_dataset


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
def kmeans_algorithm(data, K, max_iterations, weights, centroids, res):
    d = data.shape[1]  # number of features
    N = data.shape[0]  # number of instances
    distances = np.zeros((N, K))
    fig = plt.figure(figsize=(33, 5.5))

    # --INIT ASSIGNMENT--
    new_centroids = np.zeros((K, d))
    for i in range(0, K):
        distances[:, i] = np.sum((data - centroids[i, :]) ** 2, axis=1)
    pred_labels = np.argmin(distances, axis=1)
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
        ploting(data, K, pred_labels, iterations + 2, fig, centroids, res, max_iterations + 1)
        iterations = iterations + 1

    return centroids, pred_labels


# ------------------------------------------------------------------------------------------------------ HYPERPARAMETERS
m = 5  # number of different partitions
K = 3  # number of clusters
epsilon = 0.001  # Parameter for K-means: Stopping criterion 1
max_iterations = 4  # Parameter for K-means: Stopping criterion 2

# -------------------------------------------------------------------------------------------------------- PREPROCESSING
# classlab = int(input('Does your input data has labels? (1-yes, 0-no). Introduce the number: '))
#
# # Database 1
# databse1 = 'grid.arff' #'grid.arff' 'vehicle.arff'
# dat1 = arff.loadarff('datasets/'+databse1)
# df1 = pd.DataFrame(dat1[0])  # pandas
# data1_notnorm = df1.values  # numpy array
# data1 = np.copy(data1_notnorm)
#
# # Count last columns as a feature or as a label [0-Feature, 1-Label]
# if classlab:
#     cl = 1
# else:
#     cl = 0
#
# pre_pro = prp.LabelEncoder()  # Label Encoding
# pre_pro2 = prp.OneHotEncoder(sparse=False)  # One hot encoding
#
# # Preprocessing: Features may have different ranges
# scaler = prp.MinMaxScaler()
# variable = False
# features_to_delete = []
# N_features = data1.shape[1] - cl
# for i in range(0, N_features):
#     if not isinstance(data1[0][i], np.float):
#         print('Feature ' + str(i) + ' has been encoded as one hot')
#         data1[:, i] = pre_pro.fit_transform(data1[:, i])
#         data = pre_pro2.fit_transform(data1[:, i].reshape(len(data1[:, i]), 1))
#         data1 = np.concatenate((data1, data), axis=1)
#         features_to_delete.append(i)
#         variable = True
#
#     else:
#         print('Feature ' + str(i) + ' has NOT been encoded as one hot')
#         data = data1[:, i].reshape(len(data1[:, i]), 1)
#         data = np.float64(data)
#         inds = np.where(np.isnan(data))
#         mean_data = np.nanmean(data, axis=0)
#         data[inds] = np.take(mean_data, inds[1])
#         res = scaler.fit_transform(data)
#         data1[:, i] = res.reshape(len(res), )
#
# if variable == True:
#     data1 = np.concatenate((data1, data1[:, N_features].reshape(len(data1[:, i]), 1)), axis=1)
#     data1 = np.delete(data1, features_to_delete, axis=1)
#     data1 = np.delete(data1, 0, axis=1)
#
# data = data1
data, classlab = read_dataset()

# ------------------------------------------------------------------------------------------------------- RPKM algorithm
# Algorithm 2: RPKM algorithm
if classlab == 1:
    d = data[:, :-1].shape[1]  # number of features
else:
    d = data[:, :].shape[1]  # number of features

N = data.shape[0]  # number of isntances
fig2 = plt.figure(figsize=(35, 6))


# Step 1 Compute the set of weights and representatives of the sequence of thinner partitions backwards
resolution = [3, 5, 9, 17, 33, 65]
group = (resolution[m - 1] - 1) * np.ones((N, d))
dictionary = dict()
saved_dictionaries = [0] * m
for midx in range(0, m):
    res = np.linspace(0, 1, resolution[m - 1 - midx])
    # print(res)
    numgrups_axis = len(res) - 1
    numgrups = numgrups_axis ** d
    # print('Number of groups in one axis '+ str(numgrups_axis))
    # print('Number of groups in hypercube '+str(numgrups))
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

    # print(group)
    # group = np.ceil(np.true_divide(group, 2))
    dictionary_new = dict()

    for key, value in dictionary.items():
        norm_vector = value[0]
        norm_vector = norm_vector / value[1]
        dictionary[key] = (norm_vector, value[1])

        reduced_tup = tuple(np.ceil(ti / 2) for ti in key)

        if reduced_tup in dictionary_new:
            dictionary_new[reduced_tup] = (
            dictionary_new[reduced_tup][0] + norm_vector * value[1], dictionary_new[reduced_tup][1] + value[1])
        else:
            dictionary_new[reduced_tup] = (norm_vector * value[1], value[1])

    ploting2(data, dictionary, res, True, fig2, midx + 1, m)
    saved_dictionaries[midx] = dictionary
    dictionary = dictionary_new


# Step 2 Update the centroid’s set approximation
i = 1
# Initialze Centroids in smaller grid
centroids = []
for l in range(0,K):
    centroids = centroids + [list(list(saved_dictionaries[m-1].values())[0:K][l][0])]

centroids = np.array(centroids)

for midx in range(0,m):
    representative = []
    weights = []
    res = np.linspace(0, 1, resolution[midx])
    for key,value in saved_dictionaries[-(midx+1)].items():
        weights = weights + [value[1]]
        representative = representative + [list(value[0])]

    centroids, pred_labels = kmeans_algorithm(np.array(representative),K,max_iterations,np.array(weights),centroids,res)

# --------------------------------------------------------------------------------------------------------- FINAL LABELS
final_labels = []  # label of all the datapoints
for i in range(0,N):
    # a) Group is a variable that indicates the square for each dimension of each of the N datapoints
    # b) Saved_dictionaries[0] stores the square for each dimension of each representative (key) in the thinner
    # partition as well as its data location (value)
    # c) Close_represent indicates the coordinates of the closest representative for each data point
    close_represent = saved_dictionaries[0][tuple(group[i,:])]
    # Representative indicate the coordinates of all the representatives
    row_rep = np.where((np.array(representative) == tuple(close_represent[0])).all(axis=1))[0][0]
    final_labels = final_labels + [pred_labels[row_rep]]

fig3 = plt.figure(figsize=(6, 6))
ploting(data[:,0:d],K,final_labels,1,fig3,centroids,np.linspace(0, 1, resolution[midx]),1)

print('Process finished succesfully')
# ------------------------------------------------------------------------------------------------------------- PLOTTING
plt.show()
