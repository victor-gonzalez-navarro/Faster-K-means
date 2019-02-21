import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn import preprocessing as prp

def read_dataset():

    realdata = int(input('Do you want to use artificial or real data? (0-art,1-real). Introduce the number: '))
    classlab = int(input('Does your input data has labels? (1-yes, 0-no). Introduce the number: '))

    if realdata:
        # Real Database
        databse1 = 'grid.arff'  # 'grid.arff' 'vehicle.arff'
        dat1 = arff.loadarff('datasets/' + databse1)
        df1 = pd.DataFrame(dat1[0])  # pandas
        data1_notnorm = df1.values  # numpy array
        data1 = np.copy(data1_notnorm)
    else:
        # Artificial Database
        mean1 = [0, 0]; mean2 = [4, 4]; mean3 = [-4, 5]; cov1 = [[1, 0], [0, 2]]; cov3 = [[0.5, 0], [0, 0.5]]
        x1, y1 = np.random.multivariate_normal(mean1, cov1, 5000).T
        x2, y2 = np.random.multivariate_normal(mean2, cov1, 5000).T
        x3, y3 = np.random.multivariate_normal(mean3, cov3, 5000).T
        x = np.array(list(x1) + list(x2) + list(x3))
        y = np.array(list(y1) + list(y2) + list(y3))
        data1 = np.zeros((len(x), 2))
        data1[:, 0] = x
        data1[:, 1] = y

    # Count last columns as a feature or as a label [0-Feature, 1-Label]
    if classlab:
        cl = 1
    else:
        cl = 0

    pre_pro = prp.LabelEncoder()  # Label Encoding
    pre_pro2 = prp.OneHotEncoder(sparse=False)  # One hot encoding

    # Preprocessing: Features may have different ranges
    scaler = prp.MinMaxScaler()
    variable = False
    features_to_delete = []
    N_features = data1.shape[1] - cl
    for i in range(0, N_features):
        if not isinstance(data1[0][i], np.float):
            print('Feature ' + str(i) + ' has been encoded as one hot')
            data1[:, i] = pre_pro.fit_transform(data1[:, i])
            data = pre_pro2.fit_transform(data1[:, i].reshape(len(data1[:, i]), 1))
            data1 = np.concatenate((data1, data), axis=1)
            features_to_delete.append(i)
            variable = True

        else:
            print('Feature ' + str(i) + ' has NOT been encoded as one hot')
            data = data1[:, i].reshape(len(data1[:, i]), 1)
            data = np.float64(data)
            inds = np.where(np.isnan(data))
            mean_data = np.nanmean(data, axis=0)
            data[inds] = np.take(mean_data, inds[1])
            res = scaler.fit_transform(data)
            data1[:, i] = res.reshape(len(res), )

    if variable == True:
        data1 = np.concatenate((data1, data1[:, N_features].reshape(len(data1[:, i]), 1)), axis=1)
        data1 = np.delete(data1, features_to_delete, axis=1)
        data1 = np.delete(data1, 0, axis=1)

    return data1, classlab