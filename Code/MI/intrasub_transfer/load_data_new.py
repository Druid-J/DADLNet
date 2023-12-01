import scipy.io as sio
import os
import numpy as np
import yaml
from meya.fileAction import selectData,selectData_BCI2008


def get_data(yml, sub):
    datapath = yml['Meta']['initDataFolder']
    train_data = np.load(os.path.join(datapath, 'X_S{:03d}_T.npy'.format(sub)), allow_pickle=True)
    train_label = np.load(os.path.join(datapath, 'y_S{:03d}_T.npy'.format(sub)), allow_pickle=True)
    test_data = np.load(os.path.join(datapath, 'X_S{:03d}_E.npy'.format(sub)), allow_pickle=True)
    test_label = np.load(os.path.join(datapath, 'y_S{:03d}_E.npy'.format(sub)), allow_pickle=True)
    train_data, train_label = selectData_BCI2008(2, train_data, train_label)
    test_data,test_label = selectData_BCI2008(2,test_data,test_label)
    train_data, test_data = trans(train_data, test_data)
    train_data = dataset_1Dto2D(train_data)
    test_data = dataset_1Dto2D(test_data)

    return train_data, train_label, test_data, test_label


# def data_1Dto2D(data,x=4,y=9):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (   0,     data[0],  data[1],  data[2],     0,     data[3],  data[4],  data[5],     0,   )
#     data_2D[1] = (data[6],  data[7],  data[8],  data[9],  data[10], data[11], data[12], data[13], data[14])
#     data_2D[2] = (data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23])
#     data_2D[3] = (data[24],   0,      data[25], data[26], data[27], data[28], data[29],    0,     data[30])
#     return data_2D

def data_1Dto2D(data,x=4,y=9):
    'BCI2008 2D represent'
    data_2D = np.zeros([x,y])
    data_2D[0] = (   0,     data[0],  data[1],  data[2],     0,     data[3],  data[4],  data[5],     0    )
    data_2D[1] = (data[6],  data[7],  data[8],  data[9],  data[10], data[11], data[12], data[13], data[14])
    data_2D[2] = (data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23])
    data_2D[3] = (data[24],   0,      data[25], data[26], data[27], data[28], data[29],    0,     data[30])
    return data_2D


def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],dataset_1D.shape[1],4,9])
    for i in range(dataset_1D.shape[0]):
        for j in range(dataset_1D.shape[1]):
            dataset_2D[i,j] = data_1Dto2D(dataset_1D[i,j])
    return dataset_2D

def trans(tra,te):
    tra = np.swapaxes(tra, -1, -2)
    te = np.swapaxes(te, -1, -2)
    return tra, te

def data_Comebine(data1,data2):
    data1 = data1.reshape(1, data1.shape[0], data1.shape[1], data1.shape[2])
    data2 = data2.reshape(1, data2.shape[0], data2.shape[1], data2.shape[2])
    data = np.concatenate((data1, data2), axis=0)

    return data



