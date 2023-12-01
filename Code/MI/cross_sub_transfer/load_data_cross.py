import os
import numpy as np
import yaml
from meya.fileAction import selectData
from meya.loadData_YML import getData_chanSeq_OpenBMI
from sklearn.utils import shuffle
import tensorflow.keras.backend as K

def get_all_data(yml,chanDic,sub):
    datapath = yml['Meta']['initDataFolder']
    tra_data = []
    tra_label = []
    data = {}
    train_data_1 = np.load(os.path.join(datapath, 'X_train_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    train_label_1 = np.load(os.path.join(datapath, 'y_train_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    train_data_2 = np.load(os.path.join(datapath, 'X_val_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    train_label_2 = np.load(os.path.join(datapath, 'y_val_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    train_data_1, train_label_1, train_data_2, train_label_2 = data_process(yml,chanDic, train_data_1, train_label_1,
                                                                         train_data_2, train_label_2)
    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_1 = train_label_1-1
        train_label_2 = train_label_2-1

    train_data_1, train_data_2 = trans(train_data_1, train_data_2)
    train_data_1 = dataset_1Dto2D(train_data_1)
    train_data_2 = dataset_1Dto2D(train_data_2)
    tra_data.append(train_data_1)
    tra_data.append(train_data_2)
    tra_label.append(train_label_1)
    tra_label.append(train_label_2)
    del train_data_1
    del train_data_2

    #session2数据处理
    train_data_3 = np.load(os.path.join(datapath, 'X_train_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    train_label_3 = np.load(os.path.join(datapath, 'y_train_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    train_data_4 = np.load(os.path.join(datapath, 'X_val_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    train_label_4 = np.load(os.path.join(datapath, 'y_val_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)


    train_data_3, train_label_3, train_data_4, train_label_4 = data_process(yml,chanDic, train_data_3, train_label_3,
                                                                          train_data_4, train_label_4)

    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_3 = train_label_3 - 1
        train_label_4 = train_label_4 - 1

    train_data_3, train_data_4 = trans(train_data_3, train_data_4)
    train_data_3 = dataset_1Dto2D(train_data_3)
    train_data_4 = dataset_1Dto2D(train_data_4)
    tra_data.append(train_data_3)
    tra_data.append(train_data_4)
    tra_label.append(train_label_3)
    tra_label.append(train_label_4)
    del train_data_3
    del train_data_4

    train_data = tra_data[0]
    train_label = tra_label[0]
    for i in range(1, len(tra_data)):
        train_data = np.concatenate((train_data, tra_data[i]), axis=0)
        train_label = np.concatenate((train_label, tra_label[i]), axis=0)
    # val_len = int(train_data.shape[0]*0.8)
    # val_data = train_data[val_len:]
    # val_label = tra_label[val_len:]
    # train_data = train_data[:val_len]
    # train_label = train_label[:val_len]
    data['tra'] = train_data
    # data['val'] = val_data
    data['tra_l'] = train_label
    # data['val_l'] = val_label

    # return tra_data, te_data, test_data_dict, tra_data_dict, te_dataset
    return data


def data_process(yml,chanDic, train_data, train_label, test_data, test_label):
    eventTypeDic = {
        1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    train_select_event, test_select_event = [], []
    # train  events select
    train_right_Index = np.where(train_label[:, 2] == 1)[0]
    train_left_Index = np.where(train_label[:, 2] == 2)[0]
    train_event_Index = np.sort(np.concatenate((train_left_Index, train_right_Index)))
    #
    for i in train_event_Index:
        train_select_event.append(train_label[i])
    train_select_event = np.array(train_select_event, dtype='int32')

    # train数据切割
    train_x, train_y = getData_chanSeq_OpenBMI(yml, train_data, train_select_event, chanDic,
                                               eventTypeDic=eventTypeDic, getAll=False)

    train_x,train_y = shuffle(train_x,train_y)
    #
    test_right_Index = np.where(test_label[:, 2] == 1)[0]
    test_left_Index = np.where(test_label[:, 2] == 2)[0]
    test_event_Index = np.sort(np.concatenate((test_left_Index, test_right_Index)))

    for i in test_event_Index:
        test_select_event.append(test_label[i])
    test_select_event = np.array(test_select_event, dtype='int32')

    # test数据切割
    test_x, test_y = getData_chanSeq_OpenBMI(yml, test_data, test_select_event, chanDic,
                                             eventTypeDic=eventTypeDic, getAll=False)

    test_x,test_y = shuffle(test_x,test_y)

    return train_x, train_y, test_x, test_y



def data_1Dto2D(data,x=4,y=9):
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


def Segmentation_by_label(data, label):
    new_data = {}
    label_0_index = []
    label_1_index = []
    for i in range(len(label)):
        #右手标签
        if all(label[i] == np.array([1., 0.])):
            label_0_index.append(i)
        #左手标签
        if all(label[i] == np.array([0., 1.])):
            label_1_index.append(i)
    #右手/左手样本数量，时间维度，空间维度1，空间维度2
    data_0 = np.zeros((len(label_0_index), data.shape[1], data.shape[2], data.shape[3]))
    data_1 = np.zeros((len(label_1_index), data.shape[1], data.shape[2], data.shape[3]))

    idx_0=0
    for idx in label_0_index:
        data_0[idx_0] = data[idx]
        idx_0 += 1

    idx_1=0
    for idx in label_1_index:
        data_1[idx_1] = data[idx]
        idx_1 += 1

    new_data['0'] = data_0
    new_data['1'] = data_1
    return new_data
