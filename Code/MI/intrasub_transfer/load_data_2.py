import os
import numpy as np
import yaml
from meya.fileAction import selectData
from meya.loadData_YML import getData_chanSeq_OpenBMI
from sklearn.utils import shuffle
import tensorflow.keras.backend as K

def get_all_data(yml,chanDic,sub):
    datapath = yml['Meta']['initDataFolder']
    tra_data_dict = {}
    test_data_dict = {}
    te_dataset = {}
    train_data_1 = np.load(os.path.join(datapath, 'X_train_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    train_label_1 = np.load(os.path.join(datapath, 'y_train_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    test_data_1_fine = np.load(os.path.join(datapath, 'X_test_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    test_label_1_fine = np.load(os.path.join(datapath, 'y_test_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)

    train_data_1, train_label_1, test_data_1_fine, test_label_1_fine = data_process(yml,chanDic, train_data_1, train_label_1,
                                                                         test_data_1_fine, test_label_1_fine)
    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_1 = train_label_1-1
        test_label_1_fine = test_label_1_fine-1

    train_data_1, test_data_1_fine = trans(train_data_1, test_data_1_fine)
    train_data_1 = dataset_1Dto2D(train_data_1)
    test_data_1_fine = dataset_1Dto2D(test_data_1_fine)

    #session2数据处理
    train_data_2 = np.load(os.path.join(datapath, 'X_train_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    train_label_2 = np.load(os.path.join(datapath, 'y_train_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    test_data_2 = np.load(os.path.join(datapath, 'X_test_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    test_label_2 = np.load(os.path.join(datapath, 'y_test_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)


    train_data_2, train_label_2, test_data_2, test_label_2 = data_process(yml,chanDic, train_data_2, train_label_2,
                                                                          test_data_2, test_label_2)

    #testdata
    # testdatapath = "/home/xumeiyan/Public/Data/MI/OpenBMI/Traindata/dependent/2_class/400Hz_49chan_8_30Hz_0.06"
    # train_data_2_new = np.load(os.path.join(testdatapath, 'X_train_S{:03d}_fold001.npy'.format(sub)), allow_pickle=True)
    # train_label_2_new = np.load(os.path.join(testdatapath, 'y_train_S{:03d}_fold001.npy'.format(sub)), allow_pickle=True)
    # test_data_2_new = np.load(os.path.join(testdatapath, 'X_test_S{:03d}_fold001.npy'.format(sub)), allow_pickle=True)
    # test_label_2_new = np.load(os.path.join(testdatapath, 'y_test_S{:03d}_fold001.npy'.format(sub)), allow_pickle=True)


    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_2 = train_label_2 - 1
        test_label_2 = test_label_2 - 1

    train_data_2, test_data_2 = trans(train_data_2, test_data_2)
    train_data_2 = dataset_1Dto2D(train_data_2)
    test_data_2 = dataset_1Dto2D(test_data_2)


    # train_data = np.concatenate((train_data_1, train_data_2), axis=0)
    # train_label = np.concatenate((train_label_1, train_label_2), axis=0)
    test_data = np.concatenate((test_data_1_fine, test_data_2), axis=0)
    test_label = np.concatenate((test_label_1_fine, test_label_2), axis=0)
    #返回一个字典{“0”:存放右手数据，维度(样本数量，时间点，空间维度1，空间维度2),"1":存放左手数据，维度(样本数量，时间点，空间维度1，空间维度2)}
    # tra_data = Segmentation_by_label(train_data, train_label)
    # te_data = Segmentation_by_label(test_data, test_label)



    test_data_dict['te1'] = test_data_1_fine
    test_data_dict['te2'] = test_data_2
    test_data_dict['te1_label'] = test_label_1_fine
    test_data_dict['te2_label'] = test_label_2
    tra_data_dict['tra1'] = train_data_1
    tra_data_dict['tra2'] = train_data_2
    tra_data_dict['tra1_label'] = train_label_1
    tra_data_dict['tra2_label'] = train_label_2
    te_dataset['x'] = test_data
    te_dataset['y'] = test_label

    # return tra_data, te_data, test_data_dict, tra_data_dict, te_dataset
    return te_dataset, tra_data_dict, test_data_dict

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
