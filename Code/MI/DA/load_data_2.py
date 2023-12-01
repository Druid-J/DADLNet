import tensorflow.keras.backend as K
import os
import numpy as np
from meya.loadData_YML import getData_chanSeq_OpenBMI
from sklearn.utils import shuffle
from MNUMI.preprocess.BCI2008.raw import load_filter_data
from sklearn.model_selection import StratifiedKFold
from meya.loadData_YML import getData_chanSeq_BCI2008_Train
import xlsxwriter as xw

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

    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_2 = train_label_2 - 1
        test_label_2 = test_label_2 - 1

    train_data_2, test_data_2 = trans(train_data_2, test_data_2)
    train_data_2 = dataset_1Dto2D(train_data_2)
    test_data_2 = dataset_1Dto2D(test_data_2)

    test_data = np.concatenate((test_data_1_fine, test_data_2), axis=0)
    test_label = np.concatenate((test_label_1_fine, test_label_2), axis=0)


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

    return te_dataset, tra_data_dict, test_data_dict




def get_data_bci_stride(yml, subject, n_folds):
    sessions = ['T', 'E']
    ProDataPath = yml['Meta']['initDataFolder']
    eventTypeDic = {
        0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    all_data = {}
    for id_se, sess in enumerate(sessions):
        split_x, split_y = [], []
        # 加载数据这里的X是拼接的raw_data 和 event
        X, y = load_filter_data(ProDataPath, subject, sess)
        # 1.切割
        left_Index = np.where(y[:, 2] == 0)[0]
        right_Index = np.where(y[:, 2] == 1)[0]
        event_Index = np.sort(np.concatenate((left_Index, right_Index)))
        for i in event_Index:
            split_x.append(y[i])
            split_y.append(y[i][2])
        split_x = np.array(split_x, dtype='int32')
        split_y = np.array(split_y, dtype='int32')
        # 判断是session1数据或者session2数据
        if sess == 'E':
            SKf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
            for fold, (train, val) in enumerate(SKf.split(split_x, split_y)):
                event_train = split_x[train]
                event_val = split_x[val]
                #
                sp_train_x, sp_train_y = getData_chanSeq_BCI2008_Train(yml, X, event_train, eventTypeDic=eventTypeDic,
                                                                       getAll=False)
                sp_val_x, sp_val_y = getData_chanSeq_BCI2008_Train(yml, X, event_val, eventTypeDic=eventTypeDic,
                                                                   getAll=False)

                sp_train_x = np.swapaxes(sp_train_x, -1, -2)
                sp_val_x = np.swapaxes(sp_val_x, -1, -2)
                print('1',sp_train_x.shape)

                sp_train_x = dataset_1Dto2D_bci(sp_train_x)
                sp_val_x = dataset_1Dto2D_bci(sp_val_x)

                X_tr_channel, sp_train_y = shuffle(sp_train_x, sp_train_y)
                X_val_channel, sp_val_y = shuffle(sp_val_x, sp_val_y)
                all_data['fold_{}_tra'.format(fold+1)] = X_tr_channel
                all_data['fold_{}_tra_l'.format(fold+1)] = sp_train_y
                all_data['fold_{}_test'.format(fold + 1)] = X_val_channel
                all_data['fold_{}_test_l'.format(fold + 1)] = sp_val_y
        elif sess == 'T':
            # 1.切割
            sp_test_x, sp_test_y = getData_chanSeq_BCI2008_Train(yml, X, split_x,
                                                                 eventTypeDic=eventTypeDic,
                                                                 getAll=False)
            sp_test_x = np.swapaxes(sp_test_x, -1, -2)
            print('2',sp_test_x.shape) # (576,400 ,20)
            sp_test_x = dataset_1Dto2D_bci(sp_test_x)
            X_test_channel, sp_test_y = shuffle(sp_test_x, sp_test_y)
            all_data['MMD_tra'] = X_test_channel
            all_data['MMD_tra_l'] = sp_test_y
    te_data = np.concatenate((all_data['fold_1_tra'],all_data['fold_1_test']),axis=0)

    return all_data, te_data

def get_all_data_Bci2a_inter(yml,chanDic,sub):
    datapath = yml['Meta']['initDataFolder']
    sub_data = {}


    train_data = np.load(os.path.join(datapath, 'X_S{:03d}_T.npy'.format(sub)), allow_pickle=True)
    train_label = np.load(os.path.join(datapath, 'y_S{:03d}_T.npy'.format(sub)), allow_pickle=True)
    test_data = np.load(os.path.join(datapath, 'X_S{:03d}_E.npy'.format(sub)), allow_pickle=True)
    test_label = np.load(os.path.join(datapath, 'y_S{:03d}_E.npy'.format(sub)), allow_pickle=True)

    train_data, train_label, test_data, test_label = data_process_bci(yml, chanDic, train_data, train_label, test_data, test_label)
    train_data,  test_data = trans(train_data, test_data)
    train_data = dataset_1Dto2D_bci(train_data)
    test_data = dataset_1Dto2D_bci(test_data)
    sub_data['tra'] = np.concatenate((train_data, test_data), axis=0)
    sub_data['tra_l'] = np.concatenate((train_label, test_label), axis=0)

    return sub_data

def get_all_data_OpenBMI_inter(yml,chanDic,sub):
    datapath = yml['Meta']['initDataFolder']
    tra_data_dict = {}
    test_data_dict = {}
    te_dataset = {}
    train_data_1 = np.load(os.path.join(datapath, 'X_train_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    train_label_1 = np.load(os.path.join(datapath, 'y_train_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    test_data_1_fine = np.load(os.path.join(datapath, 'X_test_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)
    test_label_1_fine = np.load(os.path.join(datapath, 'y_test_S{:03d}_sess001.npy'.format(sub)), allow_pickle=True)

    train_data_1, train_label_1, test_data_1_fine, test_label_1_fine = data_process(yml, chanDic, train_data_1,
                                                                                    train_label_1,
                                                                                    test_data_1_fine, test_label_1_fine)
    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_1 = train_label_1 - 1
        test_label_1_fine = test_label_1_fine - 1

    train_data_1, test_data_1_fine = trans(train_data_1, test_data_1_fine)
    train_data_1 = dataset_1Dto2D(train_data_1)
    test_data_1_fine = dataset_1Dto2D(test_data_1_fine)

    # session2数据处理
    train_data_2 = np.load(os.path.join(datapath, 'X_train_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    train_label_2 = np.load(os.path.join(datapath, 'y_train_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    test_data_2 = np.load(os.path.join(datapath, 'X_test_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)
    test_label_2 = np.load(os.path.join(datapath, 'y_test_S{:03d}_sess002.npy'.format(sub)), allow_pickle=True)

    train_data_2, train_label_2, test_data_2, test_label_2 = data_process(yml, chanDic, train_data_2, train_label_2,
                                                                          test_data_2, test_label_2)

    if yml['ML']['loss'] == 'binary_crossentropy':
        train_label_2 = train_label_2 - 1
        test_label_2 = test_label_2 - 1

    train_data_2, test_data_2 = trans(train_data_2, test_data_2)
    train_data_2 = dataset_1Dto2D(train_data_2)
    test_data_2 = dataset_1Dto2D(test_data_2)

    test_data = np.concatenate((train_data_1,test_data_1_fine,train_data_2, test_data_2), axis=0)
    test_label = np.concatenate((train_label_1,test_label_1_fine,train_label_2, test_label_2), axis=0)

    te_dataset['x'] = test_data
    te_dataset['y'] = test_label



    return te_dataset


def data_process_bci(yml,chanDic, train_data, train_label, test_data, test_label):
    eventTypeDic = {
        0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    train_select_event, test_select_event = [], []
    # train  events select
    train_left_Index = np.where(train_label[:,2] == 0)[0]
    train_right_Index = np.where(train_label[:,2] == 1)[0]
    train_event_Index = np.sort(np.concatenate((train_left_Index, train_right_Index)))
    #
    for i in train_event_Index:
        train_select_event.append(train_label[i])
    train_select_event = np.array(train_select_event, dtype='int32')

    # train数据切割
    train_x, train_y = getData_chanSeq_BCI2008_Train(yml, train_data, train_select_event,
                                               eventTypeDic=eventTypeDic, getAll=False)

    train_x,train_y = shuffle(train_x,train_y)
    #TEST数据切割
    test_left_Index = np.where(test_label[:,2] == 0)[0]
    test_right_Index = np.where(test_label[:,2] == 1)[0]
    test_event_Index = np.sort(np.concatenate((test_left_Index, test_right_Index)))

    for i in test_event_Index:
        test_select_event.append(test_label[i])
    test_select_event = np.array(test_select_event, dtype='int32')

    # test数据切割
    test_x, test_y = getData_chanSeq_BCI2008_Train(yml, test_data, test_select_event,
                                             eventTypeDic=eventTypeDic, getAll=False)

    test_x,test_y = shuffle(test_x,test_y)

    return train_x, train_y, test_x, test_y


def data_process(yml,chanDic, train_data, train_label, test_data, test_label):
    eventTypeDic = {
        1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    train_select_event, val_select_event, test_select_event = [], [], []
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

def data_1Dto2D_bci(data,x=4,y=9):
    data_2D = np.zeros([x,y])
    data_2D[0] = (   0,     0,    data[0],  data[1],  data[2], data[3],  data[4],     0,     0,   )
    data_2D[1] = (   0,  data[5], data[6],  data[7],  data[8], data[9],  data[10], data[11], 0,   )
    data_2D[2] = (   0,     0,    data[12], data[13], data[14],data[15], data[16],    0,     0,   )
    data_2D[3] = (   0,     0,       0 ,    data[17], data[18],data[19],    0,        0,     0,   )
    return data_2D


def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],dataset_1D.shape[1],4,9])
    for i in range(dataset_1D.shape[0]):
        for j in range(dataset_1D.shape[1]):
            dataset_2D[i,j] = data_1Dto2D(dataset_1D[i,j])
    return dataset_2D

def dataset_1Dto2D_bci(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],dataset_1D.shape[1],4,9])
    for i in range(dataset_1D.shape[0]):
        for j in range(dataset_1D.shape[1]):
            dataset_2D[i,j] = data_1Dto2D_bci(dataset_1D[i,j])
    return dataset_2D

def trans(tra,te):
    tra = np.swapaxes(tra, -1, -2)
    te = np.swapaxes(te, -1, -2)
    return tra, te

def tran(yml,y):
    if yml['ML']['loss'] == 'binary_crossentropy':
        y = K.argmax(y, axis=-1)
        y = y.numpy().tolist()
        y = np.array(y, dtype='int32')
    return y

def writerxlsx_1(path, data, dataname):
    workbook = xw.Workbook(path)
    worksheet = workbook.add_worksheet(dataname)
    row = 0
    col = 0
    for a, b, c, d, e, f, g, h, i, j in data:
        worksheet.write(row, col, a)
        worksheet.write(row, col+1, b)
        worksheet.write(row, col + 2, c)
        worksheet.write(row, col + 3, d)
        worksheet.write(row, col + 4, e)
        worksheet.write(row, col + 5, f)
        worksheet.write(row, col + 6, g)
        worksheet.write(row, col + 7, h)
        worksheet.write(row, col + 8, i)
        worksheet.write(row, col + 9, j)
        row += 1
    workbook.close()