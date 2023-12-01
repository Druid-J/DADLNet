import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import os
from MI.preprocess.MnnuData import raw
from MI.preprocess.config import CONSTANT
from meya.fileAction import joinPath,saveFile
from meya.utils import CVSegment
from MI.preprocess.MnnuData.raw import read_raw,load_filter_data
from MNUMI.loadData import getRawEvent
from meya.loadData_YML import getData_chanSeq_OpenBMI
import pandas as pd
import sys
import logging
import glob
from sklearn.utils import shuffle
CONSTANT = CONSTANT['OpenBMI']
n_subjs = CONSTANT['n_subjs']

def Subject_session_DataGenerate(SavaPath,yml,evenTypeDic):
    "The mat data are read, pre-processed with filtering and downsampling, "
    "and finally the corresponding PKL/NPY files are generated."
    session = [1,2]
    n_subjs = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,19,20,22,23,24,26,27,29,31,32]
    # n_subjs = [1, 2, 3, 4, 8, 9, 10]
    Sub_dic = {}
    for subject in n_subjs:
        for id_se,sess in enumerate(session):
            BdfPath = joinPath(yml['Meta']['initDataFolder'], 's{0:02d}'.format(subject),'session{0:02d}'.format(sess),'data.bdf')
            raw,events = getRawEvent(BdfPath,yml,evenTypeDic,subject,sess)

            save_name = '/S{:03d}_sess{:03d}'.format(subject,sess)
            # Sub_dic.setdefault(save_name+'_train',Sub_dic1)
            # Sub_dic.setdefault(save_name + '_test', Sub_dic2)
            __save_data(SavaPath,save_name,raw,events)
            print('sub{:02d}_session{:02d}Data save！'.format(subject,sess))
    # df = pd.DataFrame(Sub_dic)
    # df.to_csv(SavaPath+'/OpenBMI.csv')

def subject_dependent_setting_spilt(yml, k_folds,pick_smp_freq, ProDataPath,save_path, num_class,n_subjs):
    "先滑动切割然后再生成训练和验证集"
    TrainType = 'dependent'
    n_folds = k_folds
    X_train_all, y_train_all, X_test_all, y_test_all = __load_OpenBMI_dependent(ProDataPath, n_subjs, pick_smp_freq, TrainType,num_class,
                                                                     yml)

    # Carry out subject-dependent setting with k-fold cross validation
    for person, (X_tr, y_tr, X_te, y_te) in enumerate(zip(X_train_all, y_train_all, X_test_all, y_test_all)):
        if len(X_tr.shape) != 3:
            raise Exception('Dimension Error, must have 3 dimension')

        kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
        # x_tr(200,20,400) skf是对200这个维度进行划分 random42?
        for fold, (train_index, val_index) in enumerate(kf.split(X_tr)):
            print('FOLD:', fold + 1, 'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))
            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]

            X_tr_channel = dataset_1Dto2D(X_tr_cv)
            X_val_channel = dataset_1Dto2D(X_val_cv)
            X_te_channel = dataset_1Dto2D(X_te)
            print('Check dimension of training data {}, val data {} and testing data {}'.format(X_tr_channel.shape,
                                                                                                X_val_channel.shape,
                                                                                                X_te_channel.shape))
            SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person + 1, fold + 1)
            __save_data_with_valset(save_path, SAVE_NAME, X_tr_channel, y_tr_cv, X_val_channel, y_val_cv, X_te_channel, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person + 1, fold + 1))

def subject_dependent_setting_FBCNet(yml, k_folds,chanDic, ProDataPath,save_path, num_class,n_subjs):
    "先分训练和验证数据集再滑动切割"
    TrainType = 'dependent'
    n_folds = k_folds
    if TrainType == 'dependent':
         if num_class == 2:
            eventTypeDic = {
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
            print("Tow-class dependent MI data is downloading")
            sessions = [1,2]
            for subject in range(1, n_subjs + 1):
                TestData,TestLabel=[],[]
                CrossTrainDict_sess1 = {"1": {"data": [], "label": []}, "2": {"data": [], "label": []},
                                      "3": {"data": [], "label": []}, "4": {"data": [], "label": []},
                                      "5": {"data": [], "label": []}}
                CrossValidDict_sess1 = {"1": {"data": [], "label": []}, "2": {"data": [], "label": []},
                                      "3": {"data": [], "label": []}, "4": {"data": [], "label": []},
                                      "5": {"data": [], "label": []}}
                CrossTrainDict_sess2 = {"1": {"data": [], "label": []}, "2": {"data": [], "label": []},
                                      "3": {"data": [], "label": []}, "4": {"data": [], "label": []},
                                      "5": {"data": [], "label": []}}
                CrossValidDict_sess2 = {"1": {"data": [], "label": []}, "2": {"data": [], "label": []},
                                      "3": {"data": [], "label": []}, "4": {"data": [], "label": []},
                                      "5": {"data": [], "label": []}}
                for id_se, sess in enumerate(sessions):

                    # 加载数据这里的X是raw_data 和 event
                    offlineRaw,offlineEvent,onlineRaw,onlineEvent= load_filter_data(ProDataPath, subject, sess)
                    #FBCNet多频率段数据
                    for filternum,(data,label) in enumerate(zip(offlineRaw,offlineEvent)):
                        offline_select_event, offline_select_label = [],[]
                        # 1.离线数据的event选择 1为右手 2为左手
                        offline_right_Index = np.where(label[:, 2] == 1)[0]
                        offline_left_Index = np.where(label[:, 2] == 2)[0]
                        offline_event_Index = np.sort(np.concatenate((offline_left_Index, offline_right_Index)))
                        for i in offline_event_Index:
                            offline_select_event.append(label[i])
                            offline_select_label.append(label[i][2])
                        offline_select_event = np.array(offline_select_event, dtype='int32')
                        offline_select_label = np.array(offline_select_label, dtype='int32')
                        # 2.对相关事件的events进行交叉验证
                        SKf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
                        for fold, (train, val) in enumerate(SKf.split(offline_select_event, offline_select_label)):
                            offline_event_train = offline_select_event[train]
                            offline_event_val = offline_select_event[val]

                            #80 => 4*80=320 数据切割
                            sk_train_x, sk_train_y = getData_chanSeq_OpenBMI(yml, data, offline_event_train,chanDic,
                                                                            eventTypeDic=eventTypeDic,getAll=False)
                            sk_val_x, sk_val_y = getData_chanSeq_OpenBMI(yml, data, offline_event_val,chanDic,
                                                                               eventTypeDic=eventTypeDic,getAll=False)


                            #分别保存到字典
                            # X_tr_Dic = {"TraData":sk_train_x,'Tralabel':sk_train_y}
                            # X_val_Dic = {'ValData':sk_val_x,'Vallabel':sk_val_y}

                            #对数据来源进行判断
                            if sess == 1:
                                CrossTrainDict_sess1[str(fold + 1)]["data"].append(sk_train_x)
                                CrossValidDict_sess1[str(fold + 1)]["data"].append(sk_val_x)
                                if filternum == 0:
                                    CrossTrainDict_sess1[str(fold+1)]["label"]= sk_train_y
                                    CrossValidDict_sess1[str(fold+1)]["label"]= sk_val_y
                            elif sess == 2:
                                CrossTrainDict_sess2[str(fold + 1)]["data"].append(sk_train_x)
                                CrossValidDict_sess2[str(fold + 1)]["data"].append(sk_val_x)
                                if filternum == 0:
                                    CrossTrainDict_sess2[str(fold+1)]["label"]= sk_train_y
                                    CrossValidDict_sess2[str(fold+1)]["label"]= sk_val_y


                    MultiData,Multilabel = [],[]
                    for filternum,(onlindata,onlinelabel) in enumerate(zip(onlineRaw,onlineEvent)):
                        online_select_event, online_select_label = [], []
                        # 2.在线数据event选择
                        online_right_Index = np.where(onlinelabel[:, 2] == 1)[0]
                        online_left_Index = np.where(onlinelabel[:, 2] == 2)[0]
                        online_event_Index = np.sort(np.concatenate((online_left_Index, online_right_Index)))

                        for i in online_event_Index:
                            online_select_event.append(onlinelabel[i])
                            online_select_label.append(onlinelabel[i][2])
                        online_select_event = np.array(online_select_event, dtype='int32')
                        online_select_label = np.array(online_select_label, dtype='int32')

                        #在线数据切割
                        test_x, test_y = getData_chanSeq_OpenBMI(yml, onlindata,online_select_event,chanDic,
                                                                       eventTypeDic=eventTypeDic,getAll=False)
                        #通道后置
                        # test_x = np.swapaxes(test_x, -1, -2)
                        #1D转为2D
                        # X_test_channel = dataset_1Dto2D(test_x)
                        # X_test_channel, test_y = shuffle(X_test_channel, test_y)
                        #保存两个session数据
                        MultiData.append(test_x)
                        if filternum == 0:
                            Multilabel =  test_y
                    MultiData = np.array(MultiData,dtype='float32').swapaxes(0,1)
                    Multilabel = np.array(Multilabel,dtype='int32')
                    TestData.extend(MultiData)
                    TestLabel.extend(Multilabel)

                TestData = np.array(TestData,dtype='float32')
                TestLabel = np.array(TestLabel,dtype='int32')

                #数据保存
                print("=================Start Save==================")
                for key in range(1,n_folds+1):
                    CrossTrainDict_sess1[str(key)]['data'] = np.array(CrossTrainDict_sess1[str(key)]['data'],
                                                                      dtype='float32').swapaxes(0,1)
                    CrossValidDict_sess1[str(key)]['data'] = np.array(CrossValidDict_sess1[str(key)]['data'],
                                                                      dtype='float32').swapaxes(0,1)
                    CrossTrainDict_sess2[str(key)]['data'] = np.array(CrossTrainDict_sess2[str(key)]['data'],
                                                                      dtype='float32').swapaxes(0, 1)
                    CrossValidDict_sess2[str(key)]['data'] = np.array(CrossValidDict_sess2[str(key)]['data'],
                                                                      dtype='float32').swapaxes(0, 1)
                    train_data = np.concatenate((CrossTrainDict_sess1[str(key)]['data'],
                                                 CrossTrainDict_sess2[str(key)]['data']))
                    train_label = np.concatenate((CrossTrainDict_sess1[str(key)]['label'],
                                                  CrossTrainDict_sess2[str(key)]['label']))

                    valid_data = np.concatenate((CrossValidDict_sess1[str(key)]['data'],
                                                 CrossValidDict_sess2[str(key)]['data']))
                    valid_label = np.concatenate((CrossValidDict_sess1[str(key)]['label'],
                                                  CrossValidDict_sess2[str(key)]['label']))

                    print("Check dimension of Train data：{} and Train label {}".format(train_data.shape,
                                                                                       train_label.shape))
                    print("Check dimension of Valid data：{} and Valid label {}".format(valid_data.shape,
                                                                                       valid_label.shape))
                    print("Check dimension of test data:{} and test label:{}".format(TestData.shape,
                                                                                     TestLabel.shape))
                    SAVE_NAME = 'S{:03d}_fold{:03d}'.format(subject, key)
                    __save_data_with_valset(save_path, SAVE_NAME, train_data,train_label,valid_data,valid_label,TestData, TestLabel)
                    print('The Test Set of subject {}  is DONE!!!'.format(subject))
                    print("=================== END =====================")


def subject_dependent_setting_spilt(yml, k_folds,chanDic, ProDataPath,save_path, num_class,n_subjs):
    "先分训练和验证数据集再滑动切割"
    TrainType = 'dependent'
    n_folds = k_folds
    if TrainType == 'dependent':
         if num_class == 2:
            eventTypeDic = {
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
            print("Tow-class dependent MI data is downloading")
            sessions = [1,2]
            for subject in range(1, n_subjs + 1):
                TestData,TestLabel=[],[]
                CrossTrainDict_sess1 = {"1": [], "2": [], "3": [], "4": [], "5": []}
                CrossValidDict_sess1 = {"1": [], "2": [], "3": [], "4": [], "5": []}
                CrossTrainDict_sess2 = {"1": [], "2": [], "3": [], "4": [], "5": []}
                CrossValidDict_sess2 = {"1": [], "2": [], "3": [], "4": [], "5": []}
                for id_se, sess in enumerate(sessions):
                    offline_select_event, offline_select_label,online_select_event,online_select_label = [],[],[],[]
                    offlineRaw,offlineEvent,onlineRaw,onlineEvent= load_filter_data(ProDataPath, subject, sess)
                    offline_right_Index = np.where(offlineEvent[:, 2] == 1)[0]
                    offline_left_Index = np.where(offlineEvent[:, 2] == 2)[0]
                    offline_event_Index = np.sort(np.concatenate((offline_left_Index, offline_right_Index)))
                    for i in offline_event_Index:
                        offline_select_event.append(offlineEvent[i])
                        offline_select_label.append(offlineEvent[i][2])
                    offline_select_event = np.array(offline_select_event, dtype='int32')
                    offline_select_label = np.array(offline_select_label, dtype='int32')

                    SKf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
                    for fold, (train, val) in enumerate(SKf.split(offline_select_event, offline_select_label)):
                        offline_event_train = offline_select_event[train]
                        offline_event_val = offline_select_event[val]

                        sk_train_x, sk_train_y = getData_chanSeq_OpenBMI(yml, offlineRaw, offline_event_train,chanDic,
                                                                        eventTypeDic=eventTypeDic,getAll=False)
                        sk_val_x, sk_val_y = getData_chanSeq_OpenBMI(yml, offlineRaw, offline_event_val,chanDic,
                                                                           eventTypeDic=eventTypeDic,getAll=False)
                        sk_train_x = np.swapaxes(sk_train_x, -1, -2)
                        sk_val_x = np.swapaxes(sk_val_x, -1, -2)

                        chanNum = yml['Meta']['ChannelNum']
                        sk_train_x = dataset_1Dto2D(chanNum,sk_train_x)
                        sk_val_x = dataset_1Dto2D(chanNum,sk_val_x)

                        X_tr_channel, sk_train_y = shuffle(sk_train_x, sk_train_y)
                        X_val_channel, sk_val_y = shuffle(sk_val_x, sk_val_y)

                        X_tr_Dic = {"TraData":X_tr_channel,'Tralabel':sk_train_y}
                        X_val_Dic = {'ValData':X_val_channel,'Vallabel':sk_val_y}


                        if sess == 1:
                            CrossTrainDict_sess1[str(fold + 1)]=X_tr_Dic
                            CrossValidDict_sess1[str(fold + 1)]=X_val_Dic
                        elif sess == 2:
                            CrossTrainDict_sess2[str(fold + 1)] = X_tr_Dic
                            CrossValidDict_sess2[str(fold + 1)] = X_val_Dic


                    online_right_Index = np.where(onlineEvent[:, 2] == 1)[0]
                    online_left_Index = np.where(onlineEvent[:, 2] == 2)[0]
                    online_event_Index = np.sort(np.concatenate((online_left_Index, online_right_Index)))

                    for i in online_event_Index:
                        online_select_event.append(onlineEvent[i])
                        online_select_label.append(onlineEvent[i][2])
                    online_select_event = np.array(online_select_event, dtype='int32')



                    test_x, test_y = getData_chanSeq_OpenBMI(yml, onlineRaw, online_select_event,chanDic,
                                                                   eventTypeDic=eventTypeDic,getAll=False)

                    test_x = np.swapaxes(test_x, -1, -2)

                    chanNum = yml['Meta']['ChannelNum']
                    test_x = dataset_1Dto2D(chanNum,test_x)
                    X_test_channel, test_y = shuffle(test_x, test_y)

                    TestData.extend(X_test_channel)
                    TestLabel.extend(test_y)

                TestData = np.array(TestData,dtype='float32')
                TestLabel = np.array(TestLabel,dtype='int32')


                print("=================Start Save==================")
                for key in range(1,n_folds+1):
                    train_data = np.concatenate((CrossTrainDict_sess1[str(key)]['TraData'],
                                                 CrossTrainDict_sess2[str(key)]['TraData']))
                    train_label = np.concatenate((CrossTrainDict_sess1[str(key)]['Tralabel'],
                                                  CrossTrainDict_sess2[str(key)]['Tralabel']))

                    valid_data = np.concatenate((CrossValidDict_sess1[str(key)]['ValData'],
                                                 CrossValidDict_sess2[str(key)]['ValData']))
                    valid_label = np.concatenate((CrossValidDict_sess1[str(key)]['Vallabel'],
                                                  CrossValidDict_sess2[str(key)]['Vallabel']))

                    print("Check dimension of Train data：{} and Train label {}".format(train_data.shape,
                                                                                       train_label.shape))
                    print("Check dimension of Valid data：{} and Valid label {}".format(valid_data.shape,
                                                                                       valid_label.shape))
                    print("Check dimension of test data:{} and test label:{}".format(TestData.shape,
                                                                                     TestLabel.shape))
                    SAVE_NAME = 'S{:03d}_fold{:03d}'.format(subject, key)
                    __save_data_with_valset(save_path, SAVE_NAME, train_data,train_label,valid_data,valid_label,TestData, TestLabel)
                    print('The Test Set of subject {}  is DONE!!!'.format(subject))
                    print("=================== END =====================")


def subject_independent_setting_spilt(yml,chanDic,ProDataPath,save_path, num_class):
    TrainType = 'independent'
    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']
    log = logging.getLogger()
    folderNum = 5
    if 'folderNum' in yml['ML']:
        folderNum = yml['ML']['folderNum']

    saveFolder = joinPath(basicFolder, folderName)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    Subnum = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,19,20,22,23,24,26,27,29,31,32]
    # Subnum = [1, 2, 3, 4, 8, 9, 10]
    # for i in range(n_subjs):
    #     Subnum.append(i + 1)

    crossValue = {'Train': [], 'valid': [], 'Test': []}
    crossValidFile = '%s/%s_crossValue.csv' % (saveFolder, folderName)

    for index, person in enumerate(Subnum):
        Target_sub = []
        if person in [33] :
            continue
        else:
            if index != 0:
                crossValue['Test'] = []
            Target_sub.append(person)
            train_subj = np.delete(Subnum, index)

            crossValue['Train'], crossValue['valid'], _ = CVSegment(train_subj, kFoderIndex=yml['ML']['folderNum'])
            for i in range(folderNum):
                crossValue['Test'].append(Target_sub)
            if index == 0:
                df = pd.DataFrame(crossValue, index=["1", "2", "3", "4", "5"], columns=['Train', 'valid', 'Test'])
            else:
                df2 = pd.DataFrame(crossValue, index=["1", "2", "3", "4", "5"], columns=['Train', 'valid', 'Test'])
                df = df.append(df2, ignore_index=False)
            df.to_csv(crossValidFile)
            crossIndexs = np.array(range(0, len(crossValue['Train'])), dtype='int32')
            saveFile(saveFolder, sys.argv[1], cover=True)

        for i in range(0,5):
            sub = []
            train_subs = crossValue['Train'][i]
            valid_subs = crossValue['valid'][i]
            test_subs = crossValue['Test'][i]
            sub.append(test_subs[0])
            folderIndex = i + 1
            try:
                filename = save_path+'/X_train_S{:03d}_fold{:03d}.npy'.format(test_subs[0],folderIndex)
                if os.path.exists(filename):
                    print('npy file exist')
                    continue
                else:
                    print("start load train sub:{}".format(train_subs))
                    TrainData,TrainLabel = __load_OpenBMI_split(ProDataPath,train_subs, chanDic,TrainType,num_class,yml=yml)
                    print("start load valid sub:{}".format(valid_subs))
                    ValidData,ValidLabel = __load_OpenBMI_split(ProDataPath,valid_subs, chanDic,TrainType,num_class,yml=yml)
                    print("start load test sub:{}".format(test_subs))
                    TestData,TestLabel = __load_OpenBMI_split(ProDataPath,sub, chanDic,TrainType,num_class,yml=yml)

                    channel_num = yml['Meta']['ChannelNum']
                    X_train_fil = dataset_1Dto2D(channel_num,TrainData)
                    X_val_fil = dataset_1Dto2D(channel_num,ValidData)
                    X_test_fil = dataset_1Dto2D(channel_num,TestData)

                    print(
                        "Check dimension of training data {},training data label{} ".format(X_train_fil.shape,
                                                                                            TrainLabel.shape))
                    print(
                        "Check dimension of valid data {},valid data label{} ".format(X_val_fil.shape,
                                                                                      ValidLabel.shape))
                    print(
                        "Check dimension of test data {},test data label{} ".format(X_test_fil.shape,
                                                                                    TestLabel.shape))

                    SAVE_NAME = 'S{:03d}_fold{:03d}'.format(test_subs[0], folderIndex)
                    __save_data_with_valset(save_path, SAVE_NAME, X_train_fil, TrainLabel, X_val_fil, ValidLabel, X_test_fil,
                                                TestLabel)
                    print(
                        "Complete Subject{}_fold{} data save".format(test_subs[0],folderIndex)
                    )
            except Exception as err:
                print(err)
                log.error(err)
                raise err

def subject_independent_setting_spilt_FBCNet(yml,chanDic,ProDataPath,save_path, num_class,n_subjs):
    TrainType = 'independent'

    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']
    log = logging.getLogger()
    folderNum = 5
    if 'folderNum' in yml['ML']:
        folderNum = yml['ML']['folderNum']

    saveFolder = joinPath(basicFolder, folderName)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    Subnum = []
    for i in range(n_subjs):
        Subnum.append(i + 1)

    crossValue = {'Train': [], 'valid': [], 'Test': []}
    crossValidFile = '%s/%s_crossValue.csv' % (saveFolder, folderName)

    for index, person in enumerate(Subnum):
        Target_sub = []
        if person  in [55] :
            continue
        else:
            if index != 0:
                crossValue['Test'] = []
            Target_sub.append(person)
            train_subj = np.delete(Subnum, index)

            crossValue['Train'], crossValue['valid'], _ = CVSegment(train_subj, kFoderIndex=yml['ML']['folderNum'])
            for i in range(folderNum):
                crossValue['Test'].append(Target_sub)

            if index == 0:
                df = pd.DataFrame(crossValue, index=["1", "2", "3", "4", "5"], columns=['Train', 'valid', 'Test'])
            else:
                df2 = pd.DataFrame(crossValue, index=["1", "2", "3", "4", "5"], columns=['Train', 'valid', 'Test'])
                df = df.append(df2, ignore_index=False)
            df.to_csv(crossValidFile)
            crossIndexs = np.array(range(0, len(crossValue['Train'])), dtype='int32')
            saveFile(saveFolder, sys.argv[1], cover=True)

        for i in range(0,5):
            sub = []
            train_subs = crossValue['Train'][i]
            valid_subs = crossValue['valid'][i]
            test_subs = crossValue['Test'][i]
            sub.append(test_subs[0])
            folderIndex = i + 1
            try:
                filename = '/home/xumeiyan/Public/Data/MI/OpenBMI/Traindata/independent/2_class/10Hz_49chan_8_30Hz_noslid/X_train_S{:03d}_fold{:03d}.npy'.format(test_subs[0],folderIndex)
                if os.path.exists(filename):
                    print('npy file exist')
                else:
                    print("start load train sub:{}".format(train_subs))
                    TrainData,TrainLabel = __load_OpenBMI_split_FBCNet(ProDataPath,train_subs, chanDic,TrainType,num_class,yml=yml)
                    print("start load valid sub:{}".format(valid_subs))
                    ValidData,ValidLabel = __load_OpenBMI_split_FBCNet(ProDataPath,valid_subs, chanDic,TrainType,num_class,yml=yml)
                    print("start load test sub:{}".format(test_subs))
                    TestData,TestLabel = __load_OpenBMI_split_FBCNet(ProDataPath,sub, chanDic,TrainType,num_class,yml=yml)

                    # X_train_fil = dataset_1Dto2D(TrainData)
                    # X_val_fil = dataset_1Dto2D(ValidData)
                    # X_test_fil = dataset_1Dto2D(TestData)

                    print(
                        "Check dimension of training data {},training data label{} ".format(TrainData.shape,
                                                                                            TrainLabel.shape))
                    print(
                        "Check dimension of valid data {},valid data label{} ".format(ValidData.shape,
                                                                                      ValidLabel.shape))
                    print(
                        "Check dimension of test data {},test data label{} ".format(TestData.shape,
                                                                                    TestLabel.shape))

                    SAVE_NAME = 'S{:03d}_fold{:03d}'.format(test_subs[0], folderIndex)
                    __save_data_with_valset(save_path, SAVE_NAME, TrainData, TrainLabel, ValidData, ValidLabel, TestData,
                                                TestLabel)
                    print(
                        "Complete Subject{}_fold{} data save".format(test_subs[0],folderIndex)
                    )
            except Exception as err:
                print(err)
                log.error(err)
                raise err

#OpenBMI 20channel
# def data_1Dto2D(data,x=3,y=7):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (data[0],    data[1],    data[2],    0,         data[3],    data[4],    data[5], )
#     data_2D[1] = (data[6],    data[7],    data[8],    data[9],   data[10],   data[11],   data[12],)
#     data_2D[2] = (data[13],   data[14],   data[15],   data[16],  data[17],   data[18],   data[19],)
#     return data_2D

# def data_1Dto2D(data,x=6,y=7):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (data[0],    data[1],    data[2],       0,      data[3],    data[4],    data[5], )
#     data_2D[1] = (   0,          0,           0,         0,         0,          0,           0,   )
#     data_2D[2] = (data[6],    data[7],    data[8],    data[9],   data[10],   data[11],   data[12],)
#     data_2D[3] = (   0,          0,           0,         0,         0,          0,           0,   )
#     data_2D[4] = (data[13],   data[14],   data[15],   data[16],  data[17],   data[18],   data[19],)
#     data_2D[5] = (   0,          0,           0,         0,         0,          0,           0,   )
#     return data_2D

#稀疏矩阵(14*7)
# def data_1Dto2D(data,x=14,y=7):
#     data_2D = np.zeros([y,x])
#     data_2D[0] = (0,       0,          0,        0,        0,         0,        0,         0,          0,       0,        0,      0,        0,         0,       )
#     data_2D[1] = (0,       data[0],    0,        data[1],  0,         data[2],  0,         0,          0,       data[3],  0,      data[4],  0,         data[5], )
#     data_2D[2] = (0,       0,          0,        0,        0,         0,        0,         0,          0,       0,        0,      0,        0,         0,       )
#     data_2D[3] = (0,       data[6],    0,        data[7],  0,         data[8],  0,         data[9],    0,       data[10], 0,      data[11], 0,         data[12],)
#     data_2D[4] = (0,       0,          0,        0,        0,         0,        0,         0,          0,       0,        0,      0,        0,         0,       )
#     data_2D[5] = (0,       data[13],   0,        data[14], 0,         data[15], 0,         data[16],   0,       data[17], 0,      data[18], 0,         data[19],)
#     data_2D[6] = (0,       0,          0,        0,        0,         0,        0,         0,          0,       0,        0,      0,        0,         0,       )
#     return data_2D


#OpenBMI 31channel
# def data_1Dto2D(data,x=4,y=9):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (   0,    data[0], data[1], data[2],    0,     data[3],  data[4],  data[5],    0,    )
#     data_2D[1] = (data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14])
#     data_2D[2] = (data[15],data[16],data[17],data[18],data[19], data[20], data[21], data[22], data[23])
#     data_2D[3] = (data[24],  0,     data[25],data[26],data[27], data[28], data[29],     0,    data[30])
#     return data_2D

#OpenBMI 31channel 稀疏
# def data_1Dto2D(data,x=7,y=17):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (   0,    0, data[0], 0, data[1], 0, data[2], 0,    0,    0, data[3], 0, data[4], 0, data[5], 0,    0,   )
#     data_2D[1] = (   0,    0,  0,      0,     0,   0,    0,    0,    0,    0,   0,     0,   0,     0,   0,     0,    0,   )
#     data_2D[2] = (data[6], 0, data[7], 0, data[8], 0, data[9], 0, data[10],0, data[11],0, data[12],0, data[13],0, data[14])
#     data_2D[3] = (   0,    0,   0,     0,    0,    0,    0,    0,     0,   0,   0,     0,    0,    0,    0,    0,    0,   )
#     data_2D[4] = (data[15],0, data[16],0, data[17],0, data[18],0, data[19],0, data[20],0, data[21],0, data[22],0, data[23])
#     data_2D[5] = (   0,    0,   0,     0,    0,    0,    0,    0,     0,   0,    0,    0,    0,    0,    0,    0,    0,   )
#     data_2D[6] = (data[24],0,   0,     0, data[25],0, data[26],0, data[27],0, data[28],0, data[29],0,    0,    0, data[30])
#     return data_2D

def data_1Dto2D(data,x=4,y=9):
    data_2D = np.zeros([x,y])
    data_2D[0] = (   0,     data[0],  data[1],  data[2],     0,     data[3],  data[4],  data[5],     0,   )
    data_2D[1] = (data[6],  data[7],  data[8],  data[9],  data[10], data[11], data[12], data[13], data[14])
    data_2D[2] = (data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23])
    data_2D[3] = (data[24],   0,      data[25], data[26], data[27], data[28], data[29],    0,     data[30])
    return data_2D

# def data_1Dto2D_8chan(data,x=3,y=3):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (data[0],     0,    data[1],)
#     data_2D[1] = (data[2],  data[3], data[4],)
#     data_2D[2] = (data[5],  data[6], data[7],)
#     return data_2D
#
# def data_1Dto2D_8chan(data,x=3,y=3):
#     data_2D = np.zeros([x,y])
#     data_2D[0] = (data[0],     0,   data[1],)
#     data_2D[1] = (data[2], data[3], data[4],)
#     data_2D[2] = (data[5], data[6], data[7],)
#     return data_2D

def data_1Dto2D_8chan(data,x=3,y=3):
    data_2D = np.zeros([x,y])
    data_2D[0] = (data[0],     0,     data[1],  )
    data_2D[1] = (data[2],  data[3],  data[4],  )
    data_2D[2] = (data[5],  data[6],  data[7],  )
    return data_2D

def data_1Dto2D_14chan(data,x=3,y=5):
    data_2D = np.zeros([x,y])
    data_2D[0] = (data[0],   data[1],     0,     data[2],  data[3], )
    data_2D[1] = (data[4],   data[5],  data[6],  data[7],  data[8], )
    data_2D[2] = (data[9],   data[10], data[11], data[12], data[13],)
    return data_2D

def data_1Dto2D_20chan(data,x=3,y=7):
    data_2D = np.zeros([x,y])
    data_2D[0] = (data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6],)
    data_2D[1] = (data[7],  data[8],  data[9],  data[10], data[11], data[12], data[13],)
    data_2D[2] = (data[14], data[15], data[16],    0,     data[17], data[18], data[19],)
    return data_2D
#(16800,400,20) 将20通道转为稀疏矩阵表示
def dataset_1Dto2D(channNum,dataset_1D):
    if channNum ==31:
        dataset_2D = np.zeros([dataset_1D.shape[0],dataset_1D.shape[1],4,9])
    elif channNum == 20:
        dataset_2D = np.zeros([dataset_1D.shape[0], dataset_1D.shape[1], 3, 7])
    elif channNum == 14:
        dataset_2D = np.zeros([dataset_1D.shape[0], dataset_1D.shape[1], 3, 5])
    elif channNum == 8:
        dataset_2D = np.zeros([dataset_1D.shape[0], dataset_1D.shape[1], 3, 3])
    for i in range(dataset_1D.shape[0]):
        for j in range(dataset_1D.shape[1]):
            if channNum == 31:
                dataset_2D[i,j] = data_1Dto2D(dataset_1D[i,j])
            elif channNum == 20:
                dataset_2D[i,j] = data_1Dto2D_20chan(dataset_1D[i,j])
            elif channNum == 14:
                dataset_2D[i,j] = data_1Dto2D_14chan(dataset_1D[i,j])
            elif channNum == 8:
                dataset_2D[i,j] = data_1Dto2D_8chan(dataset_1D[i,j])
    return dataset_2D


def __load_OpenBMI_dependent(PATH, n_subjs, new_smp_freq,TrainType,num_class, yml):
    #x_train_all 所有训练的数据x_train_all（54,200，20,400）
    X_train_all, y_train_all, X_test_all, y_test_all = raw.load_crop_data(PATH, n_subjs, new_smp_freq, TrainType, num_class, yml)
    return X_train_all, y_train_all, X_test_all, y_test_all

def __load_OpenBMI_split(PATH, n_subjs, chanDic, TrainType,num_class, yml):
    #x_train_all 所有训练的数据x_train_all（54,200，20,400）
    X_train_all, y_train_all = raw.load_crop_data(PATH, n_subjs, chanDic, TrainType, num_class, yml)
    return X_train_all, y_train_all

def __load_OpenBMI_split_FBCNet(PATH, n_subjs, chanDic, TrainType,num_class, yml):
    #x_train_all 所有训练的数据x_train_all（54,200，20,400）
    X_train_all, y_train_all = raw.load_crop_data_FBCNet(PATH, n_subjs, chanDic, TrainType, num_class, yml)
    return X_train_all, y_train_all


def __save_data_with_valset(save_path, NAME, X_train, y_train, X_val, y_val, X_test, y_test):
    print("Train-Magnitude",np.mean(np.absolute(X_train)))
    print("Val-Magnitude", np.mean(np.absolute(X_val)))
    print("Test-Magnitude", np.mean(np.absolute(X_test)))
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/X_val_' + NAME + '.npy', X_val)
    np.save(save_path + '/y_val_' + NAME + '.npy', y_val)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)

    # print('save DONE')

def __save_data(save_path, NAME, raw, events):
    np.save(save_path + NAME + '_Raw'+'.npy', raw)
    np.save(save_path + NAME + '_Events'+'.npy', events)

def __save_data_test(save_path, NAME, X_test, y_test):
    np.save(save_path+'/X_test_'+NAME+'.npy', X_test)
    np.save(save_path+'/y_test_'+NAME+'.npy', y_test)
    print('save DONE')

def __save_data_train(save_path, NAME, X_train, y_train):
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    print('save DONE')

def __save_data_valid(save_path, NAME, X_valid, y_valid):
    np.save(save_path+'/X_val_'+NAME+'.npy', X_valid)
    np.save(save_path+'/y_val_'+NAME+'.npy', y_valid)
    print('save DONE')
