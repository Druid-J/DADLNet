import os, tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import sys

sys.path.append('../..')
import yaml
import logging
import numpy as np
from meya.fileAction import joinPath, getModelDic, saveFile, saveRawDataToFile
from meya.appLog import createLog
from tensorflow.keras.callbacks import EarlyStopping
from meya.MLTrain import trainCount
from EEGML import ML_Model
import pickle, os
from sklearn.model_selection import KFold
import tensorflow.keras as keras

from meya.loadModel import GetModelPath
from meya.basicModel import modelComFun
from sklearn.model_selection import train_test_split
import csv
from MI.cross_sub_transfer.load_data_cross import get_all_data
from MI.cross_sub_transfer.load_data_cross_new import get_data
from meya.MLTrain import countNum
from tensorflow.keras.optimizers import Nadam
import tensorflow.keras.backend as K
from sklearn.metrics import recall_score, confusion_matrix, f1_score, precision_score
from meya.effect import calculate_performance, calculate_performance_dual
from tensorflow.keras.models import Model as M
# from MI.intersub_transfer.person_loss import Dense_with_person
from MI.intersub_transfer.tool import writerxlsx_1
import copy


def GetData(yml, sub, datapath, eventTypeDic, folderIndex):
    # learnData learnLabel中存放的是所有训练对象的数据
    learnData_tra, learnLabel_tra = load_train_Data(sub, datapath, folderIndex)
    learnData_val, learnLabel_val = load_val_data(sub, datapath, folderIndex)
    # learnLabel_tra.tolist()
    # learnLabel_val.tolist()
    # learnLabel_tra = to_categorical(learnLabel_tra)
    # learnLabel_val = to_categorical(learnLabel_val)
    print('Check dimension of training data {}, val data {} '.format(learnData_tra.shape, learnData_val.shape, ))
    # basicFolder:/data0/meya/MI/Intrasub  folderName:1S_FB1-100_S100_chanSeq_20220630_FRA_2sessionALL
    subFolderPath = joinPath(yml["Meta"]['basicFolder'], yml["Meta"]['folderName'])
    validData = (learnData_val, learnLabel_val)
    trainCount(subFolderPath, folderIndex, yml, learnData_tra, learnLabel_tra, sub, validData, ML_Model, eventTypeDic,
               modelPath)
    print("Compelete loda folder %d Data!" % folderIndex)
    # return learnData,learnLabel


def load_train_Data(sub, load_path, folderIndex):
    try:
        file_x = load_path + '/X_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x, allow_pickle=True)
        y = np.load(file_y, allow_pickle=True)
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X, y


def load_val_data(sub, load_path, folderIndex):
    try:
        file_x = load_path + '/X_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x, allow_pickle=True)
        y = np.load(file_y, allow_pickle=True)
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X, y


# def load_test_data(sub,MNNUdataPath,folderIndex):
#     pass


def __save_data_with_valset(save_path, NAME, X_train, y_train, X_val, y_val, X_test, y_test):
    saveRawDataToFile(save_path + '/X_train_' + NAME + '.npy', X_train)
    saveRawDataToFile(save_path + '/X_val_' + NAME + '.npy', X_val)
    saveRawDataToFile(save_path + '/X_test_' + NAME + '.npy', X_test)
    saveRawDataToFile(save_path + '/y_train_' + NAME + '.npy', y_train)
    saveRawDataToFile(save_path + '/y_val_' + NAME + '.npy', y_val)
    saveRawDataToFile(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')


def read_pickle(work_path):
    with open(work_path, 'rb') as f:
        try:
            history_data = pickle.load(f)
        except Exception as err:
            print(err)
            log.error(err)
            raise err
    folde_val_loss = min(history_data['val_loss'])
    return folde_val_loss


def difference_loss(y_true, y_pred):
    v1 = y_pred
    v2 = tf.transpose(y_true, perm=[1, 0, 2])
    loss = 0
    for i in range(len(v2)):
        loss += tf.reduce_mean(tf.math.abs(tf.keras.activations.softmax(v1) - tf.keras.activations.softmax(v2[i])))
    return loss



# pearson_loss
def pearson(y_true, y_pred):
    v1 = y_pred
    v2 = tf.transpose(y_true, perm=[1, 0])
    length = v1.shape[1]
    Exy = tf.linalg.tensor_diag_part((K.dot(v1, v2))) / length
    Ex = tf.reduce_sum(v1, axis=1) / length
    Ey = tf.reduce_sum(v2, axis=0) / length
    C = Exy - Ex * Ey
    res = C / tf.math.sqrt(tf.nn.moments(v1, axes=[1])[1] * tf.nn.moments(v2, axes=[0])[1])
    res = tf.math.abs(res)
    loss = 1 / tf.nn.moments(res, axes=[0])[0] - 1
    return loss


# mmd_loss
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s = tf.shape(source)[0]
    n_s = 64 if n_s is None else n_s
    n_t = tf.shape(target)[0]
    n_t = 64 if n_t is None else n_t
    n_samples = n_s + n_t
    total = tf.concat([source, target], axis=0)
    total0 = tf.expand_dims(total, axis=0)
    total1 = tf.expand_dims(total, axis=1)
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2), axis=2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / float(n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)   #[b,b]


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_s = tf.shape(source)[0]
    n_s = 64 if n_s is None else n_s
    n_t = tf.shape(target)[0]
    n_t = 64 if n_t is None else n_t
    XX = tf.reduce_sum(kernels[:n_s, :n_s]) / float(n_s ** 2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:]) / float(n_t ** 2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:]) / float(n_s * n_t)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s]) / float(n_s * n_t)
    loss = XX + YY - XY - YX
    return loss


def reset_model(param, Model):
    for layer in Model.layers:
        # 屏蔽预训练模型的权重
        layer.trainable = False
    if param <= 1 and param >= 3:
        print('param error!!!')
    if param == 1:
        # Freeze all layers.
        for layer in Model.layers[-3:-2]:
            # 解冻预训练模型的权重
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 2:
        for layer in Model.layers[-4:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 3:
        for layer in Model.layers[-2:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 4:
        for layer in Model.layers[-5:]:
            layer.trainable = True
        new_model = Model
        return new_model


def class_weight_f(yml, y):
    _, class_weight, _ = countNum(dataLabel=y, eventTypeDic=eventTypeDic, yml=yml, countWeight=True)
    return class_weight


def loadData_transfer(yml, subId):
    # func：getData 对象内数据保存路径
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = joinPath(dataPath, 'IntraLearnData')
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    learnTrainData = []
    learnTrainLabel = []
    learnValData = []
    learnValLabel = []
    learnTestData = []
    learnTestLabel = []
    try:
        # train  只用对象内一折的数据微调
        file_train_x = learnDataPath + '/X_train_S{:03d}_fold001.pkl'.format(subId)
        file_train_y = learnDataPath + '/y_train_S{:03d}_fold001.pkl'.format(subId)
        learnTrainData = np.load(file_train_x, allow_pickle=True)
        learnTrainLabel = np.load(file_train_y, allow_pickle=True)
        # valid
        file_val_x = learnDataPath + '/X_val_S{:03d}_fold001.pkl'.format(subId)
        file_val_y = learnDataPath + '/y_val_S{:03d}_fold001.pkl'.format(subId)
        learnValData = np.load(file_val_x, allow_pickle=True)
        learnValLabel = np.load(file_val_y, allow_pickle=True)
        # test
        file_test_x = learnDataPath + '/X_test_S{:03d}_fold001.pkl'.format(subId)
        file_test_y = learnDataPath + '/y_test_S{:03d}_fold001.pkl'.format(subId)
        learnTestData = np.load(file_test_x, allow_pickle=True)
        learnTestLabel = np.load(file_test_y, allow_pickle=True)
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_train_x, file_train_y))

    # learnLabel = to_categorical(learnLabel)
    return learnTrainData, learnTrainLabel, learnValData, learnValLabel, learnTestData, learnTestLabel


def write_log(filepath='test.log', data=[], mode='w'):
    '''
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    '''
    try:
        with open(filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception('I/O error')


if __name__ == '__main__':
    # 步长为100
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None
    # imports
    save_path = '/media/oymotionai/DATA/EEG/Emotion/plot_model'
    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    # meta settings
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']
    epochs = yml["ML"]["trainEpoch"]

    # saveFolder=/data0/meya/MI/Intersub_eprime/1S_FB1-100_S100_chanSeq_20220405_FNoE
    # saveFolder = joinPath(basicFolder, folderName)
    # if not os.path.exists(saveFolder):
    #     os.makedirs(saveFolder)
    #     if 'curFolderIndex' in yml['ML']:
    #         yml['ML']['curFolderIndex'] = 0
    #     else:
    #         yml['ML'].setdefault('curFolderIndex', 0)
    #     with open(sys.argv[1], "w") as f:
    #         yaml.dump(yml, f, sort_keys=False)

    mainLog = createLog(basicFolder, folderName)
    channDic = getChannels()
    # eventTypeDic = getEvents()
    # eventTypeDic = {
    #     0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
    #     1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
    #     2: {'Name': 'foot', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
    #     3: {'Name': 'tongue', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
    # }
    # modelNet = modelNetObj(yml, eventTypeDic)
    # Step-2: training para

    doshuffle = False
    if "doShuffle" in yml['Meta']:
        # true
        doshuffle = yml['Meta']['doShuffle']
    filterBank = None
    if 'FilterBank' in yml['Meta']:
        filterBank = yml['Meta']['FilterBank']

    # MNNUdataPath = '/data0/meya/code/MIN2Net_code/experiments/datasets/MNNU_BCI/time_domain/transfer/subject_independent'
    # for directory in [MNNUdataPath]:
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    Subnum = []
    step = yml['Meta']['step']
    segmentName = yml['Meta']['segmentName']
    # Should it run in process?
    runInProcess = True
    if 'runInProcess' in yml['ML']:
        runInProcess = yml['ML']['runInProcess']
    log = logging.getLogger()
    global folderIndex
    folderNum = 3
    if 'folderNum' in yml['ML']:
        folderNum = yml['ML']['folderNum']
    folderIndex = 0

    # subid = countSubNum(yml)
    # subid = [16]
    crossIndexs = [0, 1, 2, 3, 4]
    # 将yml文件保存到文件夹
    # saveFile(saveFolder, sys.argv[1],cover=True)
    # 模型加载路径
    model_path = yml["Meta"]['basicFolder']
    eventTypeDic = {
        0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    sub_all = list(range(1, 25))  # 目标受试
    subs = list(range(1, 55))  # 所有受试
    data_exc_2 = (['subject', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity', 'f1'],)
    for sub in sub_all:
        sub_train = copy.deepcopy(subs)
        sub_train.remove(sub)
        print(sub)
        print(sub_train)
        # for sub in range(1, 3):
        # foldval_loss = []
        # for i in crossIndexs:
        # train_subs = crossValue['Train'][i]
        # valid_subs = crossValue['valid'][i]
        # test_subs = crossValue['Test'][i]
        # try:
        # 1.读取相应模型文件
        best_dir = joinPath(model_path, 'sub{0:02}'.format(sub),'best')
        pkl_file, h5_file = GetModelPath(best_dir)
        from_weight = False
        best_model_path = joinPath(best_dir, h5_file)
        if not isinstance(ML_Model, modelComFun):
            modelNet = ML_Model(yml, eventTypeDic)
        modelNet.get_model(best_model_path, from_weight)
        modelNet.model.summary()
        # 2.设置模型的可训练程度
        tra_model = M(inputs=modelNet.model.input, outputs=modelNet.model.get_layer(index=38).output)
        dense_raw = modelNet.model.layers[-2]
        weight = modelNet.model.layers[-2].get_weights()
        dense_raw.set_weights(weight)
        sigmoid = modelNet.model.layers[-1]
        x = tra_model.output
        # 新加一层共享的Dense层
        x = tf.keras.layers.Dense(128, activation='elu', name='Dense_relevant_top')(x)  # 39
        x = dense_raw(x)
        x = sigmoid(x)
        tra_model = M(inputs=tra_model.input, outputs=x)
        tra_model = reset_model(1,tra_model)  # index=[-3:-2]
        t_data = get_all_data(yml, channDic, sub)  # 需要测试的目标受试数据
        # 以分类任务训练共享Dense层
        print('================fit common dense================')
        for idx in sub_train:
            callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                      restore_best_weights=True)]
            s_data = get_all_data(yml, channDic, idx)  # 依据源域的索引加载对应的源域数据
            tra_model.compile(loss='binary_crossentropy',
                              optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999),
                              metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                                       keras.metrics.TruePositives(name='tp'),
                                       keras.metrics.FalsePositives(name='fp'),
                                       keras.metrics.TrueNegatives(name='tn'),
                                       keras.metrics.FalseNegatives(name='fn'),
                                       keras.metrics.Precision(name='precision'),
                                       keras.metrics.Recall(name='recall'),
                                       keras.metrics.AUC(name='auc')])
            tra_model.fit(x=s_data['tra'], y=s_data['tra_l'],
                      shuffle=yml['ML']['shuffle'],
                      batch_size=yml['ML']['batch_size'],
                      epochs=yml['ML']['trainEpoch'],
                      class_weight=class_weight_f(yml, s_data['tra_l']),
                      callbacks=callback)
        print('================fit mmd&mlp================')
        # 循环进行MMD训练和mlp，一共53个源域受试训练数据
        mmd_model_all = {}
        t_model_all = {}
        pred_all = []  # 用于计算差异损失的Y数据
        for idx in sub_train:
            print('================fit mmd&mlp sub{}================'.format(idx))
            # 不同的源域对应不同的Dense层模块
            tra_model = M(inputs=tra_model.input, outputs=tra_model.get_layer(index=39).output)
            x = tra_model.output
            x = tf.keras.layers.Dense(64, activation='elu', name='Dense_Spe1_{}'.format(idx))(x)  # 40
            x = tf.keras.layers.Dense(32, activation='elu', name='Dense_Spe2_{}'.format(idx))(x)  # 41
            x = tf.keras.layers.Dense(1, activation='elu', name='MLP_{}'.format(idx))(x)  # 42
            x = sigmoid(x)  # 43
            t_model = M(inputs=tra_model.input, outputs=x)
            t_model = reset_model(2, t_model)  # index=-4
            callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                      restore_best_weights=True)]
            eventTypeDic = {
                0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
            s_data = get_all_data(yml, channDic, idx)  # 依据源域的索引加载对应的源域数据
            # 和对象内相同先对新加Dense层模块进行分类任务的训练，用53个源域的数据
            t_model.compile(loss='binary_crossentropy',
                          optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999),
                          metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                                   keras.metrics.TruePositives(name='tp'),
                                   keras.metrics.FalsePositives(name='fp'),
                                   keras.metrics.TrueNegatives(name='tn'),
                                   keras.metrics.FalseNegatives(name='fn'),
                                   keras.metrics.Precision(name='precision'),
                                   keras.metrics.Recall(name='recall'),
                                   keras.metrics.AUC(name='auc')])
            t_model.fit(x=s_data['tra'], y=s_data['tra_l'],
                      shuffle=yml['ML']['shuffle'],
                      batch_size=yml['ML']['batch_size'],
                      epochs=yml['ML']['trainEpoch'],
                      class_weight=class_weight_f(yml, s_data['tra_l']),
                      callbacks=callback)
            # 根据Tensorflow的特点，将平行的53Dense层模块当做53个model，这53个model除了最后的Dense层模块，上面39层
            t_model_all['{}'.format(idx)] = t_model
            mmd_model = M(inputs=t_model.input, outputs=t_model.get_layer(index=41).output)
            mmd_model = reset_model(3, mmd_model)  # index=-2
            soure_data = mmd_model.predict(x=s_data['tra'])
            mmd_model.compile(loss=MMD,
                              optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999))
            mmd_model.fit(x=t_data['tra'], y=soure_data,
                          shuffle=yml['ML']['shuffle'],
                          batch_size=yml['ML']['batch_size'],
                          epochs=yml['ML']['trainEpoch'],
                          callbacks=callback)
            mmd_model_all['{}'.format(idx)] = mmd_model
            mmd_pred = mmd_model.predict(x=t_data['tra'])
            pred_all.append(mmd_pred)
        print('================fit difference_loss================')
        te_model_all = {}
        for i, idx in enumerate(sub_train):
            print('================fit difference_loss sub{}================'.format(idx))
            callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                      restore_best_weights=True)]
            model = reset_model(3, mmd_model_all[str(idx)])  # index=-2
            pred_tra = copy.deepcopy(pred_all)
            del pred_tra[i]
            pred_tra = np.array(pred_tra)
            pred_tra = tf.transpose(pred_tra, perm=[1, 0, 2])
            print(pred_tra.shape)
            model.compile(loss=difference_loss,
                              optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999))
            model.fit(x=t_data['tra'], y=pred_tra,
                          shuffle=yml['ML']['shuffle'],
                          batch_size=yml['ML']['batch_size'],
                          epochs=yml['ML']['trainEpoch'],
                          callbacks=callback)
            dense = t_model_all[str(idx)].layers[-2]
            x = model.output
            x = dense(x)
            x = sigmoid(x)
            te_model = M(inputs=model.input, outputs=x)
            te_model_all['{}'.format(idx)] = te_model
        print('================5-fold cross-validation================')
        loss_f = tf.keras.losses.BinaryCrossentropy()
        kf = KFold(n_splits=5)
        X = t_data['tra']
        Y = t_data['tra_l']
        m = 0
        for tra_index, te_index in kf.split(X):
            sample_2 = ['val' + str(m)]
            m = m + 1
            y_pre_all = []
            labels_test = Y[te_index]
            labels_test_1 = labels_test.reshape(len(labels_test), 1)
            labels_test_2 = labels_test.reshape(len(labels_test), 1).astype(float)
            for idx in sub_train:
                print('================cross-validation sub{}================'.format(idx))
                callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                          restore_best_weights=True)]
                new_model = M(inputs=te_model_all[str(idx)].input, outputs=te_model_all[str(idx)].output)
                new_model = reset_model(4, new_model)  # index=-5
                _, class_weight, _ = countNum(dataLabel=Y[tra_index], eventTypeDic=eventTypeDic, yml=yml,
                                              countWeight=True)
                new_model.compile(loss='binary_crossentropy',
                                  optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999),
                                  metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                                           keras.metrics.TruePositives(name='tp'),
                                           keras.metrics.FalsePositives(name='fp'),
                                           keras.metrics.TrueNegatives(name='tn'),
                                           keras.metrics.FalseNegatives(name='fn'),
                                           keras.metrics.Precision(name='precision'),
                                           keras.metrics.Recall(name='recall'),
                                           keras.metrics.AUC(name='auc')])
                new_model.fit(x=X[tra_index], y=Y[tra_index],
                              shuffle=yml['ML']['shuffle'],
                              batch_size=yml['ML']['batch_size'],
                              epochs=yml['ML']['trainEpoch'],
                              class_weight=class_weight,
                              callbacks=callback)
                y_pre = new_model.predict(X[te_index])
                y_pre = np.around(y_pre, 0).astype(int)
                y_pre_all.append(y_pre)
            y_pre_all = sum(y_pre_all)/len(y_pre_all)
            y_pre_all_1 = np.around(y_pre_all, 0).astype(int)
            y_pre_all_2 = np.around(y_pre_all, 0).astype(float)
            acc = (y_pre_all_1 == labels_test_1).sum()/len(y_pre_all_1)
            loss = loss_f(labels_test_2, y_pre_all_2)
            TN, FP, FN, TP = confusion_matrix(labels_test_1, y_pre_all_1).ravel()
            sens = recall_score(labels_test_1, y_pre_all_1, average='binary')
            f1 = f1_score(labels_test_1, y_pre_all_1, average='binary')
            preci = precision_score(labels_test_1, y_pre_all_1, average='binary')
            speci = TN / (TN + FP)
            val = {'loss': [], 'acc': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'recall': [], 'specificity': [],
                   'f1-score': []}
            val['loss'] = loss
            val['acc'] = acc
            val['tp'] = TP
            val['fp'] = FP
            val['tn'] = TN
            val['fn'] = FN
            val['recall'] = sens
            val['specificity'] = speci
            val['f1-score'] = f1
            for key in val:
                sample_2.append(val[key])
            sample_tuple_2 = ((sample_2),)
            data_exc_2 += sample_tuple_2

        writerxlsx_1(
            path=os.path.join(best_dir, 'cross_sub_1.xlsx'),
            data=data_exc_2, dataname='OpenBMI_transfer_cross')








