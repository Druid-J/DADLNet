import os, tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from MI.intersub_transfer.load_data_2 import get_all_data
from MI.intersub_transfer.load_data_new import get_data
from meya.MLTrain import countNum
from tensorflow.keras.optimizers import Nadam
import tensorflow.keras.backend as K
from sklearn.metrics import recall_score, confusion_matrix, f1_score, precision_score
from meya.effect import calculate_performance, calculate_performance_dual
from tensorflow.keras.models import Model as M
# from MI.intersub_transfer.person_loss import Dense_with_person
from MI.intersub_transfer.tool import writerxlsx_1


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
    n_s = 128 if n_s is None else n_s
    n_t = tf.shape(target)[0]
    n_t = 128 if n_t is None else n_t
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
        for layer in Model.layers[-2:]:
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
        for layer in Model.layers[-3:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 4:
        for layer in Model.layers[-3:]:
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
    sub_all = list(range(1, 55))
    sub_test = [1, 2, 3, 7, 10, 11, 24, 25, 26, 27]
    data_exc = (['subject', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity', 'f1'],)
    data_exc_2 = (['subject', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity', 'f1'],)
    for sub in sub_all:
        sample = ['sub' + str(sub)]
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
        x = tra_model.output
        x = tf.keras.layers.Dense(128, activation='elu', name='Dense_relevant_1')(x)
        x = tf.keras.layers.Dense(32, activation='elu', name='Dense_relevant_3')(x)
        tra_model = M(inputs=tra_model.input, outputs=x)
        tra_model = reset_model(1, tra_model)  # index=-4
        # plot_model(tra_model, show_shapes=True, to_file='/home/xumeiyan/Public/model_image/tra_model')
        # 3.读取数据  对象内数据读取方式
        # data_tra, label_tra, data_val, label_val, data_te, label_te = loadData_transfer(yml, sub)
        # data_tra, label_tra, data_te, label_te = get_data(yml, sub)
        te_dataset, train_data, test_data = get_all_data(yml, channDic, sub)
        soure_data_1 = tra_model.predict(x=train_data['tra1'])
        soure_data_2 = tra_model.predict(x=train_data['tra2'])
        # 4.训练模型 跨对象模型的读取
        callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                  restore_best_weights=True)]
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
        # soure_data_1 = t_model.predict(x=train_data['tra1'])
        # soure_data_2 = t_model.predict(x=train_data['tra2'])

        '''
        validData = (data_val,label_val)
        if validData is not None:
            his = modelNet.model.fit(
                x=np.array(data_tra), y=np.array(label_tra),
                shuffle = yml['ML']['shuffle'],
                batch_size = yml['ML']['batch_size'],
                epochs = epochs,
                validation_data = validData,
                steps_per_epoch =yml['ML']['steps_per_epoch'],
                callbacks = [callback],
            )
        '''
        tra_model.compile(loss=pearson,
                          optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999))
        print('fit mmd te1')
        tra_model.fit(x=test_data['te1'], y=soure_data_2,
                      shuffle=yml['ML']['shuffle'],
                      batch_size=yml['ML']['batch_size'],
                      epochs=yml['ML']['trainEpoch'],
                      callbacks=callback)
        print('fit mmd te2')
        tra_model.fit(x=test_data['te2'], y=soure_data_1,
                      shuffle=yml['ML']['shuffle'],
                      batch_size=yml['ML']['batch_size'],
                      epochs=yml['ML']['trainEpoch'],
                      callbacks=callback)

        # tra_model.compile(loss=pearson_negative,
        #                   optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999))
        # print('fit positive tra_0 te_1')
        # tra_model.fit(x=te_data['0'], y=soure_data_1,
        #               shuffle=yml['ML']['shuffle'],
        #               batch_size=yml['ML']['batch_size'],
        #               epochs=yml['ML']['trainEpoch'],
        #               callbacks=callback)
        sigmoid = modelNet.model.layers[-1]
        y = tra_model.output
        y = tf.keras.layers.Dense(1, activation='elu', name='MLP')(y)
        y = sigmoid(y)
        tra_model = M(inputs=tra_model.input, outputs=y)
        tra_model = reset_model(2, tra_model)
        # te_model.summary()
        # plot_model(tra_model, show_shapes=True, to_file='/home/xumeiyan/Public/model_image/te_model')
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
        tra_model.fit(x=train_data['tra1'], y=train_data['tra1_label'],
                     shuffle=yml['ML']['shuffle'],
                     batch_size=yml['ML']['batch_size'],
                     epochs=yml['ML']['trainEpoch'],
                     class_weight=class_weight_f(yml, train_data['tra1_label']),
                     callbacks=callback)
        tra_model.fit(x=train_data['tra2'], y=train_data['tra2_label'],
                     shuffle=yml['ML']['shuffle'],
                     batch_size=yml['ML']['batch_size'],
                     epochs=yml['ML']['trainEpoch'],
                     class_weight=class_weight_f(yml, train_data['tra2_label']),
                     callbacks=callback)
        # MLP的微调
        kf = KFold(n_splits=5)
        X = te_dataset['x']
        Y = te_dataset['y']
        print(Y.shape)
        test = {'loss': [], 'acc': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'recall': [], 'specificity': [],
                'f1-score': []}
        tp, fp, fn, tn, Recall = 0, 0, 0, 0, 0
        Loss, Accu, Spe, F1 = 0, 0, 0, 0
        m = 0
        for tra_index, te_index in kf.split(X):
            print('----------{}----------'.format(m))
            sample_2 = ['val' + str(m)]
            new_model = M(inputs=tra_model.input, outputs=tra_model.output)
            new_model = reset_model(2, new_model)
            new_model.summary()
            m = m + 1
            _, class_weight, _ = countNum(dataLabel=Y[tra_index], eventTypeDic=eventTypeDic, yml=yml, countWeight=True)
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
            result = new_model.evaluate(X[te_index], Y[te_index])
            transfer_best_path = os.path.join(best_dir, 'best_model_400Hz')
            if not os.path.exists(transfer_best_path):
                os.makedirs(transfer_best_path)
                tra_model.save(os.path.join(transfer_best_path, 'transfer_Dense_3_400Hz.h5'))
            else:
                tra_model.save(os.path.join(transfer_best_path, 'transfer_Dense_3_400Hz.h5'))
            loss, accu = result[0], result[1]
            labels_test = Y[te_index]
            y_pre = new_model.predict(X[te_index])
            # 获取model.evalute中的评价指标
            if yml['ML']['loss'] == "categorical_crossentropy":
                class_num = yml['Meta']['ClassNum']
                if class_num == 2:
                    label = [0, 1]
                elif class_num == 3:
                    label = [0, 1, 2]
                elif class_num == 4:
                    label = [0, 1, 2, 3]

                y_t = K.argmax(labels_test, axis=-1)
                y_p = K.argmax(y_pre, axis=-1)
                # 10月26日 使用sklearn.metrics的计算指标
                sens = recall_score(y_t, y_p, labels=label, average='macro')
                f1 = f1_score(y_t, y_p, labels=label, average='macro')
                preci = precision_score(y_t, y_p, labels=label, average='macro')

                #######计算公式
                cm = confusion_matrix(y_t, y_p)
                FP = cm.sum(axis=0) - np.diag(cm)
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                # A_specificity = TN[0]/(TN[0]+FP[0])
                classnum = cm.shape[0]
                # #计算敏感度
                sensall = []
                for i in range(classnum):
                    sensall.append(TP[i] / (TP[i] + FN[i]))
                sens_method = sum(sensall) / classnum

                # 计算特异性
                speciall = []
                for i in range(classnum):
                    speciall.append(TN[i] / (TN[i] + FP[i]))
                speci_method = sum(speciall) / classnum

                # 计算precision
                precisionall = []
                for i in range(classnum):
                    precisionall.append(TP[i] / (TP[i] + FP[i]))
                preci_method = sum(precisionall) / classnum

                # 计算f1_socre
                class_f1 = []
                for i in range(classnum):
                    class_f1.append((2 * sensall[i] * precisionall[i]) / (sensall[i] + precisionall[i]))
                f1_score_method = sum(class_f1) / classnum

                # 因为sklearn.metrics缺少 specificity指标 所有使用公式计算
                speci = 0
                if sens == sens_method and preci == preci_method and f1 == f1_score_method:
                    speci = speci_method

            elif yml['ML']['loss'] == "binary_crossentropy":

                labels_test = labels_test.reshape(len(labels_test), 1)
                y_pre = np.around(y_pre, 0).astype(int)
                TN, FP, FN, TP = confusion_matrix(labels_test, y_pre).ravel()
                sens = recall_score(labels_test, y_pre, average='binary')
                f1 = f1_score(labels_test, y_pre, average='binary')
                preci = precision_score(labels_test, y_pre, average='binary')
                speci = TN / (TN + FP)
            val = {'loss': [], 'acc': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'recall': [], 'specificity': [],
                   'f1-score': []}
            val['loss'] = loss
            val['acc'] = accu
            val['tp'] = result[2]
            val['fp'] = result[3]
            val['tn'] = result[4]
            val['fn'] = result[5]
            val['recall'] = sens
            val['specificity'] = speci
            val['f1-score'] = f1
            for key in test:
                sample_2.append(val[key])
            sample_tuple_2 = ((sample_2),)
            data_exc_2 += sample_tuple_2

            Loss += loss
            Accu += accu
            tp += result[2]
            fp += result[3]
            tn += result[4]
            fn += result[5]
            Recall += sens
            Spe += speci
            F1 += f1

        writerxlsx_1(
            path=os.path.join(best_dir, 'val_Dense_400Hz_3.xlsx'),
            data=data_exc_2, dataname='OpenBMI_transfer')
        test['loss'] = Loss / 5
        test['acc'] = Accu / 5
        test['tp'] = tp / 5
        test['fp'] = fp / 5
        test['tn'] = tn / 5
        test['fn'] = fn / 5
        test['recall'] = Recall / 5
        test['specificity'] = Spe / 5
        test['f1-score'] = F1 / 5
        for key in test:
            sample.append(test[key])
        sample_tuple = ((sample),)
        data_exc += sample_tuple
        # 结果保存
    writerxlsx_1(
        path=os.path.join('/home/xumeiyan/Public/Reslut/CMANet/Try/202201201_base3DCNN_OpenBMI_400Hz_2class_4_9channel_0.0625_Combined_Attention_1Layer/result',
                          'transfer_Dense_3_val_400Hz.xlsx'),
        data=data_exc, dataname='OpenBMI_transfer')









