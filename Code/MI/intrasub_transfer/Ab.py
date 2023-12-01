import os, tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import multiprocessing


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


def pearson_positive(y_true, y_pred):
    v1 = y_pred
    v2 = tf.transpose(y_true, perm=[1, 0])
    length = v1.shape[1]
    Exy = tf.linalg.tensor_diag_part((K.dot(v1, v2))) / length
    Ex = tf.reduce_sum(v1, axis=1) / length
    Ey = tf.reduce_sum(v2, axis=0) / length
    C = Exy - Ex * Ey
    res = C / tf.math.sqrt(tf.nn.moments(v1, axes=[1])[1] * tf.nn.moments(v2, axes=[0])[1])
    loss = 1 / tf.nn.moments(res, axes=[0])[0] - 1
    return loss


def pearson_negative(y_true, y_pred):
    v1 = y_pred
    v2 = tf.transpose(y_true, perm=[1, 0])
    length = v1.shape[1]
    Exy = tf.linalg.tensor_diag_part((K.dot(v1, v2))) / length
    Ex = tf.reduce_sum(v1, axis=1) / length
    Ey = tf.reduce_sum(v2, axis=0) / length
    C = Exy - Ex * Ey
    res = C / tf.math.sqrt(tf.nn.moments(v1, axes=[1])[1] * tf.nn.moments(v2, axes=[0])[1])
    loss = tf.nn.moments(res, axes=[0])[0] + 1
    return loss


def reset_model(param, Model):
    for layer in Model.layers:
        # 屏蔽预训练模型的权重
        layer.trainable = False
    if param <= 1 and param >= 3:
        print('param error!!!')
    if param == 1:
        # Freeze all layers.
        for layer in Model.layers:
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
        for layer in Model.layers[-3:-2]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 4:
        for layer in Model.layers[-2:]:
            layer.trainable = True
        new_model = Model
        return new_model


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

def Train(yml,sub,m,X,Y,eventTypeDic,tra_index,te_index,sample_2,queue):
    print('————————————————————————sub————————————————————————————————:', os.getpid())
    model_path = yml["Meta"]['basicFolder']
    print('----------Ab{}----------'.format(m))
    best_dir = joinPath(model_path, 'sub{0:02}'.format(sub), 'best')

    pkl_file, h5_file = GetModelPath(best_dir)
    from_weight = False
    best_model_path = joinPath(best_dir, h5_file)
    if not isinstance(ML_Model, modelComFun):
        modelNet = ML_Model(yml, eventTypeDic)
    modelNet.get_model(best_model_path, from_weight)
    # 2.设置模型的可训练程度
    te_model = reset_model(4, modelNet.model)
    new_model = M(inputs=te_model.input, outputs=te_model.output)
    new_model = reset_model(4, new_model)
    callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                              restore_best_weights=True)]
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
    transfer_best_path = os.path.join(best_dir, 'best_model')
    if not os.path.exists(transfer_best_path):
        os.makedirs(transfer_best_path)
        new_model.save(os.path.join(transfer_best_path, 'transfer_Conv_1_04_elu_val_Ab.h5'.format(m)))
    else:
        new_model.save(os.path.join(transfer_best_path, 'transfer_Conv_1_04_elu_val_Ab.h5'.format(m)))\

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
        labels_test = labels_test.reshape(len(labels_test),1)
        y_pre = np.around(y_pre, 0).astype(int)
        TN, FP, FN, TP = confusion_matrix(labels_test, y_pre).ravel()
        sens = recall_score(labels_test, y_pre, average='binary')
        f1 = f1_score(labels_test, y_pre, average='binary')
        preci = precision_score(labels_test, y_pre, average='binary')
        speci = TN / (TN + FP)

    val = {'loss': [], 'acc': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'recall': [], 'specificity': [],'precision':[],
           'f1-score': []}
    val['loss'] = loss
    val['acc'] = accu
    val['tp'] = result[2]
    val['fp'] = result[3]
    val['tn'] = result[4]
    val['fn'] = result[5]
    val['recall'] = sens
    val['specificity'] = speci
    val['precision'] = preci
    val['f1-score'] = f1
    for key in val:
        sample_2.append(val[key])
    sample_tuple_2 = ((sample_2),)
    queue.put(sample_tuple_2)


    # Loss += loss
    # Accu += accu
    # tp += result[2]
    # fp += result[3]
    # tn += result[4]
    # fn += result[5]
    # Recall += sens
    # Spe += speci
    # F1 += f1
    return sample_tuple_2


if __name__ == '__main__':

    multiprocessing.set_start_method('forkserver', force=True)
    print('————————————————————————Main1————————————————————————————————:', os.getpid())

    # 步长为100
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None
    # imports
    # save_path = '/media/oymotionai/DATA/EEG/Emotion/plot_model'
    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    # meta settings
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']
    epochs = yml["ML"]["trainEpoch"]

    save_path = basicFolder

    mainLog = createLog(basicFolder, folderName)
    channDic = getChannels()

    # Step-2: training para

    doshuffle = False
    if "doShuffle" in yml['Meta']:
        # true
        doshuffle = yml['Meta']['doShuffle']
    filterBank = None
    if 'FilterBank' in yml['Meta']:
        filterBank = yml['Meta']['FilterBank']

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

    crossIndexs = [0, 1, 2, 3, 4]
    # 将yml文件保存到文件夹
    # saveFile(saveFolder, sys.argv[1],cover=True)
    # 模型加载路径
    eventTypeDic = {
        0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    sub_all = list(range(1, 55))
    sub_test = [1, 2, 3, 7, 10, 11, 24, 25, 26, 27]
    data_exc = (['subject', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity','precision','f1'],)
    data_exc_2 = (['verify', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity','precision','f1'],)
    for sub in sub_all:
        sample = ['sub' + str(sub)]
        # try:
        # 1.读取数据
        te_dataset = get_all_data(yml, channDic, sub)
        # 4.训练模型 跨对象模型的读取
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
        kf = KFold(n_splits=5)
        X = te_dataset['x']
        Y = te_dataset['y']
        test = {'loss': [], 'acc': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'recall': [], 'specificity': [],
                'f1-score': []}
        # tp, fp, fn, tn, Recall = 0, 0, 0, 0, 0
        # Loss, Accu, Spe, F1 = 0, 0, 0, 0
        m = 0
        #队列

        for tra_index, te_index in kf.split(X):
            queue = multiprocessing.Queue()
            sample_2 = ['val' + str(m)]
            if runInProcess:
                p = multiprocessing.Process(target=Train,args=(yml,sub,m,X,Y,eventTypeDic,tra_index,te_index,sample_2,queue))
                p.start()
                p.join()
                save_data = queue.get()
                data_exc_2 +=save_data
                m +=1
            else:
                save_data = Train(yml,sub,m,X,Y,eventTypeDic,tra_index,te_index,sample_2,queue)
                data_exc_2 += save_data
                m+=1

        path = '/data0/meya/MI/Intersub/CMANet/InterSub/202201202_base3DCNN_OpenBMI_Independent_400Hz_2class_4_9channel_1.0/sub{0:02}'.format(sub)
        writerxlsx_1(
            path=os.path.join(path, 'val_Ab.xlsx'),
            data=data_exc_2, dataname='OpenBMI_transfer')
    print("Done")
    #     test['loss'] = Loss / 5
    #     test['acc'] = Accu / 5
    #     test['tp'] = tp / 5
    #     test['fp'] = fp / 5
    #     test['tn'] = tn / 5
    #     test['fn'] = fn / 5
    #     test['recall'] = Recall / 5
    #     test['specificity'] = Spe / 5
    #     test['f1-score'] = F1 / 5
    #     for key in test:
    #         sample.append(test[key])
    #     sample_tuple = ((sample),)
    #     data_exc += sample_tuple
    #     # 结果保存
    # writerxlsx_1(
    #     path='/home/xumeiyan/Public/Reslut/CMANet/Try/202201201_base3DCNN_OpenBMI_400Hz_2class_4_9channel_0.0625_Combined_Attention_1Layer/transfer_Conv_1_04_elu_val_Ab.xlsx',
    #     data=data_exc, dataname='OpenBMI_transfer')









