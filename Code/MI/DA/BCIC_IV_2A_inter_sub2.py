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
from meya.fileAction import joinPath
from meya.appLog import createLog
from tensorflow.keras.callbacks import EarlyStopping
from EEGML import ML_Model
import os
from sklearn.model_selection import KFold
import tensorflow.keras as keras

from meya.loadModel import GetModelPath
from meya.basicModel import modelComFun
from MI.DA.load_data_2 import get_all_data_Bci2a_inter,get_all_data_OpenBMI_inter
from meya.MLTrain import countNum
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import recall_score, confusion_matrix, f1_score, precision_score
from tensorflow.keras.models import Model as M
from MI.DA.load_data_2 import writerxlsx_1
import copy



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
        layer.trainable = False
    if param <= 1 and param >= 3:
        print('param error!!!')
    if param == 1:
        # Freeze all layers.
        for layer in Model.layers[-3:-2]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 2:
        for layer in Model.layers[-5:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 3:
        for layer in Model.layers[-3:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 4:
        for layer in Model.layers[-5:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 5:
        for layer in Model.layers[-1:]:
            layer.trainable = True
        new_model = Model
        return new_model


def class_weight_f(yml, y):
    _, class_weight, _ = countNum(dataLabel=y, eventTypeDic=eventTypeDic, yml=yml, countWeight=True)
    return class_weight




if __name__ == '__main__':
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None
    # imports
    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    # meta settings
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']
    epochs = yml["ML"]["trainEpoch"]


    mainLog = createLog(basicFolder, folderName)
    channDic = getChannels()


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
    model_path = yml["Meta"]['basicFolder']
    eventTypeDic = {
        0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    }
    sub_all = list(range(1, 10))
    subs = list(range(1,10))
    data_exc_2 = (['subject', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity', 'f1'],)
    for sub in sub_all:
        sub_train = copy.deepcopy(subs)
        sub_train.remove(sub)
        # load model
        best_dir = joinPath(model_path, 'sub{0:02}'.format(sub),'best')
        pkl_file, h5_file = GetModelPath(best_dir)
        from_weight = False
        best_model_path = joinPath(best_dir, h5_file)
        if not isinstance(ML_Model, modelComFun):
            modelNet = ML_Model(yml, eventTypeDic)
        modelNet.get_model(best_model_path, from_weight)
        # Domain Adaptation Model
        tra_model = M(inputs=modelNet.model.input, outputs=modelNet.model.get_layer(index=38).output)
        sigmoid = modelNet.model.layers[-1]
        x = tra_model.output
        x = tf.keras.layers.Dense(16, activation='elu', name='Dense_relevant_top_1')(x)  # 39
        x = tf.keras.layers.Dense(16, activation='elu', name='Dense_relevant_top_2')(x)
        x = tf.keras.layers.Dense(16, activation='elu', name='Dense_relevant_top_3')(x)
        x = tf.keras.layers.Dense(1, activation='elu', name='MLP')(x)
        # x = dense_raw(x)
        x = sigmoid(x)
        tra_model = M(inputs=tra_model.input, outputs=x)
        tra_model = reset_model(2, tra_model)  # index=[-3:-2]
        print("======= Show new Dense ======== \n")
        tra_model.summary()
        t_data= get_all_data_Bci2a_inter(yml, channDic, sub)
        print('================fit common dense================\n')
        t_model_all = {}
        for idx in sub_train:
            print('====Start Sub{} common dense train ===\n'.format(idx))
            callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                      restore_best_weights=True)]
            s_data= get_all_data_Bci2a_inter(yml, channDic, idx)
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
            t_model_all['{}'.format(idx)] = tra_model
        print('================fit mmd&mlp===============\n')
        mmd_model_all = {}
        te_model_all = {}
        for idx in sub_train:
            print('================Start sub{} mmd training================\n'.format(idx))
            callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                      restore_best_weights=True)]
            eventTypeDic = {
                0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
            s_data = get_all_data_Bci2a_inter(yml, channDic, idx)
            mmd_model = M(inputs=t_model_all[str(idx)].input, outputs=t_model_all[str(idx)].get_layer(index=41).output)
            mmd_model = reset_model(3, mmd_model)  # index=-3
            print("Show Phase 2 (MMD) model\n")
            mmd_model.summary()
            soure_data = mmd_model.predict(x=s_data['tra'])
            mmd_model.compile(loss=MMD,
                              optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999))
            mmd_model.fit(x=t_data['tra'], y=soure_data,
                          shuffle=yml['ML']['shuffle'],
                          batch_size=yml['ML']['batch_size'],
                          epochs=yml['ML']['trainEpoch'],
                          callbacks=callback)
            mmd_model_all['{}'.format(idx)] = mmd_model
            dense = t_model_all[str(idx)].layers[-2]
            x = mmd_model.output
            x = dense(x)
            x = sigmoid(x)
            te_model = M(inputs=mmd_model.input, outputs=x)
            te_model_all['{}'.format(idx)] = te_model
        print('================5-fold cross-validation================\n')
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
                print('================cross-validation sub{}================\n'.format(idx))
                callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                          restore_best_weights=True)]
                new_model = M(inputs=te_model_all[str(idx)].input, outputs=te_model_all[str(idx)].output)
                new_model = reset_model(4, new_model)  # index=-5
                print("==== Test model =====\n")
                new_model.summary()
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
            path=os.path.join(best_dir, 'result_BCI2a_inter2.xlsx'),
            data=data_exc_2, dataname='BCI2a_transfer_cross')
