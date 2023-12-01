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
from tensorflow.keras.callbacks import EarlyStopping
from EEGML import ML_Model
import os
import tensorflow.keras as keras

from meya.loadModel import GetModelPath
from meya.basicModel import modelComFun

from MI.DA.load_data_2 import get_data_bci_stride
from meya.MLTrain import countNum
from tensorflow.keras.optimizers import Nadam
import tensorflow.keras.backend as K
from sklearn.metrics import recall_score, confusion_matrix, f1_score, precision_score
from tensorflow.keras.models import Model as M
from MI.DA.load_data_2 import writerxlsx_1



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
        for layer in Model.layers[-2:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 2:
        for layer in Model.layers[-5:]:
            layer.trainable = True
        new_model = Model
        return new_model
    if param == 3:
        for layer in Model.layers[-4:]:
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





if __name__ == '__main__':
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None
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
    sub_test = [1, 2, 3, 7, 10, 11, 24, 25, 26, 27]
    data_exc = (['subject', 'loss', 'acc', 'tp', 'fp', 'tn', 'fn', 'recall', 'specificity','precision', 'f1'],)
    data_exc_2 = (['subject', 'loss', 'acc', 'tp', 'tn', 'fp', 'fn', 'recall', 'specificity','precision', 'f1'],)
    for sub in sub_all:
        sample = ['sub' + str(sub)]
        # load model
        best_dir = joinPath(model_path, 'sub{0:02}'.format(sub),'best')
        pkl_file, h5_file = GetModelPath(best_dir)
        from_weight = False
        best_model_path = joinPath(best_dir, h5_file)
        if not isinstance(ML_Model, modelComFun):
            modelNet = ML_Model(yml, eventTypeDic)
        modelNet.get_model(best_model_path, from_weight)
        modelNet.model.summary()
        # Domain adaptation Model
        tra_model = M(inputs=modelNet.model.input, outputs=modelNet.model.get_layer(index=38).output)
        sigmoid = modelNet.model.layers[-1]
        x = tra_model.output
        x = tf.keras.layers.Dense(16, activation='elu', name='Dense_relevant_1')(x) #39
        x = tf.keras.layers.Dense(16, activation='elu', name='Dense_relevant_2')(x)  #40
        x = tf.keras.layers.Dense(16, activation='elu', name='Dense_relevant_3')(x)  #41
        x = tf.keras.layers.Dense(1, activation='elu', name='MLP')(x)  #42
        x = sigmoid(x)  #43
        tra_model = M(inputs=tra_model.input, outputs=x)
        tra_model = reset_model(2, tra_model)  # index=-5
        all_data,te_data = get_data_bci_stride(yml, sub, 5)
        callback = [EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                                  restore_best_weights=True)]
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
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
        tra_model.fit(x=all_data['MMD_tra'], y=all_data['MMD_tra_l'],
                     shuffle=yml['ML']['shuffle'],
                     batch_size=yml['ML']['batch_size'],
                     epochs=yml['ML']['trainEpoch'],
                     class_weight=class_weight_f(yml, all_data['MMD_tra_l']),
                     callbacks=callback)
        t_model = M(inputs=tra_model.input, outputs=tra_model.get_layer(index=41).output)
        t_model = reset_model(4, t_model) #-3
        soure_data = t_model.predict(x=all_data['MMD_tra'])
        t_model.compile(loss=MMD,
                          optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999))
        print('fit MMD')
        t_model.fit(x=te_data, y=soure_data,
                      shuffle=yml['ML']['shuffle'],
                      batch_size=yml['ML']['batch_size'],
                      epochs=yml['ML']['trainEpoch'],
                      callbacks=callback)
        #test
        dense = tra_model.layers[-2]
        weight = tra_model.layers[-2].get_weights()
        dense.set_weights(weight)
        sigmoid = modelNet.model.layers[-1]
        y = t_model.output
        y = dense(y)
        y = sigmoid(y)
        te_model = M(inputs=t_model.input, outputs=y)
        m = 0
        for fold in range(5):
            print('----------{}----------'.format(m+1))
            tra_x = all_data['fold_{}_tra'.format(fold+1)]
            tra_y = all_data['fold_{}_tra_l'.format(fold+1)]
            te_x = all_data['fold_{}_test'.format(fold+1)]
            te_y = all_data['fold_{}_test_l'.format(fold + 1)]
            sample_2 = ['val' + str(m+1)]
            new_model = M(inputs=te_model.input, outputs=te_model.output)
            new_model = reset_model(2, new_model)
            m = m + 1
            _, class_weight, _ = countNum(dataLabel=tra_y, eventTypeDic=eventTypeDic, yml=yml, countWeight=True)
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
            new_model.fit(x=tra_x, y=tra_y,
                          shuffle=yml['ML']['shuffle'],
                          batch_size=yml['ML']['batch_size'],
                          epochs=yml['ML']['trainEpoch'],
                          class_weight=class_weight,
                          callbacks=callback)
            result = new_model.evaluate(te_x, te_y)
            transfer_best_path = os.path.join(best_dir, 'best_model')
            if not os.path.exists(transfer_best_path):
                os.makedirs(transfer_best_path)
                new_model.save(os.path.join(transfer_best_path, 'transfer_Dense_2_400Hz_mmd_BCI_NEW.h5'))
            else:
                new_model.save(os.path.join(transfer_best_path, 'transfer_Dense_2_400Hz_mmd_BCI_NEW.h5'))
            loss, accu = result[0], result[1]
            labels_test = te_y
            y_pre = new_model.predict(te_x)
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
                sens = recall_score(y_t, y_p, labels=label, average='macro')
                f1 = f1_score(y_t, y_p, labels=label, average='macro')
                preci = precision_score(y_t, y_p, labels=label, average='macro')

                cm = confusion_matrix(y_t, y_p)
                FP = cm.sum(axis=0) - np.diag(cm)
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                classnum = cm.shape[0]
                sensall = []
                for i in range(classnum):
                    sensall.append(TP[i] / (TP[i] + FN[i]))
                sens_method = sum(sensall) / classnum

                speciall = []
                for i in range(classnum):
                    speciall.append(TN[i] / (TN[i] + FP[i]))
                speci_method = sum(speciall) / classnum

                precisionall = []
                for i in range(classnum):
                    precisionall.append(TP[i] / (TP[i] + FP[i]))
                preci_method = sum(precisionall) / classnum

                class_f1 = []
                for i in range(classnum):
                    class_f1.append((2 * sensall[i] * precisionall[i]) / (sensall[i] + precisionall[i]))
                f1_score_method = sum(class_f1) / classnum

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
            val = {'loss': [], 'acc': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'recall': [], 'specificity': [],'preci':[],
                   'f1-score': []}
            val['loss'] = loss
            val['acc'] = accu
            val['tp'] = result[2]
            val['fp'] = result[3]
            val['tn'] = result[4]
            val['fn'] = result[5]
            val['recall'] = sens
            val['specificity'] = speci
            val['preci'] = preci
            val['f1-score'] = f1
            for key in val:
                sample_2.append(val[key])
            sample_tuple_2 = ((sample_2),)
            data_exc_2 += sample_tuple_2

        writerxlsx_1(
            path=os.path.join(best_dir, 'result_BCI_intra.xlsx'),
            data=data_exc_2, dataname='BCI2A_transfer')









