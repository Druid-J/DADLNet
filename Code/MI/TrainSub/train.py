import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import pickle
import sys
sys.path.append('../..')
import yaml
import logging
import numpy as np
import multiprocessing
from meya.fileAction import joinPath, saveFile, saveRawDataToFile,getModelDic
from meya.appLog import createLog
from meya.MLTrain import trainCount
import tensorflow.keras.backend as K
from EEGML import ML_Model
from meya.loadModel import GetModelPath
from shutil import copy2
from os import makedirs
import nni

def Train(yml, train_subs,loadDataPath,fileName, eventTypeDic,folderIndex,modelPath,params):
    print('————————————————————————sub————————————————————————————————:', os.getpid())
    # 获取每折的数据
    print("Strat Subject{}_fold{} data load".format(train_subs, folderIndex))
    print('Loda Data from:{}'.format(loadDataPath))
    num_class = 2
    TrainData, Tralabel = load_train_Data(yml,train_subs, loadDataPath, folderIndex)
    ValidData, Vallabel = load_val_data(yml,train_subs, loadDataPath, folderIndex)

    print(
        "Check dimension of training data {},training label {},val data {},val label{} ".format(
            TrainData.shape,Tralabel.shape, ValidData.shape,Vallabel.shape))
    if (TrainData is not None and len(Tralabel) > 0 and ValidData is not None and len(Vallabel)>0):
        subFolderPath = joinPath(yml["Meta"]['basicFolder'], yml["Meta"]['folderName'])
        validData = (ValidData,Vallabel)
        trainCount(subFolderPath,fileName, folderIndex, yml, TrainData, Tralabel,train_subs,validData,ML_Model=ML_Model,
                   eventTypeDic=eventTypeDic, modelPath=modelPath,params=params)
    print("Compelete folder %d Train!" % folderIndex)

def load_train_Data(yml,sub,load_path,folderIndex):
    try:
        file_x = load_path + '/X_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x,allow_pickle=True)
        a = X[0:10,:,:,:]
        X_mean = np.mean(np.absolute(X))
        print('Raw meam:{}'.format(X_mean))
        y = np.load(file_y, allow_pickle=True)
        # labels = np.argmax(y, axis=1)
        # if yml['Meta']['ClassNum']==2:
        #     y = y-1
        print('Train_X shape{},Train_y shape{}'.format(X.shape,y.shape))
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X,y
def load_val_data(yml,sub,load_path,folderIndex):
    try:
        file_x = load_path + '/X_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x, allow_pickle=True)
        X_mean = np.mean(np.absolute(X))
        print('Raw meam:{}'.format(X_mean))
        y = np.load(file_y, allow_pickle=True)
        # labels = np.argmax(y,axis=1)
        # if yml['Meta']['ClassNum'] == 2:
        #     y = y - 1
        print('Val_X shape{},Val_y shape{}'.format(X.shape, y.shape))
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X, y



def read_pickle(work_path):
    with open(work_path,'rb') as f:
        try:
            history_data = pickle.load(f)
        except Exception as err:
            print(err)
            log.error(err)
            raise err
    folde_val_loss = min(history_data['val_loss'])
    folde_val_mean_loss =np.mean(history_data['val_loss'])
    print('foldval_loss_mean:',folde_val_mean_loss)
    return folde_val_loss,folde_val_mean_loss

def __save_data_with_valset(save_path, NAME, X_train, y_train, X_val, y_val, X_test, y_test):
    saveRawDataToFile(save_path + '/X_train_' + NAME + '.npy', X_train)
    saveRawDataToFile(save_path + '/X_val_' + NAME + '.npy', X_val)
    saveRawDataToFile(save_path + '/X_test_' + NAME + '.npy', X_test)
    saveRawDataToFile(save_path + '/y_train_' + NAME + '.npy', y_train)
    saveRawDataToFile(save_path + '/y_val_' + NAME + '.npy', y_val)
    saveRawDataToFile(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    print('————————————————————————Main1————————————————————————————————:', os.getpid())


    # 步长为100
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1],encoding='UTF-8'), Loader=yaml.FullLoader)
    getLayer = None


    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    # meta settings
    BasePath = yml['Meta']['initDataFolder']
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']


    saveFolder = joinPath(basicFolder, folderName)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
        if 'curFolderIndex' in yml['ML']:
            yml['ML']['curFolderIndex'] = 0
        else:
            yml['ML'].setdefault('curFolderIndex', 0)
        with open(sys.argv[1], "w") as f:
            yaml.dump(yml, f, sort_keys=False)

    mainLog = createLog(basicFolder, folderName)
    channDic = getChannels()


    doshuffle = False
    if "doShuffle" in yml['Meta']:
        # true
        doshuffle = yml['Meta']['doShuffle']
    filterBank = None
    if 'FilterBank' in yml['Meta']:
        filterBank = yml['Meta']['FilterBank']
    #使用MIN2Net生成数据
    Dataset = yml['Meta']['Datasets']  #Dataset name:ex[BCIC2a/OpenBMI]
    Datatype = 'time_domain'
    num_class = yml['Meta']['ClassNum']    # number of classes:ex[2,3,4]

    if Dataset =='OpenBMI':
        num_subject = 54
    #这部分只是为了计算权重
        if num_class ==2:
            eventTypeDic = {
                0:{'Name':'right','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                }
        if num_class == 3:
            eventTypeDic = {
                0:{'Name':'rest','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2:{'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                }
    elif Dataset =="BCIC2a":
        num_subject = 9
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
        if num_class == 4:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'foot', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                3: {'Name': 'tongue', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                }
    elif Dataset == "MNNUData":
        # num_subject = [1, 2, 3, 4]
        num_subject = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,19,20,22,23,24,26,27,29,31,32]
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
        if num_class == 3:
            eventTypeDic = {
                0: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset =="Kaya2018":
        num_subject = 11
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    #min2net数据路径

    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    pick_smp_freq = yml['Meta']['downSampling']
    Channel_format = yml['Meta']['Channel']
    minFreq = yml['Meta']['minFreq']
    maxFreq = yml['Meta']['maxFreq']
    step = yml['Meta']['step']
    Min2DataPath = BasePath + '/Traindata/{}/{}_class/{}Hz_{}chan_{}_{}Hz_{}_ref'.format(TrainType, num_class,
                                                                                  pick_smp_freq, Channel_format,
                                                                                  minFreq, maxFreq, step)

    # Min2DataPath = '/home/xumeiyan/Public/Data/MI/BCICIV_2a/Traindata/dependent/2_class/400Hz_20chan_0.5_100Hz_0.06'
    Subnum = []
    step = yml['Meta']['step']
    segmentName = yml['Meta']['segmentName']
    # Should it run in process?
    runInProcess = True
    if 'runInProcess' in yml['ML']:
        runInProcess = yml['ML']['runInProcess']
    log = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled=True
    global folderIndex
    folderNum = 3
    if 'folderNum' in yml['ML']:
        folderNum = yml['ML']['folderNum']
    folderIndex = 0

    #NNI 初始化
    params = {'features_1':64,'features_2':64,'features_3':64,'features_4':128,"ratio":16,'drop':0.5,'l2':0.001,'batch':32}
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    #
    fileName = 'feature{}-{}-{}-{}_ratio{}_drop{}_l2{}_batch{}'.format(params['features_1'],params['features_2'],
                                                                       params['features_3'],params['features_4'],
                                                                       params['ratio'],params['drop'],params['l2'],
                                                                       params['batch'])

    sub_loss = []
    saveFile(saveFolder, sys.argv[1], cover=True)
    for person in range(1,10):
        foldval_loss = []
        foldval_loss_mean = []
        for i in range(0,5):
            try:
                folderIndex = i + 1
                #创建folder文件夹
                curModelFolder = '%s/%s/%s/folder_%d' % (saveFolder,fileName, 'sub{0:02d}'.format(person), folderIndex)
                if not os.path.exists(curModelFolder):
                    os.makedirs(curModelFolder)
                doTrain = True
                modelPath = None
                h5TimeList, modelPath = getModelDic(curModelFolder)
                if len(h5TimeList) > 0:
                    if 'loadModel' not in yml['ML'] or not yml['ML']['loadModel']:
                        doTrain = False
                print(doTrain)
                if doTrain:
                    print('folder Index: %d' % folderIndex)
                    if runInProcess:
                        p = multiprocessing.Process(target=Train, args=(
                        yml, person,Min2DataPath,fileName, eventTypeDic,folderIndex, modelPath,params))
                        p.start()
                        p.join()
                    else:
                        #getData_chanSeq as getData
                        Train(yml, person, Min2DataPath,fileName,eventTypeDic, folderIndex, modelPath,params)
            except Exception as err:
                print(err)
                log.error(err)
                raise err
            # 读取PKL文件
            subFolderPath = joinPath(yml["Meta"]['basicFolder'], yml["Meta"]['folderName'])
            save_dir = "%s/%s/%s/folder_%d" % (subFolderPath,fileName,'sub{0:02d}'.format(person),
                                            folderIndex)
            Pklpath, H5path = GetModelPath(save_dir)
            print(save_dir, Pklpath)
            history_file = joinPath(save_dir, Pklpath)
            #获取每折中最小的loss值 loss的均值
            fval_loss,fval_mean_loss = read_pickle(history_file)
            foldval_loss.append(fval_loss)
            foldval_loss_mean.append(fval_mean_loss)

        #保存5折的平均loss 也就是单个被试五折的平均loss
        sub_loss.append(np.mean(foldval_loss_mean))
        # 找出五折中最小的val_loss的折数
        fold_index = foldval_loss.index(min(foldval_loss))
        minlossfold = fold_index + 1
        best_dir = joinPath(subFolderPath,fileName, 'sub{0:02d}/best'.format(person))
        # if not os.path.exists(best_dir):
        #     os.makedirs(best_dir)
        makedirs(best_dir, exist_ok=True)
        with open(joinPath(best_dir, "fold_bestcv.txt"), 'a') as f:
            f.write("sub{}, fold{}\n".format(person, minlossfold))
        model_dir = "%s/%s/%s/folder_%d" % (subFolderPath,fileName, 'sub{0:02d}'.format(person),
                                         minlossfold)
        BestPkl, BestH5 = GetModelPath(model_dir)
        # copy2(joinPath(model_dir, 'modle-e{}-f{}.h5'.format(epochs, minlossfold)),
        #         joinPath(best_dir, 'model-sub{0:02}.h5'.format(sub)))
        copy2(joinPath(model_dir, BestH5),
              joinPath(best_dir, BestH5))
        copy2(joinPath(model_dir, BestPkl),
              joinPath(best_dir, BestPkl))

    all_loss = np.mean(sub_loss)
    nni.report_final_result(float(all_loss))

