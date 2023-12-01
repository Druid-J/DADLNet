import os
import sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
sys.path.append('../..')
import yaml
import logging
from EEGML import ML_Model
from meya.appLog import createLog
import datetime as dt
import glob
from meya.fileAction import joinPath
from meya.MLTest import test2
import tensorflow as tf
import multiprocessing
import nni

def getTestData(yml,test_sub,index):
    edf_list = glob.glob(joinPath(yml['Meta']['initDataFolder'], 'chb{0:02d}'.format(sub), '*.edf'))
    edf_list.sort()
    edfFile=os.path.basename(edf_list[index])
    edfData, edfLabel = loadData(yml, test_sub, channDic, eventTypeDic, getData, isTrain=False,CurEdf=edfFile)
    if 'addFakeClass' in yml['ML'] and yml['ML']['addFakeClass']:
        edfLabel=[l+1 for l in edfLabel]
    if edfData is not None and len(edfLabel) > 0:
        if 'testTitle' in yml['ML']:
            testTitle=yml['ML']['testTitle']
            accuIndex=[i for i in range(len(testTitle)) if testTitle[i]=='acc'][0]
            lossIndex=[i for i in range(len(testTitle)) if testTitle[i]=='loss'][0]
            return {"Data": edfData, "Label": edfLabel, 'AccuIndex': accuIndex, 'LossIndex': lossIndex}
        else:
            return {"Data": edfData, "Label": edfLabel}
    return None




if __name__ == '__main__':

    #设置多进程启动方式
    multiprocessing.set_start_method('forkserver',force=True)
    print('————————————————————————Main1————————————————————————————————:', os.getpid())
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]),Loader=yaml.FullLoader)
    getLayer = None
    getData =None

    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)
    # meta settings
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']

    saveFolder = "%s/%s" % (basicFolder, folderName)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
        if 'curFolderIndex' in yml['ML']:
            yml['ML']['curFolderIndex']=0
        else:
            yml['ML'].setdefault('curFolderIndex', 0)
        with open(sys.argv[1], "w") as f:
            yaml.dump(yml, f,sort_keys=False)

    mainLog=createLog(basicFolder,folderName)
    channDic = getChannels()
    #设置测试数据参数
    Dataset = yml['Meta']['Datasets']  # Dataset name:ex[BCIC2a/OpenBMI]
    Datatype = 'time_domain'
    num_class = yml['Meta']['ClassNum']  # number of classes:ex[2,3,4]

    if Dataset == 'OpenBMI':
        num_subject = 54
        # 这部分只是为了计算权重
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
        if num_class == 3:
            eventTypeDic = {
                0: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset == "BCIC2a":
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
    # min2net数据路径

    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    pick_smp_freq = yml['Meta']['downSampling']
    Channel_format = yml['Meta']['Channel']
    minFreq = yml['Meta']['minFreq']
    maxFreq = yml['Meta']['maxFreq']
    step = yml['Meta']['step']
    Min2DataPath = BasePath + '/Traindata/{}/{}_class/{}Hz_{}chan_{}_{}Hz_{}_ref'.format(TrainType, num_class,pick_smp_freq,
                                                                                  Channel_format,minFreq,maxFreq,step)
    log = logging.getLogger()
    subsample = int(yml['Meta']['subsample']) + 1

    # NNI 初始化
    params = {'features_1': 64, 'features_2': 64, 'features_3': 64, 'features_4': 128, "ratio": 16, 'drop': 0.5,
              'l2': 0.001, 'batch': 32}
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    #
    fileName = 'feature{}-{}-{}-{}_ratio{}_drop{}_l2{}_batch{}'.format(params['features_1'], params['features_2'],
                                                                       params['features_3'], params['features_4'],
                                                                       params['ratio'], params['drop'], params['l2'],
                                                                       params['batch'])

    runInProcess = True
    if 'runInProcess' in yml['ML']:
        runInProcess = yml['ML']['runInProcess']
    timeStr=dt.datetime.now().strftime('%Y%m%d%H%M')

    for sub in range(1,10):
        excel_sub=sub
        try:
            #5
            folderNum = yml['ML']['folderNum']
            for folder in range(1,6):
                test_sub=[]
                curModelFolder = '%s/%s/%s/folder_%d' % (saveFolder,fileName,'sub{0:02d}'.format(sub), folder)
                if not os.path.exists(curModelFolder):
                    continue
                print('Testing subject:%d in folder Index: %d' %(sub,folder))

                test_sub.append(sub)
                if runInProcess:
                    p = multiprocessing.Process(target=test2,args=(ML_Model,yml,excel_sub,test_sub,loadData,loadData_test,getData,
                                                                   Min2DataPath,channDic,eventTypeDic,curModelFolder,folderName,fileName,timeStr,folder))
                    p.start()
                    p.join()
                else:
                    test2(ML_Model, yml,excel_sub, test_sub, loadData, loadData_test,getData, Min2DataPath,channDic, eventTypeDic,
                      curModelFolder, folderName,fileName,dateTimeStr=timeStr, folderIndex=folder)
                tf.keras.backend.clear_session()
        except Exception as err:
            raise err
    print("Done.")