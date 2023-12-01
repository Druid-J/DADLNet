import numpy as np
from sklearn.utils import shuffle
import sys
#sys.path.append('..')
import os,re
from meya.fileAction import joinPath,getRawDataFromFile,saveRawDataToFile
import mne
from math import ceil
import os,glob
from meya.loadData_YML import get_windowSize_step,get_windowSize_step2,StandardScaler,selectDataLog_binary,selectDataLog,getLearnDataPath_sess,existMulLearnDataPath,find_EventIndex_Time,existTestLearnDataPath,getLearnTestDataPath_sess
import random
from meya.hotCode import hot_code
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf

def countSubNum(yml):
    #glob.glob返回指定路径下所有符号条件的文件 以列表方式存储
    subPath=glob.glob(joinPath(yml['Meta']['initDataFolder'], 's*'))
    subId=[int(os.path.basename(s).replace('s','')) for s in subPath]
    return subId

def loadData(yml,subId,folder,channelDic,eventTypeDic,func,**kwargs):
    #func：getData
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath =joinPath(dataPath,'learnData')#需要修改
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    getAll = False
    if kwargs.keys().__contains__('isTrain'):
        getAll = not kwargs['isTrain']
    learnData_tra=[]
    learnLabel_tra=[]
    learnData_te = []
    learnLabel_te = []
    bdf_list = []
    sessnum = yml['Meta']['sessionNum']
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    if reCreate:
        load_bdf = True
    else:
        #判断是否已经存在的pkl文件路径 如果存在就不生成数据，避免多次生成数据
        # exist = existMulLearnDataPath(yml, subId, '', "",getAll=getAll)
        #二分类/三分类数据存在判断方法
        # exist = existMulLearnDataPath(yml, subId,folder)
        exist = False
        load_bdf = not exist
    if load_bdf:
        #将当前被试的数据名称存放在bdf_list中
        for i in range(1,sessnum+1):
            # bdf_list = glob.glob(joinPath(yml['Meta']['initDataFolder'],'s{0:02d}'.format(subId),'session{0:02d}'.format(i), '*data.bdf'))
            bdf_list.append(joinPath(yml['Meta']['initDataFolder'],'s{0:02d}'.format(subId),'session{0:02d}'.format(i), 'data.bdf'))
        bdf_list.sort()
        # bdf_list.sort(reverse = True)
    else:
        #如果数据文件存在，则不在lodadata中加载，设置值为None 分别在各自py文件中加载
        learnData_tra = None
        learnLabel_tra = None
        learnData_val = None
        learnLabel_val = None
        # 基础数据路径
        print('--------Data is exist------')
        trainType = yml['Meta']['trainType']
        InitPath = yml['Meta']['initDataFolder']
        # 判断训练类型
        if trainType == 'intra':
            DataPath = joinPath(InitPath, 'IntralearnData')
        elif trainType == 'binaryInter':
            DataPath = joinPath(InitPath, 'Binaryinterdata')
        elif trainType == 'inter':
            DataPath = joinPath(InitPath, 'InterlearnData')
        print('\n-----form %s load data------'%(DataPath))
        print('\n--loda subject{}_fold{} data --'.format(subId,folder))
        learnData_tra,learnLabel_tra = load_train_Data(subId,DataPath,folder)
        learnData_val, learnLabel_val = load_val_data(subId, DataPath, folder)
        print('\nTrain_x shape{},Train_y shape{},val_x shape{},val_y shape{}'.format(learnData_tra.shape,learnLabel_tra.shape,learnData_val.shape,learnLabel_val.shape))
        return learnData_tra, learnLabel_tra, learnData_val, learnLabel_val
    #     if pkl_num <= 1:
    #         learnDataPath_tra = getLearnDataPath_sess(yml, subId, '', "",getAll=getAll,tra=True)
    #         if not reCreate and os.path.exists(learnDataPath_tra):
    #             leaDic = getRawDataFromFile(learnDataPath_tra)
    #             learnData_tra.extend(leaDic['Data_tra'])
    #             learnLabel_tra.extend(leaDic['Label_tra'])
    #         learnDataPath_te = getLearnDataPath_sess(yml, subId, '', "", getAll=getAll, tra=False)
    #         if not reCreate and os.path.exists(learnDataPath_te):
    #             leaDic = getRawDataFromFile(learnDataPath_te)
    #             learnData_te.extend(leaDic['Data_te'])
    #             learnLabel_te.extend(leaDic['Label_te'])
    #     else:
    #         countLoad = 0
    #         curData_tra= []
    #         curLable_tra= []
    #         curData_te = []
    #         curLable_te = []
    #         for p_index in range(pkl_num):
    #             learnDataPath = getLearnDataPath_sess(yml, subId, '', '', split_index=p_index,getAll=getAll,tra=True)
    #             if not reCreate and os.path.exists(learnDataPath):
    #                 leaDic = getRawDataFromFile(learnDataPath)
    #                 curData_tra.extend(leaDic['Data_tra'])
    #                 curLable_tra.extend(leaDic['Label_tra'])
    #             learnDataPath = getLearnDataPath_sess(yml, subId, '', '', split_index=p_index, getAll=getAll,tra=False)
    #             if not reCreate and os.path.exists(learnDataPath):
    #                 leaDic = getRawDataFromFile(learnDataPath)
    #                 curData_te.extend(leaDic['Data'])
    #                 curLable_te.extend(leaDic['Label'])
    #                 countLoad += 1
    #         if countLoad == pkl_num:
    #             learnData_tra.extend(curData_tra)
    #             learnLabel_tra.extend(curLable_tra)
    #             learnData_te.extend(curData_te)
    #             learnLabel_te.extend(curLable_te)
    #     return learnData_tra, learnLabel_tra,learnData_te,learnLabel_te
    sub_data, sub_label = [], []
    doFilter=yml['Meta']['doFilter']
    dataGroup={}
    for fname in bdf_list:
        print('load session name:',fname)
        sessindex = int(os.path.basename(os.path.dirname(fname))[-2:])
        try:
            #读取当前被试的raw events
            raw,events = getRawEvent(sessindex,fname, yml, eventTypeDic,subId)
            #fun:loadData_YML :getData_chanSeq()
            sub_data, sub_label = func(yml, raw, events, channelDic,eventTypeDic,getAll)
            if sub_data!=None and sub_label!=None:
                dataShape=sub_data[0].shape
                if dataGroup.__contains__(dataShape):
                    # dataGroup[dataShape]['sess2_Data'].append(sub_data)
                    # dataGroup[dataShape]['sess2_Label'].append(sub_label)
                    dataGroup[dataShape]['test_Data'] = sub_data
                    dataGroup[dataShape]['test_label'] = sub_label
                    
                else:
                    dataGroup.setdefault(dataShape,{"train_Data":sub_data,"train_label":sub_label})
        except Exception as err:
            #print('Read file:{0} --------- err:{1}'.format(fname,err))
            raise err

    maxGroupKey=""
    maxGroupCount=0
    for key,value in dataGroup.items():
        if maxGroupCount<len(value['train_Data']):
            maxGroupCount=len(value['train_Data'])
            maxGroupKey=key
    sub_data_tra=dataGroup[maxGroupKey]['train_Data']
    sub_label_tra=dataGroup[maxGroupKey]['train_label']
    sub_data_te = dataGroup[maxGroupKey]['test_Data']
    sub_label_te = dataGroup[maxGroupKey]['test_label']
    sub_data_tra=np.array(sub_data_tra,dtype=np.float)
    sub_label_tra=np.array(sub_label_tra,dtype=np.int)
    sub_data_te= np.array(sub_data_te, dtype=np.float)
    sub_label_te= np.array(sub_label_te, dtype=np.int)
    # if 'shuffle' in yml['ML'] and yml['ML']['shuffle']:
    #     sub_data, sub_label = shuffle(sub_data, sub_label)
    # if pkl_num > 1:
    #     totalLen = len(sub_label)
    #     splitLen = ceil(totalLen / pkl_num)
    #     startIndex = 0
    #     endIndex = 0
    #     i = 0
    #     print('sub. %d -- splitLen: %d' % (subId, splitLen))
    #     while endIndex < totalLen:
    #         endIndex = startIndex + splitLen
    #         if endIndex >= totalLen:
    #             endIndex = totalLen
    #         splitData = sub_data[startIndex:endIndex]
    #         splitLabel = sub_label[startIndex:endIndex]
    #         learnDataPath = getLearnDataPath_sess(yml, subId, '', "", split_index=i,getAll=getAll)
    #         saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
    #         startIndex = endIndex
    #         i += 1
    # else:
        # learnDataPath_tra =getLearnDataPath_sess(yml, subId, '', "",getAll=getAll,tra=True)
        # learnDataPath_te = getLearnDataPath_sess(yml, subId, '', "", getAll=getAll,tra=False)
        # saveRawDataToFile(learnDataPath_tra, {'Data_tra': sub_data_tra, 'Label_tra': sub_label_tra})
        # saveRawDataToFile(learnDataPath_te, {'Data_te': sub_data_te, 'Label_te': sub_label_te})
    learnData_tra.extend(sub_data_tra)
    learnLabel_tra.extend(sub_label_tra)
    learnData_te.extend(sub_data_te)
    learnLabel_te.extend(sub_label_te)
    return learnData_tra,learnLabel_tra,learnData_te,learnLabel_te

def loadData_transfer(yml,subId,fold,**kwargs):
    # func：getData
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = joinPath(dataPath, 'learnData')
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    learnTrainData = []
    learnTrainLabel = []
    learnValData = []
    learnValLabel = []
    learnTestData = []
    learnTestLabel = []
    try:
        #train
        file_train_x = learnDataPath + '/X_train_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        file_train_y = learnDataPath + '/y_train_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        learnTrainData = np.load(file_train_x, allow_pickle=True)
        learnTrainLabel = np.load(file_train_y, allow_pickle=True)
        #valid
        file_val_x = learnDataPath + '/X_val_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        file_val_y = learnDataPath + '/y_val_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        learnValData = np.load(file_val_x, allow_pickle=True)
        learnValLabel = np.load(file_val_y, allow_pickle=True)
        #test
        file_test_x = learnDataPath + '/X_test_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        file_test_y = learnDataPath + '/y_test_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        learnTestData = np.load(file_test_x, allow_pickle=True)
        learnTestLabel = np.load(file_test_y, allow_pickle=True)
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_train_x, file_train_y))

    # learnLabel = to_categorical(learnLabel)
    return learnTrainData,learnTrainLabel,learnValData,learnValLabel,learnTestData,learnTestLabel


# def loadData_test(yml,subId,fold,channelDic,eventTypeDic,func,**kwargs):
def loadData_test(yml, subId, fold, datapath, **kwargs):
    #func：getData
    # dataPath = yml['Meta']['initDataFolder']
    # towClass = yml['Meta']['isTowClass']
    # if towClass:
    #     learnDataPath =joinPath(dataPath,'learnData')
    #     if not os.path.exists(learnDataPath):
    #         os.mkdir(learnDataPath)
    reCreate = False
    num_class = yml['Meta']['ClassNum']
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    getAll = False
    if kwargs.keys().__contains__('isTrain'):
        getAll = not kwargs['isTrain']
    #MNNUDATA 地址
    load_path = datapath
    learnData=[]
    learnLabel=[]
    try:
        file_x = load_path + '/X_test_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        file_y = load_path + '/y_test_S{:03d}_fold{:03d}.npy'.format(subId, fold)
        learnData = np.load(file_x, allow_pickle=True)
        # learnData_expand = learnData * 100000
        X_mean = np.mean(np.absolute(learnData))
        # X_mean_expand = np.mean(learnData_expand)
        print('Raw meam:{}'.format(X_mean))
        learnLabel = np.load(file_y, allow_pickle=True)
        # learnLabel = learnLabel-1
        # if yml['ML']['loss']=='binary_crossentropy':
        #     learnLabel = K.argmax(learnLabel,axis=-1)
        #     learnLabel = learnLabel.numpy().tolist()
        #     learnLabel = np.array(learnLabel, dtype='int32')
        # if yml['Meta']['ClassNum'] == 2:
        #     learnLabel= learnLabel - 1

        # learnData,learnLabel = shuffle(learnData,y_binary)
        # learnLabel = to_categorical(learnLabel)
        print('load train_data %s,train_label %s'%(file_x,file_y))
        print('loda test data_shape{},label_shape{}'.format(learnData.shape,learnLabel.shape))
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return learnData, learnLabel


def get_Kaya_36MI_1D():#Kaya_36通道
    return {'FT7': 8, 'FC5': 9, 'FC3': 10, 'FC1':11, 'FCz': 12, 'FC2': 13, 'FC4': 14, 'FC6': 15, 'FT8': 16, 'T7': 17,
            'C5':18, 'C3': 19, 'C1': 20, 'Cz': 21,'C2': 22, 'C4': 23, 'C6': 24, 'T8': 25, 'TP7': 26, 'CP5': 27, 'CP3': 28,
            'CP1': 29, 'CPz': 30, 'CP2': 31,'CP4': 32, 'CP6': 33,'TP8': 34, 'P7': 35, 'P5': 36, 'P3': 37, 'P1': 38, 'Pz': 39,
            'P2': 40, 'P4': 41, 'P6': 42, 'P8': 43}

def getEprimeEvent(path=None):
    if path is None:
        raise Exception('No description data found, path can not be empty')
    
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            data.append(line.strip('\n'))
        f.close()
    onset = []
    duration = []
    description = []
    
    for index, line in enumerate(data):
        if line.find('*** LogFrame Start ***') != -1:
            tempIndex = index + 4
            curLine = data[tempIndex]
            if curLine.find("Trigger") != -1:
                splitNum = curLine.find(":")
                type = curLine[splitNum + 2:]
                if type != '-':
                    description.append('1')
                    description.append(type)
                tempIndex = tempIndex + 7
                curLine = data[tempIndex]
                if curLine.find('num3.ActionDelay') != -1:
                    tempIndex = tempIndex+1
                    curLine = data[tempIndex]
                if curLine.find('num3.ActionTime') != -1:
                    splitNum = curLine.find(":")
                    time = curLine[splitNum + 2:]
                    if time != '-':
                        onset.append(int(time))
                        duration.append(0)
                    tempIndex = tempIndex + 4
                    curLine = data[tempIndex]
                    if curLine.find('stimul.ActionDelay') != -1:
                        tempIndex = tempIndex + 1
                        curLine = data[tempIndex]
                    if curLine.find('stimul.ActionTime') != -1:
                        splitNum = curLine.find(":")
                        time = curLine[splitNum + 2:]
                        if time != '-':
                            onset.append(int(time))
                            duration.append(0)
                        tempIndex = tempIndex + 5
                        curLine = data[tempIndex]
                        if curLine.find('rest.ActionTime') != -1:
                            splitNum = curLine.find(":")
                            time = curLine[splitNum + 2:]
                            if time != '-':
                                onset.append(int(time))
                                duration.append(0)
                                description.append('4')
            
            if curLine.find("Sample") != -1:
                tempIndex = tempIndex + 5
                curLine = data[tempIndex]
                if curLine.find('TextDisplay3.ActionDelay') != -1:
                    tempIndex = tempIndex + 1
                    curLine = data[tempIndex]
                if curLine.find('ActionTime') != -1:
                    splitNum = curLine.find(":")
                    time = curLine[splitNum + 2:]
                    if time != '-':
                        onset.append(int(time))
                        duration.append(0)
                        description.append('5')
    return onset, duration, description

def getRawEvent(sessindex,bdfFile,yml, eventTypeDic,subId):
    #1.读取bdf文件 并获取通道，采样率等信息
    rawEEG = mne.io.read_raw_bdf(bdfFile, exclude=yml['Meta']['excludeChs'], verbose=False, preload=True)
    ch_names = rawEEG.info['ch_names']
    data, _ = rawEEG[:len(ch_names)]
    fs = rawEEG.info['sfreq']
    #event文件路径
    eventFile=joinPath(os.path.dirname(bdfFile),os.path.basename(bdfFile).replace('data','evt'))
    ## read events
    try:
        annotationData = mne.io.read_raw_bdf(eventFile)
        try:
            tal_data = annotationData._read_segment_file([], [], 0, 0, int(annotationData.n_times), None, None)
            print('mne version <= 0.20')
        except:
            idx = np.empty(0, int)
            tal_data = annotationData._read_segment_file(np.empty((0, annotationData.n_times)), idx, 0, 0,
                                                         int(annotationData.n_times), np.ones((len(idx), 1)), None)
            print('mne version > 0.20')
            
        file = joinPath(yml['Meta']['initDataFolder'], 's{0:02d}'.format(subId),'session{0:02d}'.format(sessindex), 's{0:02d}-summary.txt'.format(subId))
        if os.path.exists(file):
            onset, duration, description = getEprimeEvent(file)
            onset = np.array([i for i in onset], dtype=np.int64)
            duration = np.array([int(i) for i in duration], dtype=np.int64)
            #No MI set 0 for event
            desc = np.array([int(i) if int(i) in eventTypeDic.keys() else 0 for i in description], dtype=np.int64)#np.array([int(i) for i in description], dtype=np.int64)
            #不对标签进行处理
            # desc = np.array([int(i) for i in description], dtype=np.int64)
            events = np.vstack((onset, duration, desc)).T
            events=np.r_[[[0, 0, 0]], events]#一开始也被定义为rest
        else:
            #标签以及 每个标签对应时间
            onset, duration, description = read_annotations_bdf(tal_data[0])
            onset = np.array([i * fs for i in onset], dtype=np.int64)
            duration = np.array([int(i) for i in duration], dtype=np.int64)
            desc = np.array([int(i) if int(i) in eventTypeDic.keys() else 0 for i in description], dtype=np.int64)
            #不对标签进行处理
            # desc = np.array([int(i) for i in description], dtype=np.int64)
            events = np.vstack((onset, duration, desc)).T
            events = np.r_[[[0, 0, 0]], events]
    except:
        print('not found any event')
        events = []

    # ch_type= ['eeg' for i in range(len(ch_names))]
    # ch_type.extend(['stim'] * len(eventTypeDic))
    # ch_names.extend([eValue['Name'] for eValue in eventTypeDic.values()])
    # info = mne.create_info(ch_names=ch_names, sfreq=yml['Meta']['frequency'], ch_types=ch_type)
    # from mne.io import RawArray
    # data = np.pad(data.T, ((0, 0), (0, len(eventTypeDic))))
    # hc=hot_code(list(eventTypeDic.keys()))
    # data[0:events[0][0], -3:] = hc.one_hot_encode(desc)
    # for ed in events:
    #     data[ed[0]:ed[0] + ed[1], -3:] = events([ed[2] for j in range(ed[1])])
    #
    # raw=RawArray(data.T, info, verbose=False)
    # raw.add_events(events)
    if yml['Meta']['doFilter']:
        minFre = yml['Meta']['minFreq']
        maxFre = yml['Meta']['maxFreq']
        rawEEG.filter(l_freq=minFre, h_freq=maxFre)
    yml['Meta']['frequency']=rawEEG.info['sfreq']

    return rawEEG,events

def read_annotations_bdf(annotations):
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == np.int32:  # BDF
                this_chan.dtype = np.uint8
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                for s in this_chan:
                    tals.extend(s)
            else:
                for s in this_chan:
                    i = int(s)
                    tals.extend(np.uint8([i % 256, i // 256]))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return zip(*events) if events else (list(), list(), list())

def getEvents(Dataset,classNum):
    if Dataset == 'OpenBMI':
        if classNum == 2:
            return {
                    1:{'Name':'right','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                    2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                   }
        elif classNum == 3:
            return {0:{'Name':'rest','StaTime':0,'TimeSpan':1000,'IsExtend':False},
                    1:{'Name':'right','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                    2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                    }
    elif Dataset == 'BCI2a':
        if classNum == 2:
            return {
                    1:{'Name':'right','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                    2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                   }
        elif classNum == 4:
            return {0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                    1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                    2: {'Name': 'foot', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                    3: {'Name': 'tongue', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                    }
    elif Dataset == "MNNUData":
        if classNum == 2:
            return {
                    2:{'Name':'left','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                    3: {'Name':'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                   }
        elif classNum == 3:
            return {0: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                    2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                    3: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                    }


def get_26MI_1D():
    return {'FT7':23,'FC5':21,'FC3':19,'FC1':17,'FCz':16,'FC2':18,'FC4':20,'FC6':22,'FT8':24,'T7':32,'C5':30,'C3':28,'C1':26,'Cz':25,'C2':27,
            'C4':29,'C6':31,'T8':33,'TP7':40,'CP5':38,'CP3':36,'CP1':34,'CP2':35,'CP4':37,'CP6':39,'TP8':41}

def get_20MI_1D_MNNU():
    return {'FC5':21,'FC3':19,'FC1':17,'FCz':16,'FC2':18,'FC4':20,'FC6':22,'C5':30,'C3':28,'C1':26,'Cz':25,'C2':27,
            'C4':29,'C6':31,'CP5':38,'CP3':36,'CP1':34,'CP2':35,'CP4':37,'CP6':39}

def get_31MI_1D():#OpenBMI31通道
    return {'FC5': 7, 'FC3': 32, 'FC1': 8, 'FC2': 9, 'FC4': 33, 'FC6': 10,'T7':11, 'C5': 34, 'C3': 12, 'C1': 35, 'Cz': 13,
            'C2': 36, 'C4': 14, 'C6': 37, 'T8':15,'TP7': 47,'CP5':17,'CP3': 38, 'CP1': 18, 'CPz': 39, 'CP2': 19, 'CP4': 40, 'CP6': 20,
            'TP8':52,'P7':22,'P3':23,'P1':41,'Pz':24,'P2':42,'P4':25,'P8':26}

def get_8MI_1D():#EEGSym论文8通道
    return {'F3': 3, 'C3': 12, 'P3': 23,
            'Cz': 13, 'Pz': 24,
            'F4': 5, 'C4': 14, 'P4': 25}

def get_16MI_1D():#EEGSym论文16通道
    return {'F3': 3, 'C3': 12, 'P3': 23, 'F7': 2, 'T7': 11, 'P7': 22, 'O1': 28,
            'Cz': 13, 'Pz': 24,
            'F4': 5, 'C4': 14, 'P4': 25, 'F8': 6, 'T8': 15, 'P8': 26, 'O2': 30}

def get_8MI_1D_DLDANet_3():#FBCNet/OpenBMI 20通道
    return {'FC5': 7,'FC6': 10,'C5': 34,'Cz': 13,'C6': 37,'CP5': 17,'CPz': 39,'CP6': 20}

def get_8MI_1D_DLDANet_2():#FBCNet/OpenBMI 20通道
    return {'FC3':32,'FC4':33,'C3':12,'Cz':13,'C4':14,'CP3':38,'CPz':39,'CP4':40}

def get_8MI_1D_DLDANet_1():#FBCNet/OpenBMI 8通道方案1
    return {'FC1':8,'FC2':9,'C1':35,'Cz':13,'C2':36,'CP1':18,'CPz':39,'CP2':19}

# def get_14MI_1D_2():#FBCNet/OpenBMI 14通道
#     return {'FC5': 7,  'FC1': 8, 'FC2': 9,  'FC6': 10, 'C5': 34,  'C1': 35, 'Cz': 13,
#             'C2': 36, 'C6': 37, 'CP5': 17,  'CP1': 18, 'CPz': 39, 'CP2': 19, 'CP6': 20}

def get_14MI_1D():#FBCNet/OpenBMI 20通道
    return {'FC3':32,'FC1':8,'FC2':9,'FC4':33,'C3':12,'C1':35,'Cz':13,'C2':36,
            'C4':14,'CP3':38,'CP1':18,'CPz':39,'CP2':19,'CP4':40}


def get_20MI_1D():#FBCNet/OpenBMI 20通道
    return {'FC5':7,'FC3':32,'FC1':8,'FC2':9,'FC4':33,'FC6':10,'C5':34,'C3':12,'C1':35,'Cz':13,'C2':36,
            'C4':14,'C6':37,'CP5':17,'CP3':38,'CP1':18,'CPz':39,'CP2':19,'CP4':40,'CP6':20}

def get_31MI_1D_EEGSym():#EEGSym论文31通道
    return {'FC1': 8, 'C1': 35, 'CP1': 18, 'P1': 41, 'FC3': 32, 'C3': 12, 'CP3': 38, 'P3': 23, 'FC5': 7, 'C5': 34, 'CP5': 17, 'T7': 11, 'TP7': 47, 'P7': 22,
            'Cz': 13, 'CPz': 39, 'Pz': 24,
            'FC2': 9, 'C2': 36, 'CP2': 19, 'P2': 42, 'FC4': 33, 'C4': 14, 'CP4': 40, 'P4': 25, 'FC6': 10, 'C6': 37, 'CP6': 20, 'T8': 15, 'TP8': 52, 'P8': 26}


def load_train_Data(sub,load_path,folderIndex):
    try:
        file_x = load_path + '/X_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x,allow_pickle=True)
        y = np.load(file_y,allow_pickle=True)
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X,y

def load_val_data(sub,load_path,folderIndex):
    try:
        file_x = load_path + '/X_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x, allow_pickle=True)
        y = np.load(file_y, allow_pickle=True)
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X, y

## EEGSym DA代码
def preprocessing_function(augmentation=True):
    """Custom Data Augmentation for EEGSym.

        Parameters
        ----------
        augmentation : Bool
            If the augmentation is performed to the input.

        Returns
        -------
        data_augmentation : function
            Data augmentation performed to each trial
    """

    def data_augmentation(trial):
        """Custom Data Augmentation for EEGSym.

            Parameters
            ----------
            trial : tf.tensor
                Input of the

            Returns
            -------
            data_augmentation : keras.models.Model
                Data augmentation performed to each trial
        """

        samples, ncha, _ = trial.shape

        augmentations = dict()
        augmentations["patch_perturbation"] = 0
        augmentations["random_shift"] = 0
        augmentations["hemisphere_perturbation"] = 0
        augmentations["no_augmentation"] = 0

        selectionables = ["patch_perturbation", "random_shift",
                          "hemisphere_perturbation", "no_augmentation"]
        probabilities = None

        if augmentation:
            selection = np.random.choice(selectionables, p=probabilities)
            augmentations[selection] = 1

            method = np.random.choice((0, 2))
            std = 'self'
            # elif data_augmentation == 1:  # Random shift
            for _ in range(augmentations["random_shift"]):  # Random shift
                # Select position where to erase that timeframe
                position = 0
                if position == 0:
                    samples_shifted = np.random.randint(low=1, high=int(
                        samples * 0.5 / 3))
                else:
                    samples_shifted = np.random.randint(low=1, high=int(
                        samples * 0.1 / 3))

                if method == 0:
                    shifted_samples = np.zeros((samples_shifted, ncha, 1))
                else:
                    if std == 'self':
                        std_applied = np.std(trial)
                    else:
                        std_applied = std
                    center = 0
                    shifted_samples = np.random.normal(center, std_applied,
                                                       (samples_shifted, ncha,
                                                        1))
                if position == 0:
                    trial = np.concatenate((shifted_samples, trial),
                                           axis=0)[:samples]
                else:
                    trial = np.concatenate((trial, shifted_samples),
                                           axis=0)[samples_shifted:]

            for _ in range(
                    augmentations["patch_perturbation"]):  # Patch perturbation
                channels_affected = np.random.randint(low=1, high=ncha - 1)
                pct_max = 1
                pct_min = 0.2
                pct_erased = np.random.uniform(low=pct_min, high=pct_max)
                # Select time to be erased acording to pct_erased
                # samples_erased = np.min((int(samples*ncha*pct_erased//channels_affected), samples))#np.random.randint(low=1, high=samples//3)
                samples_erased = int(samples * pct_erased)
                # Select position where to erase that timeframe
                if samples_erased != samples:
                    samples_idx = np.arange(samples_erased) + np.random.randint(
                        samples - samples_erased)
                else:
                    samples_idx = np.arange(samples_erased)
                # Select indexes to erase (always keep at least a channel)
                channel_idx = np.random.permutation(np.arange(ncha))[
                              :channels_affected]
                channel_idx.sort()
                for channel in channel_idx:
                    if method == 0:
                        trial[samples_idx, channel] = 0
                    else:
                        if std == 'self':
                            std_applied = np.std(trial[:, channel]) \
                                          * np.random.uniform(low=0.01, high=2)
                        else:
                            std_applied = std
                        center = 0
                        trial[samples_idx, channel] += \
                            np.random.normal(center, std_applied,
                                             trial[samples_idx, channel,
                                             :].shape)
                        # Standarize the channel again after the change
                        temp_trial_ch_mean = np.mean(trial[:, channel], axis=0)
                        temp_trial_ch_std = np.std(trial[:, channel], axis=0)
                        trial[:, channel] = (trial[:,
                                             channel] - temp_trial_ch_mean) / temp_trial_ch_std

            for _ in range(augmentations["hemisphere_perturbation"]):
                # Select side to mix/change for noise
                left_right = np.random.choice((0, 1))
                if method == 0:
                    if left_right == 1:
                        channel_idx = np.arange(ncha)[:int((ncha / 2) - 1)]
                        channel_mix = np.random.permutation(channel_idx.copy())
                    else:
                        channel_idx = np.arange(ncha)[-int((ncha / 2) - 1):]
                        channel_mix = np.random.permutation(channel_idx.copy())
                    temp_trial = trial.copy()
                    for channel, channel_mixed in zip(channel_idx, channel_mix):
                        temp_trial[:, channel] = trial[:, channel_mixed]
                    trial = temp_trial
                else:
                    if left_right == 1:
                        channel_idx = np.arange(ncha)[:int((ncha / 2) - 1)]
                    else:
                        channel_idx = np.arange(ncha)[-int((ncha / 2) - 1):]
                    for channel in channel_idx:
                        trial[:, channel] = np.random.normal(0, 1,
                                                             trial[:,
                                                             channel].shape)

        return trial

    return data_augmentation

## EEGSym DA代码
def trial_iterator(X, y, batch_size=32, shuffle=True, augmentation=True):
    """Custom trial iterator to pretrain EEGSym.

        Parameters
        ----------
        X : tf.tensor
            Input tensor of  EEG features.
        y : tf.tensor
            Input tensor of  labels.
        batch_size : int
            Number of features in each batch.
        shuffle : Bool
            If the features are shuffled at each training epoch.
        augmentation : Bool
            If the augmentation is performed to the input.

        Returns
        -------
        trial_iterator : tf.keras.preprocessing.image.NumpyArrayIterator
            Iterator used to train the model.
    """

    trial_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function(
            augmentation=augmentation))

    trial_iterator = tf.keras.preprocessing.image.NumpyArrayIterator(
        X, y, trial_data_generator, batch_size=batch_size, shuffle=shuffle,
        sample_weight=None,
        seed=None, data_format=None, save_to_dir=None, save_prefix='',
        save_format='png', subset=None, dtype=None
    )
    return trial_iterator
