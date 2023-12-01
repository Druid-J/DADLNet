import numpy as np
from sklearn.utils import shuffle
import scipy.io as scio
from mne.io import RawArray
import sys
sys.path.append('..')
import os
from meya.fileAction import joinPath,getRawDataFromFile,saveRawDataToFile
import mne
import scipy.io as sio
from meya.hotCode import hot_code
import random
from mne.time_frequency import tfr_multitaper
from sklearn.preprocessing import StandardScaler
from math import sqrt,ceil
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test

def get_channelName():
    return  ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10',
'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','FC3','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9',
'FTT9h','FTT7h','TP7','TPP9h','FT10','FTT10h','FTT8h','TP8','TPP10h','F9','F10','AF7','AF3','AF4','AF8','PO3','PO4']

def get_20MI_1D():#OpenBMI左右对称
    return {'FC5':7,'FC3':32,'FC1':8,'FC2':9,'FC4':33,'FC6':10,'C5':34,'C3':12,'C1':35,'Cz':13,'C2':36,'C4':14,'C6':37,'CP5':17,
            'CP3':38,'CP1':18,'CPz':39,'CP2':19,'CP4':40,'CP6':20}

def getEvents():
    # return {0:{'Name':'rest','StaTime':0,'TimeSpan':1000,'IsExtend':False},
    #         2:{'Name':'left','StaTime':0,'TimeSpan':4000,'IsExtend':False},
    #         3: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
    #         }
    return {
            0: {'Name': 'rest', 'StaTime':[0], 'TimeSpan':3000, 'IsExtend':False},
            1: {'Name': 'right', 'StaTime': [0], 'TimeSpan': 4000, 'IsExtend': False},
            2: {'Name': 'left', 'StaTime': [0], 'TimeSpan': 4000, 'IsExtend': False}
            }

def getLearnDataPath(yml,isTrain,subId):
    dataPath = yml['Meta']['initDataFolder']
    dataSegType=yml['Meta']['segmentName']
    if isTrain:
        learnDataPath = '%s/learnData/MI_Train_%s_s%d.pkl' % (dataPath, dataSegType, subId)
    else:
        learnDataPath = '%s/learnData/MI_Test_%s_s%d.pkl' % (dataPath, dataSegType, subId)
    return learnDataPath

def loadData(yml,subId,channelDic,eventTypeDic,func,isTrain=True):
    dataPath=yml['Meta']['initDataFolder']
    learnDataPath='%s/learnData'%dataPath
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    learnDataPath = getLearnDataPath(yml, isTrain, subId)

    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    if not reCreate and os.path.exists(learnDataPath):
        leaDic=getRawDataFromFile(learnDataPath)
        learnData=leaDic['Data']
        label=leaDic['Label']
    else:
        subData = []
        subLabel = []
        windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
        if isTrain:
            key = 'EEG_MI_train'
        else:
            key = 'EEG_MI_test'
        #为每个被试读取session1和session2的数据
        # for sess in ['session1','session2']:
        for sess in ['sess01', 'sess02']:
            # fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
            fname = joinPath(yml['Meta']['initDataFolder'],"{}_subj{:0>2d}_EEG_MI.mat".format(sess,subId))
            matlab_data = scio.loadmat(fname)
            #？
            raw, events = getRawEvent(matlab_data, key, yml,eventTypeDic)
            events = np.array(events, dtype='int32')
            data=raw._data[0:len(get_channelName())].T
            expand=1
            if 'expandData' in yml['Meta']:
                expand=yml['Meta']['expandData']
            if expand!=1:
                data=data*expand
            # learnData, label = func(yml,data,events,windowSize,eventTypeDic,channelDic)
            learnData, label = func(yml, raw, events,channelDic, eventTypeDic, getAll=False)
            subData.extend(learnData)
            subLabel.extend(label)
        (subData, subLabel) = shuffle(subData, subLabel)
        subData=np.array(subData,dtype='float64')
        subLabel=np.array(subLabel,dtype='int32')
        saveRawDataToFile(learnDataPath, {'Data': subData, 'Label': subLabel})
    return learnData, label


def getRawEvent_BCI2008(yml,DataPath,epochWindow = [0,4]):
    """
    BCI2008raw数据读取方法
    """

    offset = 2
    RawdataPath = DataPath+'.gdf'
    # load the gdf file using MNE
    raw = mne.io.read_raw_gdf(RawdataPath)
    raw.load_data()

    if 'picksChannel' in yml['Meta']:
        picks = yml['Meta']['dataChannel']
        raw.pick_channels(picks, ordered=True)

    #滤波
    if yml['Meta']['multiBand']:
        filterBand = yml['Meta']['filter']
        multiBandData,multiBandLabel = [],[]
        for i,band in enumerate(filterBand):
            raw_copy = raw.copy()
            minFre = band[0]
            maxFre = band[1]
            print("---------Start filter--------")
            raw_copy.filter(l_freq=minFre, h_freq=maxFre)

            if 'notch' in yml['Meta']:
                for notch in yml['Meta']['notch']:
                    raw_copy.notch_filter(notch, method=yml['Meta']['FilterType'])


            if yml['Meta']['DodownSampling']:
                raw_copy = raw_copy.resample(yml['Meta']['downSampling'], npad='auto')
                fs = yml['Meta']['downSampling']
            else:
                fs = yml['Meta']['frequency']

            _, eventDic = mne.events_from_annotations(raw_copy)

            eventCode = [eventDic['768']]

            gdf_events = mne.events_from_annotations(raw_copy)[0][:, [0, 2]].tolist()

            eeg = raw_copy.get_data()


            chans = list(range(1, 21))
            # drop channels
            if chans is not None:
                eeg = eeg[chans, :]


            events = [event for event in gdf_events if event[1] in eventCode]

            epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs

            x = np.concatenate([eeg[:, epochInterval + event[0]] for event in events], axis=1)



            TruePath = DataPath + '.mat'
            label_T = sio.loadmat(TruePath)["classlabel"].squeeze()
            label_T = label_T - 1

            event_time = []
            for i in range(len(events)):
                if i == 0:
                    event_time.append(i)
                else:
                    event_time.append(i * len(epochInterval))
            duration = np.zeros((len(event_time),), dtype=int)
            eventData = np.vstack((event_time, duration, label_T)).T
            multiBandData.append(x)
            multiBandLabel.append(eventData)
        multiBandData = np.array(multiBandData,dtype='float32')
        multiBandLabel = np.array(multiBandLabel,dtype='int32')
        return multiBandData, multiBandLabel
    else:
        if yml['Meta']['reference']:
            print("start average reference\n")
            raw.set_eeg_reference(ref_channels='average')
        doFilter = yml['Meta']['doFilter']
        if doFilter:
            minFre = yml['Meta']['minFreq']
            maxFre = yml['Meta']['maxFreq']
            print("---------Start filter--------")
            raw.filter(l_freq=minFre, h_freq=maxFre, method=yml['Meta']['FilterType'])

        if 'notch' in yml['Meta']:

            for freq in yml['Meta']['notch']:
                raw.notch_filter(freq, method=yml['Meta']['FilterType'])

        if 'CAR' in yml['Meta']:

            raw.set_eeg_reference(ref_channels='average')

        if yml['Meta']['DodownSampling']:
            raw = raw.resample(yml['Meta']['downSampling'],npad='auto')
            fs = yml['Meta']['downSampling']
        else:
            fs = yml['Meta']['frequency']

        _,eventDic = mne.events_from_annotations(raw)

        eventCode = [eventDic['768']]

        gdf_events= mne.events_from_annotations(raw)[0][:, [0, 2]].tolist()

        eeg = raw.get_data()

        chans = list(range(1, 21))
        if 'dataChannel' in yml['Meta']:
            chans = list(range(0, len(yml['Meta']['dataChannel'])))
        # drop channels
        if chans is not None:
            eeg = eeg[chans, :]

        events = [event for event in gdf_events if event[1] in eventCode]

        epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs

        x = np.concatenate([eeg[:, epochInterval + event[0]] for event in events], axis=1)


        #读取true label
        TruePath = DataPath+'.mat'
        label_T = sio.loadmat(TruePath)["classlabel"].squeeze()
        label_T = label_T - 1

        event_time = []
        for i in range(len(events)):
            if i == 0:
                event_time.append(i)
            else:
                event_time.append(i * len(epochInterval))
        duration = np.zeros((len(event_time),), dtype=int)
        eventData = np.vstack((event_time, duration, label_T)).T
        return x,eventData

def getRawEvent(matlab_data,key,yml,doFilter=True):
    dataset = matlab_data[key][0][0]
    data = np.array(dataset['x'])
    eventTime = np.array(dataset['t'][0])
    # Fix TimePoint
    fix_eventTime=[]
    fix_time = 3
    orig_smp_freq = yml['Meta']['frequency']
    for i in range(len(eventTime)):
        fix_eventTime.append(eventTime[i]-fix_time*orig_smp_freq)
    rest_eventTime = []
    # Rest TimePoint
    MI_time = 4
    for i in range(len(eventTime)):
        rest_eventTime.append(eventTime[i]+MI_time*orig_smp_freq)
    fix_event = np.array(fix_eventTime,dtype=int)
    rest_event = np.array(rest_eventTime,dtype=int)
    event_Time = np.concatenate((eventTime,fix_event),axis=0)
    event_Time = np.concatenate((event_Time,rest_event),axis=0)
    event_Time.sort()
    # insert Rest&Fix label
    fix_index = []
    rest_index = []
    for i in range(len(event_Time)):
        if event_Time[i] in fix_event:
            fix_index.append(i)
        elif event_Time[i] in rest_event:
            rest_index.append(i)
    label_1d = np.array(dataset['y_dec'][0])
    for i in range(len(fix_index)):
        label_1d = np.insert(label_1d,fix_index[i],0)
        label_1d= np.insert(label_1d, rest_index[i], 0)
    # event
    duration = np.zeros((300,),dtype=int)
    eventData = np.vstack((event_Time,duration,label_1d)).T
    eventData = np.r_[[[0,0,0]],eventData]
    #get channel name
    ch_names = get_channelName()
    ch_type = ['eeg' for i in range(len(ch_names))]
    '''
    montage =mne.channels.make_standard_montage('standard_1005')# read_montage('standard_1005', ch_names)    
    '''
    info = mne.create_info(ch_names=ch_names, sfreq=yml['Meta']['frequency'], ch_types=ch_type)
    raw = RawArray(data.T, info, verbose=False)

    #EEGsym select channel
    if yml['Meta']['picksChannel']:
        picks = yml['Meta']['dataChannel']
        raw.pick_channels(picks, ordered=True)
    # FBCNet MultiFilter
    if yml['Meta']['multiBand']:
        filterBand = yml['Meta']['filter']
        multiBandData, multiBandLabel = [],[]
        for num, band in enumerate(filterBand):
            raw_copy = raw.copy()
            minFre = band[0]
            maxFre = band[1]
            print("---------Start filter--------")
            raw_copy.filter(l_freq=minFre, h_freq=maxFre, method=yml['Meta']['FilterType'])

            if 'CAR' in yml['Meta']:
                raw_copy.set_eeg_reference(ref_channels='average')
            # recording channels
            if num == 0:
                dataChannel = {}
                j = 0
                for chan in raw_copy.ch_names:
                    dataChannel.setdefault(chan, j)
                    j = j + 1
                yml['ML']['dataChannel'] = dataChannel

            if yml['Meta']['DodownSampling']:
                raw_copy = raw_copy.resample(yml['Meta']['downSampling'], npad='auto')
                sampleRate = yml['Meta']['downSampling'] / yml['Meta']['frequency']
                eventDataCopy = eventData.copy()
                for eD in eventDataCopy:
                    eD[0] = eD[0] * sampleRate
                    eD[1] = eD[1] * sampleRate
            # computer order of magnitude
            if num == 0:
                rawData = raw_copy._data
                channelDic_x = {}
                for i in range(len(rawData)):
                    channelDic_x.setdefault('chan{}'.format(i), np.mean(np.absolute(rawData[i])))
                rawData_mean = np.mean(np.absolute(rawData))
                print("__________rawData_mean{}_________".format(rawData_mean))

            multiBandData.append(raw_copy._data)
            multiBandLabel.append(eventDataCopy)
        multiBandData = np.array(multiBandData, dtype='float32')
        multiBandLabel = np.array(multiBandLabel, dtype='int32')

        return multiBandData, multiBandLabel, channelDic_x
    else:

        if yml['Meta']['reference']:
            print("Start Average Reference\n")
            raw.set_eeg_reference(ref_channels='average')

        if yml['Meta']['doFilter']:
            minFre = yml['Meta']['minFreq']
            maxFre = yml['Meta']['maxFreq']
            print("---------Start filter--------")

            raw.filter(l_freq=minFre, h_freq=maxFre, method=yml['Meta']['FilterType'])




        dataChannel = {}
        i = 0
        for chan in raw.ch_names:
            dataChannel.setdefault(chan, i)
            i = i + 1
        yml['ML']['dataChannel'] = dataChannel

        if yml['Meta']['DodownSampling']:
            raw = raw.resample(yml['Meta']['downSampling'], npad='auto')
            sampleRate = yml['Meta']['downSampling'] / yml['Meta']['frequency']
            for eD in eventData:
                eD[0] = eD[0] * sampleRate
                eD[1] = eD[1] * sampleRate
        rawData = raw._data
        channelDic_x = {}
        for i in range(len(rawData)):
            channelDic_x.setdefault('chan{}'.format(i), np.mean(np.absolute(rawData[i])))
        rawData_mean = np.mean(np.absolute(rawData))
        print("______________________________rawData_mean{}__________________".format(rawData_mean))

        return rawData, eventData, channelDic_x
