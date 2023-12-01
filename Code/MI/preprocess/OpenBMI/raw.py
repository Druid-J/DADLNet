import numpy as np
import scipy.io as sio
import mne
from MI.utils import resampling
from MI.preprocess.config import CONSTANT
from sklearn.model_selection import train_test_split
from meya_load_OpenBMI.New_loadData import getRawEvent
from meya.fileAction import selectData
from mne.time_frequency import tfr_multitaper
from meya.loadData_YML import getData_chanSeq_OpenBMI
from sklearn.utils import shuffle
CONSTANT = CONSTANT['OpenBMI']

def read_raw(PATH, session, subject,num_class,yml,channelDic,eventTypeDic,func):
    mat_file_name = PATH + '/sess'+str(session).zfill(2)+'_subj'+str(subject).zfill(2)+'_EEG_MI.mat'
    mat = sio.loadmat(mat_file_name)
    print('This is data from: ', mat_file_name)
    #loda raw event
    raw_train, events_train, sub_dic1= getRawEvent(mat, key='EEG_MI_train', yml=yml)

    raw_test, events_test ,sub_dic2= getRawEvent(mat, key='EEG_MI_test', yml=yml)

    return raw_train, events_train, raw_test, events_test,sub_dic1,sub_dic2

def read_ERDS(PATH, session, subject, num_class,yml,channelDic,eventTypeDic,func):
    mat_file_name = PATH + '/sess'+str(session).zfill(2)+'_subj'+str(subject).zfill(2)+'_EEG_MI.mat'
    mat = sio.loadmat(mat_file_name)
    print('This is data from: ', mat_file_name)
    tmin = yml['Meta']['tmin']
    tmax = yml['Meta']['tmax']

    if num_class == 2:
        raw, events = getRawEvent(mat, key='EEG_MI_train', yml=yml)##读取raw event
        picks = mne.pick_channels(raw.info["ch_names"],["C3","Cz","C4"])
        events_ids = dict(right=1,left=2)
        epochs = mne.Epochs(raw,events,events_ids,tmin=tmin,tmax=tmax+0.5,picks=picks,baseline=None,preload=True)
        freqs = np.arange(4,61)
        n_cycles = freqs
        baseline = [-1,0]
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,use_fft=True, return_itc=False, average=False,decim=1)
        tfr.crop(0,4)
        tfr.apply_baseline(baseline,mode='percent')
        erds = tfr.data[0]
        for i in range(1,100):
            erds = np.concatenate((erds,tfr.data[i]),axis=-1)

        #生成Event
        event_time = []
        start = 0
        for i in range(100):  #需要判断时间点对应的数值是否为0
            event_time.append(start)
            start = start+4001
        #测试
        for i in event_time:
            for j in range(57):
                for k in  range(3):
                   if erds[k,j,i] == 0:
                      continue
                   else:
                       print('Error')

        label = np.array(tfr.events[:,2])#需要判断是否与原数据相同
        #测试
        for i in range(100):
            if label[i] == tfr.events[:,2][i]:
                continue
            else:
                print("error")
        duration = np.zeros((100,),dtype=int)
        eventData = np.vstack((event_time, duration, label)).T
        # eventData = np.r_[[[0, 0, 0]], eventData]
        events = np.array(eventData, dtype='int32')
        # func:loadData_YML:getData_chanSeq
        raw_train_data, label_train_data = func(yml, erds, events, channelDic, eventTypeDic, getAll=False)

        raw_test, events_test = getRawEvent(mat, key='EEG_MI_test', yml=yml)
        events_test = np.array(events_test, dtype='int32')
        raw_test_data, label_test_data = func(yml, raw_test, events_test, channelDic, eventTypeDic, getAll=False)
        return raw_train_data, label_train_data, raw_test_data, label_test_data

    elif num_class == 3:
        raw_train,events_train = getRawEvent(mat,key='EEG_MI_train',yml=yml)
        events_train = np.array(events_train,dtype='int32')
        #func:loadData_YML:getData_chanSeq
        raw_train_data,label_train_data = func(yml,raw_train,events_train,channelDic,eventTypeDic,getAll=False)

        raw_test,events_test = getRawEvent(mat,key='EEG_MI_test',yml=yml)
        events_test = np.array(events_test,dtype='int32')
        raw_test_data, label_test_data = func(yml, raw_test, events_test, channelDic, eventTypeDic, getAll=False)
        return raw_train_data, label_train_data, raw_test_data, label_test_data


def load_filter_data(load_path,sub,sess):
    try:
        file_x_train = load_path + '/X_train_S{:03d}_sess{:03d}.npy'.format(sub, sess)
        file_y_train = load_path + '/y_train_S{:03d}_sess{:03d}.npy'.format(sub, sess)
        X_train = np.load(file_x_train, allow_pickle=True)#1456 31 1000
        y_train = np.load(file_y_train, allow_pickle=True)
        # y_train = np.array(y_train,dtype=int)

        file_x_te = load_path + '/X_test_S{:03d}_sess{:03d}.npy'.format(sub, sess)
        file_y_te = load_path + '/y_test_S{:03d}_sess{:03d}.npy'.format(sub, sess)
        X_test = np.load(file_x_te, allow_pickle=True)
        y_test = np.load(file_y_te, allow_pickle=True)
        # y_test = np.array(y_test, dtype=int)
        print('OfflineRaw shape{},OfflineEvents shape{},OnlineRaw shape{},OfflineEvents shape{}'.format(X_train.shape, y_train.shape,X_test.shape,y_test.shape))
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x_train, file_y_train))
    return X_train, y_train,X_test,y_test

def load_crop_data(PATH, n_subjs, chanDic,TrainType,num_class,yml):
    if TrainType == 'dependent':
        pass
        # if num_class == 2:
        #     print("Tow-class dependent MI data is Process")
        #     sessions = [1, 2]
        #     X_sub_tra, y_sub_tra,X_sub_te,y_sub_te = [], [],[],[]
        #     for subject in range(1,n_subjs+1):
        #         x_session_tra, y_session_tra, x_session_te,y_session_te= [],[],[],[]
        #         for id_se, sess in enumerate(sessions):
        #             X_tr, y_tr, X_te, y_te = load_filter_data(PATH, subject, sess)
        #             X_tr_select, y_tr_select, X_te_select, y_te_select = selectData(num_class,X_tr, y_tr, X_te, y_te)
        #             # （54,2,150,20,400）(channel,windowsSize)
        #             x_session_tra.extend(X_tr_select)
        #             x_session_te.extend(X_te_select)
        #             # （54,2,）
        #             y_session_tra.extend(y_tr_select)
        #             y_session_te.extend(y_te_select)
        #
        #         x_session_tra = np.array(x_session_tra,dtype='float32')
        #         x_session_te = np.array(x_session_te, dtype='float32')
        #         y_session_tra = np.array(y_session_tra, dtype='int32')
        #         y_session_te = np.array(y_session_te, dtype='int32')
        #         X_sub_tra.append(x_session_tra)
        #         y_sub_tra.append(y_session_tra)
        #         X_sub_te.append(x_session_te)
        #         y_sub_te.append(y_session_te)
        #     # (sub,trail,channel,sampleing)
        #     X_sub_tra = np.array(X_sub_tra, dtype='float32')
        #     y_sub_tra = np.array(y_sub_tra,dtype='int32')
        #     X_sub_te = np.array(X_sub_te, dtype='float32')
        #     y_sub_te = np.array(y_sub_te, dtype='int32')
        #     #交换维度
        #     X_sub_tra = np.swapaxes(X_sub_tra, -1, -2)
        #     X_sub_te = np.swapaxes(X_sub_te, -1, -2)
        #     return X_sub_tra, y_sub_tra, X_sub_te, y_sub_te
        # elif num_class == 3:
        #     print("不支持生成对象内三分类数据")
    elif TrainType == 'independent':
        if num_class == 3:
            eventTypeDic = {
                0: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 3000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
            print("Three-class Independnt MI data is Process")
            sessions = [1, 2]
            X_sub, y_sub = [], []
            for subject in n_subjs:
                x_session,y_session = [],[]
                for id_se, sess in enumerate(sessions):
                    offline_select_event, online_select_event = [], []
                    offlineRaw, offlineEvent, onlineRaw, onlineEvent = load_filter_data(PATH, subject, sess)
                    # 1.离线数据选择events
                    offline_rest_Index = np.where(offlineEvent[:, 2] == 0)[0]
                    offline_right_Index = np.where(offlineEvent[:, 2] == 1)[0]
                    offline_left_Index = np.where(offlineEvent[:, 2] == 2)[0]
                    offline_event_Index = np.sort(np.concatenate((offline_left_Index, offline_right_Index,offline_rest_Index)))
                    for i in offline_event_Index:
                        offline_select_event.append(offlineEvent[i])
                    offline_select_event = np.array(offline_select_event, dtype='int32')
                    # 2.切割数据
                    offlineData, offlineLabel = getData_chanSeq_OpenBMI(yml, offlineRaw, offline_select_event, chanDic,
                                                                        eventTypeDic=eventTypeDic, getAll=False)
                    offlineData, offlineLabel = shuffle(offlineData, offlineLabel)

                    # 1.在线数据选择events
                    online_right_Index = np.where(onlineEvent[:, 2] == 1)[0]
                    online_left_Index = np.where(onlineEvent[:, 2] == 2)[0]
                    online_event_Index = np.sort(np.concatenate((online_left_Index, online_right_Index)))
                    for i in online_event_Index:
                        online_select_event.append(onlineEvent[i])
                    online_select_event = np.array(online_select_event, dtype='int32')
                    # 2.切割数据
                    onlineData, onlineLabel = getData_chanSeq_OpenBMI(yml, onlineRaw, online_select_event, chanDic,
                                                                      eventTypeDic=eventTypeDic, getAll=False)
                    onlineData, onlineLabel = shuffle(onlineData, onlineLabel)
                    # 3.合并
                    SessionData = np.concatenate((offlineData, onlineData), axis=0)
                    SessionLabel = np.concatenate((offlineLabel, onlineLabel), axis=0)
                    # （700*4,31,1000）
                    x_session.extend(SessionData)
                    y_session.extend(SessionLabel)
                X_sub.extend(x_session)
                y_sub.extend(y_session)
            X_sub = np.array(X_sub, dtype='float32')  # (sub*trail,channel,timepoint)
            y_sub = np.array(y_sub, dtype='int32')  # (sub*trail,2)
            X_sub = np.swapaxes(X_sub, -1, -2)
            # -1表示自适应
            return X_sub, y_sub
        elif num_class == 2:
            eventTypeDic = {
                        1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                        2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                           }
            print("Tow-class Independent MI data is Process")
            sessions = [1, 2]
            X_sub, y_sub = [], []
            for subject in n_subjs:
                x_session, y_session = [], []
                for id_se, sess in enumerate(sessions):
                    offline_select_event,online_select_event = [],[]
                    offlineRaw, offlineEvent, onlineRaw, onlineEvent = load_filter_data(PATH, subject, sess)
                    #1.离线数据选择events
                    offline_right_Index = np.where(offlineEvent[:, 2] == 1)[0]
                    offline_left_Index = np.where(offlineEvent[:, 2] == 2)[0]
                    offline_event_Index = np.sort(np.concatenate((offline_left_Index, offline_right_Index)))
                    for i in offline_event_Index:
                        offline_select_event.append(offlineEvent[i])
                    offline_select_event = np.array(offline_select_event, dtype='int32')
                    #2.切割数据
                    offlineData, offlineLabel = getData_chanSeq_OpenBMI(yml, offlineRaw, offline_select_event, chanDic,
                                                                     eventTypeDic=eventTypeDic, getAll=False)
                    offlineData,offlineLabel = shuffle(offlineData,offlineLabel)

                    # 1.在线数据选择events
                    online_right_Index = np.where(onlineEvent[:, 2] == 1)[0]
                    online_left_Index = np.where(onlineEvent[:, 2] == 2)[0]
                    online_event_Index = np.sort(np.concatenate((online_left_Index, online_right_Index)))
                    for i in online_event_Index:
                        online_select_event.append(onlineEvent[i])
                    online_select_event = np.array(online_select_event, dtype='int32')
                    # 2.切割数据
                    onlineData, onlineLabel = getData_chanSeq_OpenBMI(yml, onlineRaw, online_select_event, chanDic,
                                                                        eventTypeDic=eventTypeDic, getAll=False)
                    onlineData, onlineLabel = shuffle(onlineData, onlineLabel)
                    #3.合并
                    SessionData = np.concatenate((offlineData,onlineData),axis=0)
                    SessionLabel = np.concatenate((offlineLabel,onlineLabel),axis=0)
                    # （700*4,31,1000）
                    x_session.extend(SessionData)
                    y_session.extend(SessionLabel)
                X_sub.extend(x_session)
                y_sub.extend(y_session)
            X_sub = np.array(X_sub, dtype='float32')  #(sub*trail,channel,timepoint)
            y_sub = np.array(y_sub, dtype='int32')  #(sub*trail,2)
            X_sub = np.swapaxes(X_sub,-1,-2)
            # -1表示自适应
            return X_sub, y_sub

def load_crop_data_FBCNet(PATH, n_subjs, chanDic,TrainType,num_class,yml):
    if TrainType == 'dependent':
        pass
        # if num_class == 2:
        #     print("Tow-class dependent MI data is Process")
        #     sessions = [1, 2]
        #     X_sub_tra, y_sub_tra,X_sub_te,y_sub_te = [], [],[],[]
        #     for subject in range(1,n_subjs+1):
        #         x_session_tra, y_session_tra, x_session_te,y_session_te= [],[],[],[]
        #         for id_se, sess in enumerate(sessions):
        #             X_tr, y_tr, X_te, y_te = load_filter_data(PATH, subject, sess)
        #             X_tr_select, y_tr_select, X_te_select, y_te_select = selectData(num_class,X_tr, y_tr, X_te, y_te)
        #             # （54,2,150,20,400）(channel,windowsSize)
        #             x_session_tra.extend(X_tr_select)
        #             x_session_te.extend(X_te_select)
        #             # （54,2,）
        #             y_session_tra.extend(y_tr_select)
        #             y_session_te.extend(y_te_select)
        #
        #         x_session_tra = np.array(x_session_tra,dtype='float32')
        #         x_session_te = np.array(x_session_te, dtype='float32')
        #         y_session_tra = np.array(y_session_tra, dtype='int32')
        #         y_session_te = np.array(y_session_te, dtype='int32')
        #         X_sub_tra.append(x_session_tra)
        #         y_sub_tra.append(y_session_tra)
        #         X_sub_te.append(x_session_te)
        #         y_sub_te.append(y_session_te)
        #     # (sub,trail,channel,sampleing)
        #     X_sub_tra = np.array(X_sub_tra, dtype='float32')
        #     y_sub_tra = np.array(y_sub_tra,dtype='int32')
        #     X_sub_te = np.array(X_sub_te, dtype='float32')
        #     y_sub_te = np.array(y_sub_te, dtype='int32')
        #     #交换维度
        #     X_sub_tra = np.swapaxes(X_sub_tra, -1, -2)
        #     X_sub_te = np.swapaxes(X_sub_te, -1, -2)
        #     return X_sub_tra, y_sub_tra, X_sub_te, y_sub_te
        # elif num_class == 3:
        #     print("不支持生成对象内三分类数据")
    elif TrainType == 'independent':
        if num_class == 3:
            eventTypeDic = {
                0: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 3000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
            print("Three-class Independnt MI data is Process")
            sessions = [1, 2]
            X_sub, y_sub = [], []
            for subject in n_subjs:
                x_session,y_session = [],[]
                for id_se, sess in enumerate(sessions):
                    offline_select_event, online_select_event = [], []
                    offlineRaw, offlineEvent, onlineRaw, onlineEvent = load_filter_data(PATH, subject, sess)
                    # 1.离线数据选择events
                    offline_rest_Index = np.where(offlineEvent[:, 2] == 0)[0]
                    offline_right_Index = np.where(offlineEvent[:, 2] == 1)[0]
                    offline_left_Index = np.where(offlineEvent[:, 2] == 2)[0]
                    offline_event_Index = np.sort(np.concatenate((offline_left_Index, offline_right_Index,offline_rest_Index)))
                    for i in offline_event_Index:
                        offline_select_event.append(offlineEvent[i])
                    offline_select_event = np.array(offline_select_event, dtype='int32')
                    # 2.切割数据
                    offlineData, offlineLabel = getData_chanSeq_OpenBMI(yml, offlineRaw, offline_select_event, chanDic,
                                                                        eventTypeDic=eventTypeDic, getAll=False)
                    offlineData, offlineLabel = shuffle(offlineData, offlineLabel)

                    # 1.在线数据选择events
                    online_right_Index = np.where(onlineEvent[:, 2] == 1)[0]
                    online_left_Index = np.where(onlineEvent[:, 2] == 2)[0]
                    online_event_Index = np.sort(np.concatenate((online_left_Index, online_right_Index)))
                    for i in online_event_Index:
                        online_select_event.append(onlineEvent[i])
                    online_select_event = np.array(online_select_event, dtype='int32')
                    # 2.切割数据
                    onlineData, onlineLabel = getData_chanSeq_OpenBMI(yml, onlineRaw, online_select_event, chanDic,
                                                                      eventTypeDic=eventTypeDic, getAll=False)
                    onlineData, onlineLabel = shuffle(onlineData, onlineLabel)
                    # 3.合并
                    SessionData = np.concatenate((offlineData, onlineData), axis=0)
                    SessionLabel = np.concatenate((offlineLabel, onlineLabel), axis=0)
                    # （700*4,31,1000）
                    x_session.extend(SessionData)
                    y_session.extend(SessionLabel)
                X_sub.extend(x_session)
                y_sub.extend(y_session)
            X_sub = np.array(X_sub, dtype='float32')  # (sub*trail,channel,timepoint)
            y_sub = np.array(y_sub, dtype='int32')  # (sub*trail,2)
            X_sub = np.swapaxes(X_sub, -1, -2)
            # -1表示自适应
            return X_sub, y_sub
        elif num_class == 2:
            eventTypeDic = {
                        1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                        2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                           }
            print("Tow-class Independent MI data is Process")
            sessions = [1, 2]
            X_sub, y_sub = [], []
            for subject in n_subjs:
                x_session, y_session = [], []
                for id_se, sess in enumerate(sessions):
                    offlineRaw, offlineEvent, onlineRaw, onlineEvent = load_filter_data(PATH, subject, sess)
                    multiofflinedata,multiofflinelabel = [],[]
                    for filternum,(offdata,offlabel) in enumerate(zip(offlineRaw,offlineEvent)):
                        offline_select_event = []
                        #1.离线数据选择events
                        offline_right_Index = np.where(offlabel[:, 2] == 1)[0]
                        offline_left_Index = np.where(offlabel[:, 2] == 2)[0]
                        offline_event_Index = np.sort(np.concatenate((offline_left_Index, offline_right_Index)))
                        for i in offline_event_Index:
                            offline_select_event.append(offlabel[i])
                        offline_select_event = np.array(offline_select_event, dtype='int32')

                        #2.切割数据
                        offlineData, offlineLabel = getData_chanSeq_OpenBMI(yml, offdata, offline_select_event, chanDic,
                                                                         eventTypeDic=eventTypeDic, getAll=False)
                        # offlineData,offlineLabel = shuffle(offlineData,offlineLabel)
                        multiofflinedata.append(offlineData)
                        if filternum == 0:
                            multiofflinelabel = offlineLabel
                    multionlinedata, multionlinelabel = [], []
                    for filternum,(ondata,onlabel) in enumerate(zip(onlineRaw,onlineEvent)):
                        online_select_event = []
                        # 1.在线数据选择events
                        online_right_Index = np.where(onlabel[:, 2] == 1)[0]
                        online_left_Index = np.where(onlabel[:, 2] == 2)[0]
                        online_event_Index = np.sort(np.concatenate((online_left_Index, online_right_Index)))
                        for i in online_event_Index:
                            online_select_event.append(onlabel[i])
                        online_select_event = np.array(online_select_event, dtype='int32')

                        # 2.切割数据
                        onlineData, onlineLabel = getData_chanSeq_OpenBMI(yml, ondata, online_select_event, chanDic,
                                                                            eventTypeDic=eventTypeDic, getAll=False)
                        # onlineData, onlineLabel = shuffle(onlineData, onlineLabel)
                        multionlinedata.append(onlineData)
                        if filternum == 0:
                            multionlinelabel = onlineLabel

                    multiofflinedata = np.array(multiofflinedata,dtype='float32').swapaxes(0, 1)
                    multionlinedata = np.array(multionlinedata, dtype='float32').swapaxes(0, 1)
                    #3.合并
                    SessionData = np.concatenate((multiofflinedata,multionlinedata),axis=0)
                    SessionLabel = np.concatenate((multiofflinelabel,multionlinelabel),axis=0)
                    # （700*4,31,1000）
                    x_session.extend(SessionData)
                    y_session.extend(SessionLabel)
                X_sub.extend(x_session)
                y_sub.extend(y_session)
            X_sub = np.array(X_sub, dtype='float32')  #(sub*trail,channel,timepoint)
            y_sub = np.array(y_sub, dtype='int32')  #(sub*trail,2)
            X_sub = np.swapaxes(X_sub,-1,-2)
            # -1表示自适应
            return X_sub, y_sub

                
def chanel_selection(sel_chs):
    orig_chs = CONSTANT['orig_chs']
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch ,'---', 'Index_is:', ch_id)
    return chs_id

def __segment_data(mat_arr, type_data):
    data = mat_arr['EEG_MI_'+type_data][0]['x'][0]
    t = mat_arr['EEG_MI_'+type_data][0]['t'][0][0]
    low_cut = 0
    high_cut = CONSTANT['trial_len'] #8s
    orig_smp_freq = CONSTANT['orig_smp_freq'] #1000
    orig_n_chs = CONSTANT['n_chs'] #62
    n_trials = CONSTANT['n_trials_2_class'] #100
    #(100,8000,62)
    data_seg = np.zeros((n_trials, high_cut*orig_smp_freq, orig_n_chs))
    # print('This pre-processing is for task {} low cut {} high cut {}'.format(task,low_cut,high_cut))
    for i in range(n_trials):
        start_pos = t[i]+(low_cut*orig_smp_freq)
        stop_pos = t[i]+(high_cut*orig_smp_freq)
        data_seg[i, :, :] = data[start_pos:stop_pos, :]
    return data_seg

def __segment_data_whole_period(mat_arr, type_data):
    data = mat_arr['EEG_MI_'+type_data][0]['x'][0]
    t = mat_arr['EEG_MI_'+type_data][0]['t'][0][0]
    low_cut = -3
    high_cut = CONSTANT['trial_len'] #8s
    data_len = 11
    orig_smp_freq = CONSTANT['orig_smp_freq'] #1000
    orig_n_chs = CONSTANT['n_chs'] #62
    n_trials = CONSTANT['n_trials_2_class'] #100
    data_seg = np.zeros((n_trials, data_len*orig_smp_freq, orig_n_chs))
    # print('This pre-processing is for task {} low cut {} high cut {}'.format(task,low_cut,high_cut))
    for i in range(n_trials):
        start_pos = t[i]+(low_cut*orig_smp_freq)
        stop_pos = t[i]+(high_cut*orig_smp_freq)
        # print("Debugg the selected period is:", (stop_pos-start_pos)/orig_smp_freq)
        data_seg[i, :, :] = data[start_pos:stop_pos, :]
    return data_seg

def __add_on_resting(X, y, smp_freq):
    print("MI Right, MI Left and Resting EEG Segmentation Process is being processed...")
    print("This data contains {} time ponts with sampling frequency of {} Hz.".format(X.shape[2], smp_freq))
    start_pos_mi = int(CONSTANT['MI']['start']*smp_freq) #0s
    stop_pos_mi = int(CONSTANT['MI']['stop']*smp_freq) #4s
    start_pos_rest = int(CONSTANT['MI']['stop']*smp_freq) #4s
    stop_pos_rest = int(CONSTANT['trial_len']*smp_freq) #8s
    index_class1 = np.where(y == 0)[0]
    index_class2 = np.where(y == 1)[0]
    X_class1, y_class1 = X[index_class1], y[index_class1]
    X_class2, y_class2 = X[index_class2], y[index_class2]
    # Split data into resting and MI signals
    X_mi_class1 = X_class1[:,:,start_pos_mi:stop_pos_mi]#（50,20,4000）
    X_mi_class2 = X_class2[:,:,start_pos_mi:stop_pos_mi]#（50,20,4000）
    X_rest_class1 = X_class1[:,:,start_pos_rest:stop_pos_rest]#（50,20,4000）
    X_rest_class2 = X_class2[:,:,start_pos_rest:stop_pos_rest]#（50,20,4000）
    # Choose a half of resting data to keep balancing the number of classes in our data
    X_rest_class1_50per,_ ,_ ,_= train_test_split(X_rest_class1, y_class1, random_state=42, test_size=0.5)
    X_rest_class2_50per,_ ,_ ,_= train_test_split(X_rest_class2, y_class2, random_state=42, test_size=0.5)
    X_rest_all = np.concatenate((X_rest_class1_50per, X_rest_class2_50per), axis=0)
    # Build class for resting data np.full为填充函数 用2填充满大小为参数1的矩阵
    y_rest_all = np.full(X_rest_all.shape[0], 2)
    # Combine all classes again
    X_new_all = np.concatenate((X_mi_class1, X_mi_class2, X_rest_all), axis=0)
    y_new_all = np.concatenate((y_class1, y_class2, y_rest_all), axis=0)
    return X_new_all, y_new_all

def __transitory_mi(X, y, smp_freq, start, stop):
    print("MI Right ang MI Left EEG including transitory period is being processed...")
    print("This data contains {} time ponts with sampling frequency of {} Hz.".format(X.shape[2], smp_freq))
    start_pos_mi = int(start*smp_freq)
    stop_pos_mi = int(stop*smp_freq)
    # Segment needed MI period
    X_mi = X[:,:,start_pos_mi:stop_pos_mi]
    return X_mi, y
