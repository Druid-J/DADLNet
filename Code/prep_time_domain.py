import MI
from MNUMI.preprocess.BCI2008 import time_domain
import sys
import yaml
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from MI.loadData import getEvents
from meya.loadData_YML import getData_chanSeq as getData
from meya.fileAction import saveFile

k_folds = 5


if __name__ =='__main__':
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    for pkg,function in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(function)
        exec (stri)

    channDic = getChannels()
    pick_smp_freq = yml['Meta']['downSampling']
    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    num_class = yml['Meta']['ClassNum']
    Channel_format = yml['Meta']['Channel']
    n_subjs= yml['Meta']['subsample']
    minFreq = yml['Meta']['minFreq']
    maxFreq = yml['Meta']['maxFreq']
    Datasets = yml['Meta']['Datasets']
    step = yml['Meta']['step']
    eventTypeDic = getEvents(Datasets,num_class)


    RawDataPath = BasePath+"/raw"
    ProcessDataSavePath = BasePath + "/DADLNet/ProcessData/{}Hz_{}chan_{}_{}Hz_{}_DeepConv".format(pick_smp_freq,Channel_format,minFreq,maxFreq,step)
    if not os.path.exists(ProcessDataSavePath):
        os.makedirs(ProcessDataSavePath)
    save_path = BasePath + '/DADLNet/Traindata/{}/{}_class/{}Hz_{}chan_{}_{}Hz_{}_DeepConv'.format(TrainType, num_class, pick_smp_freq,Channel_format,minFreq,maxFreq,step)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_count = os.listdir(ProcessDataSavePath)

    saveFile(save_path, sys.argv[1], cover=True)

    if Datasets =='OpenBMI':
        #目前只是简单判断 处理数据是否生成
        if len(file_count) !=433:
            '生成ProcessData'
            MI.preprocess.OpenBMI.time_domain.Subject_session_DataGenerate(RawDataPath, ProcessDataSavePath, num_class,n_subjs, yml, channDic, eventTypeDic, func=getData)

        if TrainType == 'dependent':
            '对象内数据生成'
            MI.preprocess.OpenBMI.time_domain.subject_dependent_setting_spilt(yml=yml,
                                                                          k_folds=k_folds,
                                                                          chanDic=channDic,
                                                                          ProDataPath=ProcessDataSavePath,
                                                                          save_path=save_path,
                                                                          num_class=num_class,
                                                                          n_subjs=n_subjs)
        elif TrainType == 'independent':
            '跨对象的数据生成'
            MI.preprocess.OpenBMI.time_domain.subject_independent_setting_spilt(yml=yml,
                                                                 chanDic = channDic,
                                                                 ProDataPath=ProcessDataSavePath,
                                                                 save_path=save_path,
                                                                 num_class=num_class,
                                                                 n_subjs=n_subjs)
    elif Datasets =='BCIC2a':
        if len(file_count) != 36:
            'BCI2008数据生成方法'
            time_domain.Subject_session_DataGenerate(RawDataPath, ProcessDataSavePath, num_class, n_subjs,yml)

        if TrainType == 'dependent':
            '对象内数据生成'
            time_domain.subject_dependent_setting_spilt(yml=yml,
                                                    k_folds=k_folds,
                                                    ProDataPath=ProcessDataSavePath,
                                                    save_path=save_path,
                                                    num_class=num_class,
                                                    n_subjs=n_subjs)

        elif TrainType =='independent':
            "跨对象数据生成"
            time_domain.subject_independent_setting_spilt(yml=yml,
                                                        ProDataPath=ProcessDataSavePath,
                                                        save_path=save_path,
                                                        num_class=num_class,
                                                        n_subjs=n_subjs)

    elif Datasets == "MNNUData":
        "MNNUData process数据生成"
        MI.preprocess.MnnuData.time_domain.Subject_session_DataGenerate(ProcessDataSavePath, yml, eventTypeDic)

        if TrainType == 'dependent':
            '对象内数据生成'
            MI.preprocess.MnnuData.time_domain.subject_dependent_setting_spilt(yml=yml,
                                                    k_folds=k_folds,
                                                    ProDataPath=ProcessDataSavePath,
                                                    save_path=save_path,
                                                    num_class=num_class,
                                                    n_subjs=n_subjs)
        elif TrainType =='independent':
            MI.preprocess.MnnuData.time_domain.subject_independent_setting_spilt(yml=yml,
                                                        chanDic=channDic,
                                                        ProDataPath=ProcessDataSavePath,
                                                        save_path=save_path,
                                                        num_class=num_class)


