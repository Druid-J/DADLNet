import MI
from MI.config import CONSTANT
import sys
import yaml
import os
# from meya_load_OpenBMI.loadData import get_31MI_1D as getChannels,loadData,getEvents
from meya.loadData_YML import getErdsData_chanSeq as getErdsData

k_folds = 5


# save_path = 'datasets'
# save_path = '/data0/meya/code/MIN2Net_code/experiments/datasets'
# save_path = '/home/xumeiyan/Public/Data/MI'
# ProcessDataSavePath = '/home/xumeiyan/Public/Data/MI/OpenBMI/filterData'
num_class = 2

if __name__ =='__main__':
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    for pkg,function in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(function)
        exec (stri)

    channDic = getChannels()
    # eventTypeDic = getEvents()
    eventTypeDic = {1:{'Name':'right','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                    2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }

    pick_smp_freq = yml['Meta']['downSampling']
    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    ClassNum = yml['Meta']['ClassNum']
    Channel_format = yml['Meta']['Channel']

    RawDataPath = BasePath+"/raw"
    # ProcessDataSavePath = BasePath+"/ProcessData/"
    ProcessDataSavePath = BasePath + "/ProcessData/ERDS/{}Hz_{}chan".format(pick_smp_freq,Channel_format)
    if not os.path.exists(ProcessDataSavePath):
        os.makedirs(ProcessDataSavePath)
    # save_path = BasePath+'/Traindata'
    save_path = BasePath + '/Traindata/ERDS/{}/{}_class/{}Hz_{}chan'.format(TrainType, ClassNum, pick_smp_freq,Channel_format)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    MI.preprocess.OpenBMI.time_domain.Subject_session_ERDsDataGenerate(RawDataPath, ProcessDataSavePath, num_class, yml, channDic, eventTypeDic, func=getErdsData)




    # prep.BCIC2a.time_domain.subject_dependent_setting(k_folds=k_folds,
    #                                                   pick_smp_freq=pick_smp_freq,
    #                                                   bands=bands,
    #                                                   order=order,
    #                                                   save_path=save_path,
    #                                                   num_class=num_class,
    #                                                   sel_chs=CONSTANT['BCIC2a']['sel_chs'])
    #
    # prep.OpenBMI.time_domain.subject_dependent_setting(k_folds=k_folds,
    #                                                    pick_smp_freq=pick_smp_freq,
    #                                                    bands=bands,
    #                                                    order=order,
    #                                                    save_path=save_path,
    #                                                    num_class=num_class,
    #                                                    sel_chs=CONSTANT['OpenBMI']['sel_chs'])
    #
    #

    # prep.BCIC2a.time_domain.subject_independent_setting(k_folds=k_folds,
    #                                                     pick_smp_freq=pick_smp_freq,
    #                                                     bands=bands,
    #                                                     order=order,
    #                                                     save_path=save_path,
    #                                                     num_class=num_class,
    #                                                     sel_chs=CONSTANT['BCIC2a']['sel_chs'])

    # prep.OpenBMI.time_domain.subject_independent_setting_spilt(yml=yml,
    #                                                      pick_smp_freq=pick_smp_freq,
    #                                                      ProDataPath=ProcessDataSavePath,
    #                                                      save_path=save_path,
    #                                                      num_class=num_class,)

    MI.preprocess.OpenBMI.Time_domain.time_domain.subject_dependent_setting_spilt(yml=yml,
                                                                                  k_folds=k_folds,
                                                                                  pick_smp_freq=pick_smp_freq,
                                                                                  ProDataPath=ProcessDataSavePath,
                                                                                  save_path=save_path,
                                                                                  num_class=num_class,
                                                                                  sel_chs=CONSTANT['OpenBMI']['sel_chs'])
