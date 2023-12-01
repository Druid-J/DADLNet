import numpy as np
from sklearn.utils import shuffle
import scipy.io as scio
from mne.io import RawArray
import sys
sys.path.append('..')
import os
from meya.fileAction import joinPath,getRawDataFromFile,saveRawDataToFile
import mne
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

def get_index_channel():
    return {1:'Fp1',2:'Fp2',3:'F7',4:'F3',5:'Fz',6:'F4',7:'F8',8:'FC5',9:'FC1',10:'FC2',11:'FC6',12:'T7',13:'C3',14:'Cz',15:'C4',16:'T8',
17:'TP9',18:'CP5',19:'CP1',20:'CP2',21:'CP6',22:'TP10',23:'P7',24:'P3',25:'Pz',26:'P4',27:'P8',28:'PO9',29:'O1',30:'Oz',
31:'O2',32:'PO10',33:'FC3',34:'FC4',35:'C5',36:'C1',37:'C2',38:'C6',39:'CP3',40:'CPz',41:'CP4',42:'P1',43:'P2',44:'POz',
45:'FT9',46:'FTT9h',47:'TPP7h',48:'TP7',49:'TPP9h',50:'FT10',51:'FTT10h',52:'FTT8h',53:'TP8',54:'TPP10h',55:'F9',56:'F10',
57:'AF7',58:'AF3',59:'AF4',60:'AF8',61:'PO3',62:'PO4'}

def get_index_MIchannel():
    return {8:'FC5',9:'FC1',10:'FC2',11:'FC6',13:'C3',14:'Cz',15:'C4',18:'CP5',19:'CP1',20:'CP2',21:'CP6',24:'P3',25:'Pz',26:'P4'
        ,33:'FC3',34:'FC4',35:'C5',36:'C1',37:'C2',38:'C6',39:'CP3',40:'CPz',41:'CP4',42:'P1',43:'P2'}

def get_channel_index():
    return {'Fp1':1,'Fp2':2,'F7':3,'F3':4,'Fz':5,'F4':6,'F8':7,'FC5':8,'FC1':9,'FC2':10,'FC6':11,'T7':12,'C3':13,'Cz':14,'C4':15,'T8':16,
'TP9':17,'CP5':18,'CP1':19,'CP2':20,'CP6':21,'TP10':22,'P7':23,'P3':24,'Pz':25,'P4':26,'P8':27,'PO9':28,'O1':29,'Oz':30,
'O2':31,'PO10':32,'FC3':33,'FC4':34,'C5':35,'C1':36,'C2':37,'C6':38,'CP3':39,'CPz':40,'CP4':41,'P1':42,'P2':43,'POz':44,
'FT9':45,'FTT9h':46,'TPP7h':47,'TP7':48,'TPP9h':49,'FT10':50,'FTT10h':51,'FTT8h':52,'TP8':53,'TPP10h':54,'F9':55,'F10':56,
'AF7':57,'AF3':58,'AF4':59,'AF8':60,'PO3':61,'PO4':62}

def getChannels():
    return {8:'FC5',33:'FC3',9:'FC1',10:'FC2',34:'FC4',11:'FC6',
            35:'C5',13:'C3',36:'C1',14:'Cz',37:'C2',15:'C4',38:'C6',
            18:'CP5',39:'CP3',19:'CP1',40:'CPz',20:'CP2',41:'CP4',21:'CP6'}
def getChannels_C3C4():
    return {13:'C3',15:'C4'}
def getChannels_C3C4Cz():
    return {13:'C3',15:'C4',14:'Cz'}
def getChannels_1D_C3C4Cz():
    return {13:{'Name':'C3','Loca':[1]},14:{'Name':'Cz','Loca':[2]},15:{'Name':'C4','Loca':[3]}}
def getChannel_1D():
    return {1:{'Name':'Fp1','Loca':[16]},2:{'Name':'Fp2','Loca':[35]},3:{'Name':'F7','Loca':[7]},4:{'Name':'F3','Loca':[18]},5:{'Name':'Fz','Loca':[29]},6:{'Name':'F4','Loca':[37]},7:{'Name':'F8','Loca':[49]},8:{'Name':'FC5','Loca':[13]},9:{'Name':'FC1','Loca':[25]},10:{'Name':'FC2','Loca':[38]},11:{'Name':'FC6','Loca':[50]},12:{'Name':'T7','Loca':[8]},13:{'Name':'C3','Loca':[20]},14:{'Name':'Cz','Loca':[30]},15:{'Name':'C4','Loca':[44]},16:{'Name':'T8','Loca':[55]},
17:{'Name':'TP9','Loca':[3]},18:{'Name':'CP5','Loca':[15]},19:{'Name':'CP1','Loca':[27]},20:{'Name':'CP2','Loca':[40]},21:{'Name':'CP6','Loca':[52]},22:{'Name':'TP10','Loca':[61]},23:{'Name':'P7','Loca':[10]},24:{'Name':'P3','Loca':[22]},25:{'Name':'Pz','Loca':[32]},26:{'Name':'P4','Loca':[46]},27:{'Name':'P8','Loca':[57]},28:{'Name':'PO9','Loca':[4]},29:{'Name':'O1','Loca':[24]},30:{'Name':'Oz','Loca':[34]},
31:{'Name':'O2','Loca':[48]},32:{'Name':'PO10','Loca':[62]},33:{'Name':'FC3','Loca':[19]},34:{'Name':'FC4','Loca':[43]},35:{'Name':'C5','Loca':[14]},36:{'Name':'C1','Loca':[26]},37:{'Name':'C2','Loca':[39]},38:{'Name':'C6','Loca':[51]},39:{'Name':'CP3','Loca':[21]},40:{'Name':'CPz','Loca':[31]},41:{'Name':'CP4','Loca':[45]},42:{'Name':'P1','Loca':[28]},43:{'Name':'P2','Loca':[41]},44:{'Name':'POz','Loca':[33]},
45:{'Name':'FT9','Loca':[2]},46:{'Name':'FTT9h','Loca':[6]},47:{'Name':'FTT7h','Loca':[11]},48:{'Name':'TP7','Loca':[9]},49:{'Name':'FTT9h','Loca':[5]},50:{'Name':'FT10','Loca':[60]},51:{'Name':'FTT10h','Loca':[58]},52:{'Name':'FTT8h','Loca':[53]},53:{'Name':'TP8','Loca':[56]},54:{'Name':'TPP10h','Loca':[59]},55:{'Name':'F9','Loca':[1]},56:{'Name':'F10','Loca':[54]},
57:{'Name':'AF7','Loca':[12]},58:{'Name':'AF3','Loca':[17]},59:{'Name':'AF4','Loca':[36]},60:{'Name':'AF8','Loca':[42]},61:{'Name':'PO3','Loca':[23]},62:{'Name':'PO4','Loca':[47]}}

def getChannel_1D_22MI():
    return {
        8: {'Name': 'FC5', 'Loca': [3]}, 9: {'Name': 'FC1', 'Loca': [9]}, 10: {'Name': 'FC2', 'Loca': [12]},
        11: {'Name': 'FC6', 'Loca': [18]}, 12: {'Name': 'T7', 'Loca': [1]}, 13: {'Name': 'C3', 'Loca': [7]},
        15: {'Name': 'C4', 'Loca': [17]}, 16: {'Name': 'T8', 'Loca': [21]}, 18: {'Name': 'CP5', 'Loca': [5]},
        19: {'Name': 'CP1', 'Loca': [11]}, 20: {'Name': 'CP2', 'Loca': [14]}, 21: {'Name': 'CP6', 'Loca': [20]},
        33: {'Name': 'FC3', 'Loca': [6]}, 34: {'Name': 'FC4', 'Loca': [15]}, 35: {'Name': 'C5', 'Loca': [4]},
        36: {'Name': 'C1', 'Loca': [10]}, 37: {'Name': 'C2', 'Loca': [13]}, 38: {'Name': 'C6', 'Loca': [19]},
        39: {'Name': 'CP3', 'Loca': [8]}, 41: {'Name': 'CP4', 'Loca': [17]}, 48: {'Name': 'TP7', 'Loca': [2]},
        53: {'Name': 'TP8', 'Loca': [22]}
    }

def getChannel_1D_6MI():
    return {
        13: {'Name': 'C3', 'Loca': [1]},24:{'Name': 'P3', 'Loca': [2]},
        14:{'Name':'Cz','Loca':[3]},25:{'Name': 'Pz', 'Loca': [4]},
        15: {'Name': 'C4', 'Loca': [5]},26: {'Name': 'P4', 'Loca': [6]},
    }

def getChannel_1D_20MI():#左右对称
    return {
        8: {'Name': 'FC5', 'Loca': [1]}, 9: {'Name': 'FC1', 'Loca': [7]}, 10: {'Name': 'FC2', 'Loca': [18]},
        11: {'Name': 'FC6', 'Loca': [12]}, 13: {'Name': 'C3', 'Loca': [5]},14:{'Name':'Cz','Loca':[10]},
        15: {'Name': 'C4', 'Loca': [16]}, 18: {'Name': 'CP5', 'Loca': [3]},
        19: {'Name': 'CP1', 'Loca': [9]}, 20: {'Name': 'CP2', 'Loca': [20]}, 21: {'Name': 'CP6', 'Loca': [14]},
        33: {'Name': 'FC3', 'Loca': [4]}, 34: {'Name': 'FC4', 'Loca': [15]}, 35: {'Name': 'C5', 'Loca': [2]},
        36: {'Name': 'C1', 'Loca': [8]}, 37: {'Name': 'C2', 'Loca': [19]}, 38: {'Name': 'C6', 'Loca': [13]},
        39: {'Name': 'CP3', 'Loca': [6]},40:{'Name':'CPz','Loca':[11]}, 41: {'Name': 'CP4', 'Loca': [17]}
    }
def getChannel_1D_20MI_2():#从左往右对称
    return {
        8: {'Name': 'FC5', 'Loca': [1]}, 9: {'Name': 'FC1', 'Loca': [7]}, 10: {'Name': 'FC2', 'Loca': [12]},
        11: {'Name': 'FC6', 'Loca': [18]}, 13: {'Name': 'C3', 'Loca': [5]},14:{'Name':'Cz','Loca':[10]},
        15: {'Name': 'C4', 'Loca': [16]}, 18: {'Name': 'CP5', 'Loca': [3]},
        19: {'Name': 'CP1', 'Loca': [9]}, 20: {'Name': 'CP2', 'Loca': [14]}, 21: {'Name': 'CP6', 'Loca': [20]},
        33: {'Name': 'FC3', 'Loca': [4]}, 34: {'Name': 'FC4', 'Loca': [15]}, 35: {'Name': 'C5', 'Loca': [2]},
        36: {'Name': 'C1', 'Loca': [8]}, 37: {'Name': 'C2', 'Loca': [13]}, 38: {'Name': 'C6', 'Loca': [19]},
        39: {'Name': 'CP3', 'Loca': [6]},40:{'Name':'CPz','Loca':[11]}, 41: {'Name': 'CP4', 'Loca': [17]}
    }
def getChannel_1D_26MI():#左右对称
    return {
        8: {'Name': 'FC5', 'Loca': [4]}, 9: {'Name': 'FC1', 'Loca': [10]}, 10: {'Name': 'FC2', 'Loca': [23]},
        11: {'Name': 'FC6', 'Loca': [17]}, 12: {'Name': 'T7', 'Loca': [1]}, 13: {'Name': 'C3', 'Loca': [8]},14:{'Name':'Cz','Loca':[13]},
        15: {'Name': 'C4', 'Loca': [21]}, 16: {'Name': 'T8', 'Loca': [14]}, 18: {'Name': 'CP5', 'Loca': [6]},
        19: {'Name': 'CP1', 'Loca': [12]}, 20: {'Name': 'CP2', 'Loca': [25]}, 21: {'Name': 'CP6', 'Loca': [19]},
        33: {'Name': 'FC3', 'Loca': [7]}, 34: {'Name': 'FC4', 'Loca': [20]}, 35: {'Name': 'C5', 'Loca': [5]},
        36: {'Name': 'C1', 'Loca': [11]}, 37: {'Name': 'C2', 'Loca': [24]}, 38: {'Name': 'C6', 'Loca': [18]},
        39: {'Name': 'CP3', 'Loca': [9]},40:{'Name':'CPz','Loca':[26]}, 41: {'Name': 'CP4', 'Loca': [22]}, 47:{'Name':'FTT7h','Loca':[3]},48: {'Name': 'TP7', 'Loca': [2]},
        52:{'Name':'FTT8h','Loca':[16]},53: {'Name': 'TP8', 'Loca': [15]}
    }

def getChannel_1D_26MI_2():#从左往右
    return {
        8: {'Name': 'FC5', 'Loca': [4]}, 9: {'Name': 'FC1', 'Loca': [10]}, 10: {'Name': 'FC2', 'Loca': [15]},
        11: {'Name': 'FC6', 'Loca': [21]}, 12: {'Name': 'T7', 'Loca': [1]}, 13: {'Name': 'C3', 'Loca': [8]},14:{'Name':'Cz','Loca':[13]},
        15: {'Name': 'C4', 'Loca': [19]}, 16: {'Name': 'T8', 'Loca': [25]}, 18: {'Name': 'CP5', 'Loca': [6]},
        19: {'Name': 'CP1', 'Loca': [12]}, 20: {'Name': 'CP2', 'Loca': [17]}, 21: {'Name': 'CP6', 'Loca': [23]},
        33: {'Name': 'FC3', 'Loca': [7]}, 34: {'Name': 'FC4', 'Loca': [18]}, 35: {'Name': 'C5', 'Loca': [5]},
        36: {'Name': 'C1', 'Loca': [11]}, 37: {'Name': 'C2', 'Loca': [16]}, 38: {'Name': 'C6', 'Loca': [22]},
        39: {'Name': 'CP3', 'Loca': [9]},40:{'Name':'CPz','Loca':[14]}, 41: {'Name': 'CP4', 'Loca': [20]}, 47:{'Name':'FTT7h','Loca':[3]},48: {'Name': 'TP7', 'Loca': [2]},
        52:{'Name':'FTT8h','Loca':[24]},53: {'Name': 'TP8', 'Loca': [26]}
    }

def getChannels_2D():
    #['C1','Fz','C2']['C3','Cz','C4']['CP1','POz','CP2']
    return {36:{'C1':[0,0]},5:{'Fz':[0,1]},37:{'C2':[0,2]},13:{'C3':[1,0]},14:{'Cz':[1,1]},15:{'C4':{1,2}},19:{'CP1':[2,0]},44:{'POz':[2,1]},20:{'CP2':[2,2]}}

def getChannels_3D():
    return {36:{'Name':'C1','Loca':[0,0,0]},37:{'Name':'C2','Loca':[0,0,2]},14:{'Name':'Cz','Loca':[0,1,1]},19:{'Name':'CP1','Loca':[0,2,0]},20:{'Name':'CP2','Loca':[0,2,2]},5:{'Name':'Fz','Loca':[1,0,1]},13:{'Name':'C3','Loca':[1,1,0]},15:{'Name':'C4','Loca':[1,1,2]},44:{'Name':'POz','Loca':[1,2,1]},4:{'Name':'F3','Loca':[2,0,0]},6:{'Name':'F4','Loca':[2,0,2]},30:{'Name':'Oz','Loca':[2,1,1]},29:{'Name':'O1','Loca':[2,2,0]},31:{'Name':'O2','Loca':[2,2,2]}}
def getChannels_3D_24():#{'Name':'','Loca':{}}
    return {1:{'Name':'Fp1','Loca':[0,0,0]},2:{'Name':'Fp2','Loca':[0,0,2]},4:{'Name':'F3','Loca':[0,1,1]},6:{'Name':'F4','Loca':[0,1,3]},8:{'Name':'FC5','Loca':[0,2,0]},11:{'Name':'FC6','Loca':[0,2,2]},9:{'Name':'FC1','Loca':[0,3,1]},10:{'Name':'FC2','Loca':[0,3,3]},
            13:{'Name':'C3','Loca':[1,0,1]},15:{'Name':'C4','Loca':[1,0,3]},5:{'Name':'Fz','Loca':[1,1,0]},14:{'Name':'Cz','Loca':[1,1,2]},36:{'Name':'C1','Loca':[1,2,1]},37:{'Name':'C2','Loca':[1,2,3]},39:{'Name':'CP3','Loca':[1,3,0]},41:{'Name':'CP4','Loca':[1,3,2]},
            19:{'Name':'CP1','Loca':[2,0,0]},20:{'Name':'CP2','Loca':[2,0,2]},30:{'Name':'Oz','Loca':[2,1,1]},40:{'Name':'CPz','Loca':[2,1,3]},29:{'Name':'O1','Loca':[2,2,0]},31:{'Name':'O2','Loca':[2,2,2]},24:{'Name':'P3','Loca':[2,3,1]},26:{'Name':'P4','Loca':[2,3,3]}}
def getChannels_MI_20():
    return {8:{'Name':'FC5','Loca':[1]},33:{'Name':'FC3','Loca':[4]},9:{'Name':'FC1','Loca':[7]},10:{'Name':'FC2','Loca':[12]},34:{'Name':'FC4','Loca':[15]},11:{'Name':'FC6','Loca':[18]},
            35: {'Name': 'C5', 'Loca': [2]},13:{'Name':'C3','Loca':[5]},36:{'Name':'C1','Loca':[8]},14:{'Name':'Cz','Loca':[10]},37:{'Name':'C2','Loca':[13]},15:{'Name':'C4','Loca':[16]},38:{'Name':'C6','Loca':[19]},
            18: {'Name': 'CP5', 'Loca': [3]},39:{'Name':'CP3','Loca':[6]},19:{'Name':'CP1','Loca':[9]},40:{'Name':'CPz','Loca':[11]},20:{'Name':'CP2','Loca':[14]},41:{'Name':'CP4','Loca':[17]},21:{'Name':'CP6','Loca':[20]}}
def getChannels_MI_20_2():
    return {8:{'Name':'FC5','Loca':[1]},33:{'Name':'FC3','Loca':[2]},9:{'Name':'FC1','Loca':[3]},10:{'Name':'FC2','Loca':[4]},34:{'Name':'FC4','Loca':[5]},11:{'Name':'FC6','Loca':[6]},
            35: {'Name': 'C5', 'Loca': [7]},13:{'Name':'C3','Loca':[8]},36:{'Name':'C1','Loca':[9]},14:{'Name':'Cz','Loca':[10]},37:{'Name':'C2','Loca':[11]},15:{'Name':'C4','Loca':[12]},38:{'Name':'C6','Loca':[13]},
            18: {'Name': 'CP5', 'Loca': [14]},39:{'Name':'CP3','Loca':[15]},19:{'Name':'CP1','Loca':[16]},40:{'Name':'CPz','Loca':[17]},20:{'Name':'CP2','Loca':[18]},41:{'Name':'CP4','Loca':[19]},21:{'Name':'CP6','Loca':[20]}}
def getChannels_MI_34():
    return {3:{'Name':'F7','Loca':[]},4:{'Name':'F3','Loca':[]},5:{'Name':'FZ','Loca':[]},6:{'Name':'F4','Loca':[]},7:{'Name':'F8','Loca':[]},8:{'Name':'FC5','Loca':[]},9:{'Name':'FC1','Loca':[]},10:{'Name':'FC2','Loca':[]},11:{'Name':'FC6','Loca':[]},13:{'Name':'C3','Loca':[]},14:{'Name':'CZ','Loca':[]},15:{'Name':'C4','Loca':[]},18:{'Name':'CP5','Loca':[]},19:{'Name':'CP1','Loca':[]},20:{'Name':'CP2','Loca':[]},21:{'Name':'CP6','Loca':[]},23:{'Name':'P7','Loca':[]},24:{'Name':'P3','Loca':[]},25:{'Name':'PZ','Loca':[]},26:{'Name':'P4','Loca':[]},27:{'Name':'P8','Loca':[]},33:{'Name':'FC3','Loca':[]},34:{'Name':'FC4','Loca':[]},
35:{'Name':'C5','Loca':[]},36:{'Name':'C1','Loca':[]},37:{'Name':'C2','Loca':[]},38:{'Name':'C6','Loca':[]},39:{'Name':'CP3','Loca':[]},40:{'Name':'CPZ','Loca':[]},41:{'Name':'CP4','Loca':[]},42:{'Name':'P1','Loca':[]},43:{'Name':'P2','Loca':[]},55:{'Name':'F9','Loca':[]},56:{'Name':'F10','Loca':[]}}

def getChannels_MI_3():
    return {13:{'Name':'C3','Loca':[]},14:{'Name':'Cz','Loca':[]},15:{'Name':'C4','Loca':[]}}

def getEvent_erd():

    return {1: {'Name': 'right', 'StaTime': [0], 'TimeSpan': 4000, 'IsExtend': False},
            2: {'Name': 'left', 'StaTime': [0], 'TimeSpan': 4000, 'IsExtend': False},
            3: {'Name': 'rest', 'StaTime': [4000], 'TimeSpan': 0, 'PreEndTime': 0, 'IsExtend': True}}
    '''
    return {1:{'Name':'right','StaTime':[-500],'TimeSpan':5000,'IsExtend':False},
            2:{'Name':'left','StaTime':[-500],'TimeSpan':5000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
     '''
def getEvent_erd_line():
    return {1: {'Name': 'right', 'StaTime': [-2000], 'TimeSpan': 8010, 'IsExtend': False},
            2: {'Name': 'left', 'StaTime': [-2000], 'TimeSpan': 8010, 'IsExtend': False}}

def getEvent_erd_2():
    return {1: {'Name': 'right', 'StaTime': [0], 'TimeSpan': 6000, 'IsExtend': False},
            2: {'Name': 'left', 'StaTime': [0], 'TimeSpan': 6000, 'IsExtend': False}}

def getEvent_Tran_Cue500():
    return {4:{'Name':"transfer",'StaTime':[-500,3500],'TimeSpan':1000,'IsExtend':True},
            1:{'Name':'right','StaTime':[500],'TimeSpan':3000,'IsExtend':False},
            2:{'Name':'left','StaTime':[500],'TimeSpan':3000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[4500],'TimeSpan':4000,'PreEndTime':0,'IsExtend':True}}
def getEvent_Tran_Cue1000():
    return {4:{'Name':"transfer",'StaTime':[-1000,3000],'TimeSpan':2000,'IsExtend':True},
            1:{'Name':'right','StaTime':[1000],'TimeSpan':2000,'IsExtend':False},
            2:{'Name':'left','StaTime':[1000],'TimeSpan':2000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':4000,'PreEndTime':0,'IsExtend':True}}
def getEvent_noTran():
    return {1:{'Name':'right','StaTime':[0],'TimeSpan':4000,'IsExtend':False},
            2:{'Name':'left','StaTime':[0],'TimeSpan':4000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
def getEvent_noTran_Cue2000():
    return {1:{'Name':'right','StaTime':[2000],'TimeSpan':4000,'IsExtend':False},
            2:{'Name':'left','StaTime':[2000],'TimeSpan':4000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[6000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
def getEvent_noTran_T5000():
    return {1:{'Name':'right','StaTime':[0],'TimeSpan':5000,'IsExtend':False},
            2:{'Name':'left','StaTime':[0],'TimeSpan':5000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[6500],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
def getEvent_noTran_Cue1000():
    return {1:{'Name':'right','StaTime':[1000],'TimeSpan':3500,'IsExtend':False},
            2:{'Name':'left','StaTime':[1000],'TimeSpan':3500,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
def getEvent_noTran_Cue500():
    return {1:{'Name':'right','StaTime':[500],'TimeSpan':3000,'IsExtend':False},
            2:{'Name':'left','StaTime':[500],'TimeSpan':3000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[4000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
def getEvent_noTran_Cue500_T2000():
    return {1:{'Name':'right','StaTime':[500],'TimeSpan':2000,'IsExtend':False},
            2:{'Name':'left','StaTime':[500],'TimeSpan':2000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}

def getEvent_noTran_Cue500_T4000():
    return {1:{'Name':'right','StaTime':[500],'TimeSpan':4000,'IsExtend':False},
            2:{'Name':'left','StaTime':[500],'TimeSpan':4000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
'''
#无效，训练ACC到0.68后又降
def getEvent_noTran_8000():
    return {1:{'Name':'right','StaTime':[0],'TimeSpan':8000,'IsExtend':False},
            2:{'Name':'left','StaTime':[0],'TimeSpan':8000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[8000],'TimeSpan':0,'PreEndTime':0,'IsExtend':True}}
'''
def getEvent_countSample():
    return {4:{'Name':"transfer",'StaTime':[-1000,3000],'TimeSpan':2000,'IsExtend':True},
            1:{'Name':'right','StaTime':[1000],'TimeSpan':2000,'IsExtend':False},
            2:{'Name':'left','StaTime':[1000],'TimeSpan':2000,'IsExtend':False},
            3:{'Name': 'rest','StaTime':[5000],'TimeSpan':0,'PreEndTime':1000,'IsExtend':True}}
def getEvent_NoRest():#Should check the segment data!!!
    return {4:{'Name':"transfer",'StaTime':[-1000,3000],'TimeSpan':2000,'IsExtend':True},
            1:{'Name':'right','StaTime':[1000],'TimeSpan':2000,'IsExtend':False},
            2:{'Name':'left','StaTime':[1000],'TimeSpan':2000,'IsExtend':False}}
def getEventNames_2():
    """Return Event name."""
    return {1:'right', 2:'left'}#,3: 'other'
def getEventNames_3():
    return {1:'right', 2:'left',3: 'rest'}

def findEventIndex(splitTime,time,crossTime,refEventIndex,eventLen):
    for i in range(refEventIndex,eventLen-1):
        if time>=splitTime[i] and time+crossTime<=splitTime[i+1]:
            return i
        elif time<splitTime[i]:
            return -1

def getData_forecast(initData,eventData,windowSize,nStyle,step,foreTime):
    initDatalen = len(initData)
    dataList=[]
    label=[]
    curEventTime=foreTime
    curDataTime=0
    splitTime=eventData[:,0]
    eventLen=len(eventData)
    events_names=getEventNames_3()
    classNum=len(events_names)
    splitIndex_event=0
    splitIndex_data=0
    while curEventTime + windowSize < initDatalen and curDataTime + windowSize < initDatalen:
        try:
            tempIndex=findEventIndex(splitTime,curEventTime,windowSize,splitIndex_event,eventLen)
            if tempIndex>-1:
                splitIndex_event=tempIndex
                tempIndex=findEventIndex(splitTime,curDataTime,windowSize,splitIndex_data,eventLen)
                if tempIndex>-1:
                    splitIndex_data=tempIndex
                    if(splitTime[splitIndex_event]<=curEventTime and curEventTime+windowSize<=splitTime[splitIndex_event+1] and splitTime[splitIndex_data]<=curDataTime and curDataTime+windowSize<=splitTime[splitIndex_data+1]):
                        currData = initData[curDataTime:curDataTime + windowSize]
                        dataList.append(currData)
                        eventMatrix = np.zeros(classNum)
                        curEventIndex=eventData[splitIndex_event,1]
                        eventMatrix[int(curEventIndex) - 1]=1
                        label.append(eventMatrix)
            curDataTime += step
            curEventTime += step
        except Exception as err:
            raise err
    return np.array(dataList, dtype='float64'), np.array(label, dtype='float64')

def loadData_sub(yml,subId,channelDic,eventTypeDic,func,**kwargs):
    learn_data=[]
    learn_label=[]
    subData, subLabel = loadData(yml,subId,channelDic,eventTypeDic,func,isTrain=True)
    learn_data.extend(subData)
    learn_label.extend(subLabel)
    subData, subLabel = loadData(yml, subId, channelDic, eventTypeDic, func, isTrain =False)
    learn_data.extend(subData)
    learn_label.extend(subLabel)
    return learn_data,learn_label

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

def getRawEvent(matlab_data,key,yml,eventTypeDic,doFilter=True):
    hc=hot_code(list(eventTypeDic.keys()))
    #从读取到的sessionX文件中按关键字‘train/test’加载数据
    dataset = matlab_data[key][0][0]
    #data = np.array(dataset['x'])*1e-6
    data = np.array(dataset['x'])
    #datashape记录一次实验数据
    datashape = data.shape
    # channelIndex = list(index_channel.keys())
    #evenTime记录每次执行动作的时间点
    eventTime = np.array(dataset['t'][0])
    #计算fixation时间节点
    fix_eventTime=[]
    fix_time = 3
    orig_smp_freq = yml['Meta']['frequency']
    for i in range(len(eventTime)):
        fix_eventTime.append(eventTime[i]-fix_time*orig_smp_freq)
    #计算rest时间节点
    rest_eventTime = []
    MI_time = 4
    for i in range(len(eventTime)):
        rest_eventTime.append(eventTime[i]+MI_time*orig_smp_freq)
    #添加对应标签1.将fix与rest时间点加入event并排序
    fix_event = np.array(fix_eventTime,dtype=int)
    rest_event = np.array(rest_eventTime,dtype=int)
    event_Time = np.concatenate((eventTime,fix_event),axis=0)
    event_Time = np.concatenate((event_Time,rest_event),axis=0)
    event_Time.sort()
    fix_index = []
    rest_index = []
    for i in range(len(event_Time)):
        if event_Time[i] in fix_event:
            fix_index.append(i)
        elif event_Time[i] in rest_event:
            rest_index.append(i)
    #动作的标签【1,2】总数为100个
    label_1d = np.array(dataset['y_dec'][0])
    # label_pad =np.zeros((200,),dtype=int)
    # label = np.concatenate((label_1d,label_pad),axis=0)
    #在指定位置插入标签
    for i in range(len(fix_index)):
        label_1d = np.insert(label_1d,fix_index[i],0)
        label_1d= np.insert(label_1d, rest_index[i], 0)
    duration = np.zeros((300,),dtype=int)
    eventData = np.vstack((event_Time,duration,label_1d)).T
    eventData = np.r_[[[0,0,0]],eventData]
    # eventData.append([0,eventTime[0][0]-1,3])
    # lastEveIndex = len(eventTime) - 1
    # for i in range(lastEveIndex):
    #     for key, value in eventTypeDic.items():#通过字典判断当前标签的属性，然后对其做切割操作
    #         if value['IsExtend'] == True:
    #             eventData.extend(getSplitEvent(key, value, eventTime[i], eventTime[i + 1]))
    #         elif key == label_1d[i]:
    #             eventData.extend(getSplitEvent(key, value, eventTime[i], eventTime[i + 1]))
    # for key, value in eventTypeDic.items():#对最后一个标签对应的时间做处理
    #     if value['IsExtend'] == True:
    #         eventData.extend(getSplitEvent(key, value, eventTime[lastEveIndex], datashape[0] - 1))
    #     elif key == label_1d[lastEveIndex]:
    #         eventData.extend(getSplitEvent(key, value, eventTime[lastEveIndex], datashape[0] - 1))
    # eventData.sort(key=lambda ed: ed[0]) #按照eventData的第一个维度(时间)排序
    ch_names = get_channelName()#获取通道名称 62个EEG通道
    ch_type = ['eeg' for i in range(len(ch_names))]

    # Add Event hot code into data
    # data = np.pad(data, ((0, 0), (0, len(eventTypeDic))))#填充操作，p(1418040,62)=>(1418040,64)
    # data[0:eventData[0][0], -len(eventTypeDic):] = hc.one_hot_encode([yml['Meta']['otherEventCode'] for j in range(eventData[0][0])])
    # for ed in eventData:
    #     data[ed[0]:ed[0] + ed[1], -len(eventTypeDic):] = hc.one_hot_encode([ed[2] for j in range(ed[1])])

    # ch_type.extend(['stim'] * len(eventTypeDic))
    # ch_names.extend([eValue['Name'] for eValue in eventTypeDic.values()])
    # dataset.append(matlab_data[k][0][0])

    # read EEG standard montage from mne
    '''
    montage =mne.channels.make_standard_montage('standard_1005')# read_montage('standard_1005', ch_names)    
    '''
    #info = mne.create_info(ch_names=ch_names, sfreq=yml['Meta']['frequency'], ch_types=ch_type, montage=montage)
    info = mne.create_info(ch_names=ch_names, sfreq=yml['Meta']['frequency'], ch_types=ch_type)
    #info['filename'] = fname
    # data = 1e-6*np.array(data[ch_names]).T
    # data = np.array(data[ch_names]).T
    # create raw object
    raw = RawArray(data.T, info, verbose=False)
    channel_name = raw.ch_names
    # raw.add_events(eventData)
    if doFilter:
        minFre = yml['Meta']['minFreq']
        maxFre = yml['Meta']['maxFreq']
        raw.filter(l_freq=minFre,h_freq=maxFre)
        '''
        raw=mne_apply(
            lambda a: bandpass_cnt(
                a,
                minFre,
                maxFre,
                yml['Meta']['frequency'],
                filt_order=3,
                axis=1,
            ),
            raw,
        )
        '''

    if 'downSampling' in yml['Meta']:
        #raw = raw.copy().resample(yml['Meta']['downSampling'], npad='auto')
        raw = raw.resample(yml['Meta']['downSampling'], npad='auto')
        sampleRate=yml['Meta']['downSampling']/yml['Meta']['frequency']
        for eD in eventData:
            eD[0]=eD[0]*sampleRate
            eD[1]=eD[1]*sampleRate
    return raw,eventData

def getSplitEvent(eveKey,evePara,curTime,endTime):
    addEveArray=[]
    for i in range(len(evePara['StaTime'])):
        if evePara['TimeSpan']==0:
            #timeSpan = endTime-curTime + evePara['StaTime'][i]-evePara['PreEndTime']
            timeSpan = endTime - (curTime + evePara['StaTime'][i]) - evePara['PreEndTime']
        else:
            timeSpan=evePara['TimeSpan']
        addEveArray.append([curTime+ evePara['StaTime'][i],timeSpan,eveKey])
    return addEveArray

def loadData_matLoad1(yml,subId,channelDic,eventTypeDic,func,**kwargs):
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    subData=[]
    subLabel=[]
    for sess in ['session1','session2']:
        load_matlab = False
        trainDataPath=getLearnDataPath_sess(yml,subId,'EEG_MI_train',sess)
        testDataPath=getLearnDataPath_sess(yml,subId,'EEG_MI_test',sess)
        if reCreate:
            load_matlab=True
        else:
            if not os.path.exists(trainDataPath) or not os.path.exists(testDataPath):
                load_matlab=True
        if load_matlab:
            fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
            matlab_data = scio.loadmat(fname)
            if reCreate or not os.path.exists(trainDataPath):
                trainData,trainLabel=getData(yml, matlab_data, 'EEG_MI_train', eventTypeDic, channelDic, windowSize, func)
                trainData,trainLabel=shuffle(trainData,trainLabel)
                saveRawDataToFile(trainDataPath, {'Data': trainData, 'Label': trainLabel})
            else:
                trainPKL=getRawDataFromFile(trainDataPath)
                trainData=trainPKL['Data']
                trainLabel=trainPKL['Label']
            if reCreate or not os.path.exists(testDataPath):
                testData,testLabel=getData(yml, matlab_data, 'EEG_MI_test', eventTypeDic, channelDic, windowSize, func)
                testData, testLabel=shuffle(testData,testLabel)
                saveRawDataToFile(testDataPath, {'Data': testData, 'Label': testLabel})
            else:
                testPKL=getRawDataFromFile(testDataPath)
                testData=testPKL['Data']
                testLabel=testPKL['Label']
        else:
            trainPKL = getRawDataFromFile(trainDataPath)
            trainData = trainPKL['Data']
            trainLabel = trainPKL['Label']

            testPKL = getRawDataFromFile(testDataPath)
            testData = testPKL['Data']
            testLabel = testPKL['Label']
        subData.extend(trainData)
        subLabel.extend(trainLabel)
        subData.extend(testData)
        subLabel.extend(testLabel)
        subData,subLabel=shuffle(subData,subLabel)
    return subData,subLabel

def getData(yml,matlab_data,key,eventTypeDic,channelDic,windowSize,func):
    raw, events = getRawEvent(matlab_data, key, yml, eventTypeDic)
    events = np.array(events, dtype='int32')
    data = raw._data[0:len(get_channelName())].T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        data = data * expand
    learnData, label = func(yml, data, events, windowSize, eventTypeDic, channelDic)
    return learnData,label

def loadData_mulPkl(yml,subId,channelDic,eventTypeDic,func,isTrain=True):
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    learnData=[]
    learnLabel=[]
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    #hc = hot_code(list(eventTypeDic.keys()))
    for sess in ['session1','session2']:
        if reCreate:
            load_matlab = True
        else:
            existTrain=existMulLearnDataPath(yml,subId,'EEG_MI_train',sess)
            existTest=existMulLearnDataPath(yml,subId,'EEG_MI_test',sess)
            load_matlab=not existTrain&existTest
        if load_matlab:
            fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
            matlab_data = scio.loadmat(fname)
        for matKey in ['EEG_MI_train','EEG_MI_test']:
            if pkl_num<=1:
                learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess)
                if not reCreate and os.path.exists(learnDataPath):
                    leaDic = getRawDataFromFile(learnDataPath)
                    learnData.extend(leaDic['Data'])
                    learnLabel.extend(leaDic['Label'])
                    continue
            else:
                countLoad = 0
                curData=[]
                curLable=[]
                for p_index in range(pkl_num):
                    learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sess,split_index=p_index)
                    if not reCreate and os.path.exists(learnDataPath):
                        leaDic = getRawDataFromFile(learnDataPath)
                        curData.extend(leaDic['Data'])
                        curLable.extend(leaDic['Label'])
                        countLoad+=1
                if countLoad==pkl_num:
                    learnData.extend(curData)
                    learnLabel.extend(curLable)
                    continue
            doFilter=yml['Meta']['doFilter']
            raw, events = getRawEvent(matlab_data, matKey, yml, eventTypeDic,doFilter=doFilter)
            sub_data, sub_label=func(yml,raw, events,channelDic,eventTypeDic)
            sub_data,sub_label=shuffle(sub_data,sub_label)
            #sub_label = hc.one_hot_encode(sub_label)
            sub_data = np.array(sub_data, dtype='float64')
            sub_label = np.array(sub_label, dtype='int32')
            if pkl_num>1:
                totalLen=len(sub_label)
                splitLen=ceil(totalLen/pkl_num)
                startIndex=0
                endIndex=0
                i=0
                print('sub. %d -- splitLen: %d'%(subId,splitLen))
                while endIndex<totalLen:
                    endIndex=startIndex+splitLen
                    if endIndex>=totalLen:
                        endIndex=totalLen
                    splitData=sub_data[startIndex:endIndex]
                    splitLabel=sub_label[startIndex:endIndex]
                    learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess,split_index=i)
                    saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
                    startIndex=endIndex
                    i+=1
            else:
                saveRawDataToFile(learnDataPath, {'Data': sub_data, 'Label': sub_label})
            learnData.extend(sub_data)
            learnLabel.extend(sub_label)
    return learnData,learnLabel

def loadData_mulGroupHead(yml,subId,channelDic,eventTypeDic,func):
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    learnData=[]
    learnLabel=[]
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    #hc = hot_code(list(eventTypeDic.keys()))
    for sess in ['session1','session2']:
        if reCreate:
            load_matlab = True
        else:
            existTrain=existMulLearnDataPath(yml,subId,'EEG_MI_train',sess)
            existTest=existMulLearnDataPath(yml,subId,'EEG_MI_test',sess)
            load_matlab=not existTrain&existTest
        if load_matlab:
            fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
            matlab_data = scio.loadmat(fname)
        for matKey in ['EEG_MI_train','EEG_MI_test']:
            if pkl_num<=1:
                learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess)
                if not reCreate and os.path.exists(learnDataPath):
                    leaDic = getRawDataFromFile(learnDataPath)
                    learnData.extend(leaDic['Data'])
                    learnLabel.extend(leaDic['Label'])
                    continue
            else:
                countLoad = 0
                curData=[]
                curLable=[]
                for p_index in range(pkl_num):
                    learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sess,split_index=p_index)
                    if not reCreate and os.path.exists(learnDataPath):
                        leaDic = getRawDataFromFile(learnDataPath)
                        curData.extend(leaDic['Data'])
                        curLable.extend(leaDic['Label'])
                        countLoad+=1
                if countLoad==pkl_num:
                    learnData.extend(curData)
                    learnLabel.extend(curLable)
                    continue
            doFilter=yml['Meta']['doFilter']
            minFreqs=yml['Meta']['minFreq']
            maxFreqs=yml['Meta']['maxFreq']
            sub_data, sub_label=[],[]
            for i in range(0,len(minFreqs)):
                raw, events = getRawEvent(matlab_data, matKey, yml, eventTypeDic, doFilter=doFilter)
                raw.filter(l_freq=minFreqs[i],h_freq=maxFreqs[i])
                sub_data_g, sub_label_g=func(yml,raw, events,channelDic,eventTypeDic)
                sub_data.append(sub_data_g)
                if i==0:
                    sub_label=sub_label_g
            sub_data = np.array(sub_data, dtype='float64')
            sub_label = np.array(sub_label, dtype='int32')
            sub_data=sub_data.transpose(1,0,2,3)
            sub_data,sub_label=shuffle(sub_data,sub_label)
            if pkl_num>1:
                totalLen=len(sub_label)
                splitLen=ceil(totalLen/pkl_num)
                startIndex=0
                endIndex=0
                i=0
                print('sub. %d -- splitLen: %d'%(subId,splitLen))
                while endIndex<totalLen:
                    endIndex=startIndex+splitLen
                    if endIndex>=totalLen:
                        endIndex=totalLen
                    splitData=sub_data[startIndex:endIndex]
                    splitLabel=sub_label[startIndex:endIndex]
                    learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess,split_index=i)
                    saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
                    startIndex=endIndex
                    i+=1
            else:
                saveRawDataToFile(learnDataPath, {'Data': sub_data, 'Label': sub_label})
            learnData.extend(sub_data)
            learnLabel.extend(sub_label)
    return learnData,learnLabel

def loadData_session(yml,subId,channelDic,eventTypeDic,func,sess=None):
    if sess is None:
        return loadData_mulPkl(yml,subId,channelDic,eventTypeDic,func)
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    learnData=[]
    learnLabel=[]
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    #hc = hot_code(list(eventTypeDic.keys()))

    if reCreate:
        load_matlab = True
    else:
        existTrain=existMulLearnDataPath(yml,subId,'EEG_MI_train',sess)
        existTest=existMulLearnDataPath(yml,subId,'EEG_MI_test',sess)
        load_matlab=not existTrain&existTest
    if load_matlab:
        fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
        matlab_data = scio.loadmat(fname)
    for matKey in ['EEG_MI_train','EEG_MI_test']:
        if pkl_num<=1:
            learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess)
            if not reCreate and os.path.exists(learnDataPath):
                leaDic = getRawDataFromFile(learnDataPath)
                learnData.extend(leaDic['Data'])
                learnLabel.extend(leaDic['Label'])
                continue
        else:
            countLoad = 0
            curData=[]
            curLable=[]
            for p_index in range(pkl_num):
                learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sess,split_index=p_index)
                if not reCreate and os.path.exists(learnDataPath):
                    leaDic = getRawDataFromFile(learnDataPath)
                    curData.extend(leaDic['Data'])
                    curLable.extend(leaDic['Label'])
                    countLoad+=1
            if countLoad==pkl_num:
                learnData.extend(curData)
                learnLabel.extend(curLable)
                continue
        doFilter=yml['Meta']['doFilter']
        raw, events = getRawEvent(matlab_data, matKey, yml, eventTypeDic,doFilter=doFilter)
        sub_data, sub_label=func(yml,raw, events,channelDic,eventTypeDic)
        sub_data,sub_label=shuffle(sub_data,sub_label)
        #sub_label = hc.one_hot_encode(sub_label)
        sub_data = np.array(sub_data, dtype='float64')
        sub_label = np.array(sub_label, dtype='int32')
        if pkl_num>1:
            totalLen=len(sub_label)
            splitLen=ceil(totalLen/pkl_num)
            startIndex=0
            endIndex=0
            i=0
            print('sub. %d -- splitLen: %d'%(subId,splitLen))
            while endIndex<totalLen:
                endIndex=startIndex+splitLen
                if endIndex>=totalLen:
                    endIndex=totalLen
                splitData=sub_data[startIndex:endIndex]
                splitLabel=sub_label[startIndex:endIndex]
                learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess,split_index=i)
                saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
                startIndex=endIndex
                i+=1
        else:
            saveRawDataToFile(learnDataPath, {'Data': sub_data, 'Label': sub_label})
        learnData.extend(sub_data)
        learnLabel.extend(sub_label)
    return learnData,learnLabel

def loadData_mulPkl_f(yml,subId,channelDic,eventTypeDic,func):
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    noRetun=False
    if 'noReturn' in yml['Meta']:
        noRetun=yml['Meta']['noReturn']
    if not noRetun:
        pkl_paths=[]
        data_num=[]
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    #hc = hot_code(list(eventTypeDic.keys()))
    for sess in ['session1','session2']:
        if reCreate:
            load_matlab = True
        else:
            existTrain=existMulLearnDataPath(yml,subId,'EEG_MI_train',sess)
            existTest=existMulLearnDataPath(yml,subId,'EEG_MI_test',sess)
            load_matlab=not existTrain&existTest
        if load_matlab:
            fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
            matlab_data = scio.loadmat(fname)
        elif noRetun:
            continue
        for matKey in ['EEG_MI_train','EEG_MI_test']:
            if pkl_num<=1:
                learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess)
                if not reCreate and os.path.exists(learnDataPath):
                    if noRetun:
                        continue
                    leaDic = getRawDataFromFile(learnDataPath)
                    pkl_paths.append(learnDataPath)
                    data_num.append(len(leaDic['Label']))
                    del leaDic
                    continue
            else:
                countLoad = 0
                tempPklPath=[]
                tempNumLog=[]
                for p_index in range(pkl_num):
                    learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sess,split_index=p_index)
                    if not reCreate and os.path.exists(learnDataPath):
                        countLoad += 1
                        if noRetun:
                            continue
                        leaDic = getRawDataFromFile(learnDataPath)
                        tempPklPath.append(learnDataPath)
                        tempNumLog.append(len(leaDic['Label']))
                        del leaDic

                if countLoad==pkl_num:
                    if not noRetun:
                        pkl_paths.extend(tempPklPath)
                        data_num.extend(tempNumLog)
                    continue
            doFilter=yml['Meta']['doFilter']
            raw, events = getRawEvent(matlab_data, matKey, yml, eventTypeDic,doFilter=doFilter)
            sub_data, sub_label=func(yml,raw, events,channelDic,eventTypeDic)
            sub_data,sub_label=shuffle(sub_data,sub_label)
            #sub_label = hc.one_hot_encode(sub_label)
            sub_data = np.array(sub_data, dtype='float32')
            sub_label = np.array(sub_label, dtype='int32')
            if pkl_num>1:
                totalLen=len(sub_label)
                splitLen=ceil(totalLen/pkl_num)
                startIndex=0
                endIndex=0
                i=0
                print('sub. %d -- splitLen: %d'%(subId,splitLen))
                while endIndex<totalLen:
                    endIndex=startIndex+splitLen
                    if endIndex>=totalLen:
                        endIndex=totalLen
                    splitData=sub_data[startIndex:endIndex]
                    splitLabel=sub_label[startIndex:endIndex]
                    learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess,split_index=i)
                    saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
                    if not noRetun:
                        pkl_paths.append(learnDataPath)
                        data_num.append(len(splitLabel))
                    del splitData,splitLabel
                    startIndex=endIndex
                    i+=1
            else:
                saveRawDataToFile(learnDataPath, {'Data': sub_data, 'Label': sub_label})
                if not noRetun:
                    pkl_paths.append(learnDataPath)
                    data_num.append(len(sub_label))
                del sub_data,sub_label

        if load_matlab:
            del matlab_data
    if not noRetun:
        return pkl_paths,data_num

def loadData_sess_f(yml,subId,channelDic,eventTypeDic,func,sess):
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    noRetun=False
    if 'noReturn' in yml['Meta']:
        noRetun=yml['Meta']['noReturn']
    if not noRetun:
        pkl_paths=[]
        data_num=[]
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    #hc = hot_code(list(eventTypeDic.keys()))

    if reCreate:
        load_matlab = True
    else:
        existTrain=existMulLearnDataPath(yml,subId,'EEG_MI_train',sess)
        existTest=existMulLearnDataPath(yml,subId,'EEG_MI_test',sess)
        load_matlab=not existTrain&existTest
    if load_matlab:
        fname = joinPath(yml['Meta']['initDataFolder'], sess, 's%d' % subId, 'EEG_MI.mat')
        matlab_data = scio.loadmat(fname)
    elif noRetun:
        return
    for matKey in ['EEG_MI_train','EEG_MI_test']:
        if pkl_num<=1:
            learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess)
            if not reCreate and os.path.exists(learnDataPath):
                if noRetun:
                    continue
                leaDic = getRawDataFromFile(learnDataPath)
                pkl_paths.append(learnDataPath)
                data_num.append(len(leaDic['Label']))
                del leaDic
                continue
        else:
            countLoad = 0
            tempPklPath=[]
            tempNumLog=[]
            for p_index in range(pkl_num):
                learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sess,split_index=p_index)
                if not reCreate and os.path.exists(learnDataPath):
                    countLoad += 1
                    if noRetun:
                        continue
                    leaDic = getRawDataFromFile(learnDataPath)
                    tempPklPath.append(learnDataPath)
                    tempNumLog.append(len(leaDic['Label']))
                    del leaDic

            if countLoad==pkl_num:
                if not noRetun:
                    pkl_paths.extend(tempPklPath)
                    data_num.extend(tempNumLog)
                continue
        doFilter=yml['Meta']['doFilter']
        raw, events = getRawEvent(matlab_data, matKey, yml, eventTypeDic,doFilter=doFilter)
        sub_data, sub_label=func(yml,raw, events,channelDic,eventTypeDic)
        sub_data,sub_label=shuffle(sub_data,sub_label)
        #sub_label = hc.one_hot_encode(sub_label)
        sub_data = np.array(sub_data, dtype='float32')
        sub_label = np.array(sub_label, dtype='int32')
        if pkl_num>1:
            totalLen=len(sub_label)
            splitLen=ceil(totalLen/pkl_num)
            startIndex=0
            endIndex=0
            i=0
            print('sub. %d -- splitLen: %d'%(subId,splitLen))
            while endIndex<totalLen:
                endIndex=startIndex+splitLen
                if endIndex>=totalLen:
                    endIndex=totalLen
                splitData=sub_data[startIndex:endIndex]
                splitLabel=sub_label[startIndex:endIndex]
                learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sess,split_index=i)
                saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
                if not noRetun:
                    pkl_paths.append(learnDataPath)
                    data_num.append(len(splitLabel))
                del splitData,splitLabel
                startIndex=endIndex
                i+=1
        else:
            saveRawDataToFile(learnDataPath, {'Data': sub_data, 'Label': sub_label})
            if not noRetun:
                pkl_paths.append(learnDataPath)
                data_num.append(len(sub_label))
            del sub_data,sub_label

    if matlab_data is not None:
        del matlab_data
    if not noRetun:
        return pkl_paths,data_num

def loadData_Sess_Key(yml,subId,channelDic,eventTypeDic,func,sessName,isTrain=True):
    dataPath = yml['Meta']['initDataFolder']
    learnDataPath = '%s/learnData' % dataPath
    if not os.path.exists(learnDataPath):
        os.mkdir(learnDataPath)
    reCreate = False
    if 'reCreate' in yml['Meta']:
        reCreate = yml['Meta']['reCreate']
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    hc = hot_code(list(eventTypeDic.keys()))
    if isTrain:
        matKey='EEG_MI_train'
    else:
        matKey='EEG_MI_test'
    if reCreate:
        load_matlab = True
    else:
        load_matlab=not existMulLearnDataPath(yml,subId,matKey,sessName)
    if load_matlab:
        fname = joinPath(yml['Meta']['initDataFolder'], sessName, 's%d' % subId, 'EEG_MI.mat')
        matlab_data = scio.loadmat(fname)
        raw, events = getRawEvent(matlab_data, matKey, yml, eventTypeDic,doFilter=False)
        sub_data, sub_label=func(yml,raw, events,channelDic,eventTypeDic)
        sub_data,sub_label=shuffle(sub_data,sub_label)
        sub_label = hc.one_hot_encode(sub_label)
        sub_data = np.array(sub_data, dtype='float64')
        sub_label = np.array(sub_label, dtype='int32')
        if pkl_num>1:
            totalLen=len(sub_label)
            splitLen=ceil(totalLen/pkl_num)
            startIndex=0
            endIndex=0
            i=0
            print('sub. %d -- splitLen: %d'%(subId,splitLen))
            while endIndex<totalLen:
                endIndex=startIndex+splitLen
                if endIndex>=totalLen:
                    endIndex=totalLen
                splitData=sub_data[startIndex:endIndex]
                splitLabel=sub_label[startIndex:endIndex]
                learnDataPath=getLearnDataPath_sess(yml,subId,matKey,sessName,split_index=i)
                saveRawDataToFile(learnDataPath, {'Data': splitData, 'Label': splitLabel})
                startIndex=endIndex
                i+=1
        else:
            learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sessName)
            saveRawDataToFile(learnDataPath, {'Data': sub_data, 'Label': sub_label})
    else:
        sub_data = []
        sub_label = []
        if pkl_num <= 1:
            learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sessName)
            if not reCreate and os.path.exists(learnDataPath):
                leaDic = getRawDataFromFile(learnDataPath)
                sub_data.extend(leaDic['Data'])
                sub_label.extend(leaDic['Label'])
        else:
            countLoad = 0
            for p_index in range(pkl_num):
                learnDataPath = getLearnDataPath_sess(yml, subId, matKey, sessName, split_index=p_index)
                if not reCreate and os.path.exists(learnDataPath):
                    leaDic = getRawDataFromFile(learnDataPath)
                    sub_data.extend(leaDic['Data'])
                    sub_label.extend(leaDic['Label'])
                    countLoad += 1
    return sub_data,sub_label

def getdata_multitaper(yml,raw, events,chanDic,eventTypeDic):
    channelNames = [v['Name'] for v in chanDic.values()]
    picks = mne.pick_channels(raw.info["ch_names"], channelNames)
    event_segLog = {}  # eventId,{SegLog:[],Data:initData (epochs number),}
    minTimeSpan = 10000000
    for key, value in eventTypeDic.items():
        if value['TimeSpan'] != 0 and value['TimeSpan'] < minTimeSpan:
            minTimeSpan = value['TimeSpan']
    minTimeSpan = (minTimeSpan - 1) / yml['Meta']['frequency']
    windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
    step = yml['Meta']['step']
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    freqs = np.arange(yml['Meta']['minFreq'], yml['Meta']['maxFreq'], 1)  # frequencies from 2-35Hz
    for key, value in eventTypeDic.items():
        # epoch data ##################################################################
        epochs = mne.Epochs(raw, events, {value['Name']: key}, 0, minTimeSpan, picks=picks,
                            baseline=None, preload=True)
        # Run TF decomposition overall epochs
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True, return_itc=False, average=False,
                             decim=yml['Meta']['decim'])
        tfr.crop(0, minTimeSpan)
        tfr.apply_baseline(yml['Meta']['baseline'], mode="percent")

        dShape = tfr.data.shape
        event_segLog.setdefault(key, [])
        for i in range(dShape[0]):
            data = np.transpose(tfr.data[i])
            if expand != 1:
                data = data * expand
            curSegDic = {'Data': data, 'SegLog': []}
            curTime = 0
            while curTime + windowSize <= len(data):
                curSegDic['SegLog'].append(curTime)
                curTime += step
            event_segLog[key].append(curSegDic)
    # Equal Data
    keyLogDic = equalSegDicData(event_segLog)  # {key,[{SegLog:,Index:}]} Index is in the event_segLog[key]
    sub_data = []
    sub_label = []
    for key, value in keyLogDic.items():
        for sl in value:
            curData = event_segLog[key][sl['Index']]['Data']
            curData = curData.transpose((0, 2, 1))
            cShape = curData.shape
            sub_data.append(curData[sl['SegLog']:sl['SegLog'] + windowSize])
            sub_label.append(key)
    hc = hot_code(list(eventTypeDic.keys()))
    sub_label=hc.one_hot_encode(sub_label)
    return sub_data, sub_label

def getdata_morlet(yml,raw, events,chanDic,eventTypeDic):
    channelNames = [v['Name'] for v in chanDic.values()]
    picks = mne.pick_channels(raw.info["ch_names"], channelNames)
    event_segLog = {}  # eventId,{SegLog:[],Data:initData (epochs number),}
    minTimeSpan = 10000000
    for key, value in eventTypeDic.items():
        if value['TimeSpan'] != 0 and value['TimeSpan'] < minTimeSpan:
            minTimeSpan = value['TimeSpan']
    # minTimeSpan=(minTimeSpan-1)/yml['Meta']['frequency']
    minTimeSpan = (minTimeSpan-1) / yml['Meta']['frequency']
    windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
    step = yml['Meta']['step']
    freqs = np.arange(yml['Meta']['minFreq'], yml['Meta']['maxFreq'], 1, dtype='float32')
    for eType_key, eType_value in eventTypeDic.items():
        # epoch data ##################################################################
        epochs = mne.Epochs(raw, events, {eType_value['Name']: eType_key}, 0, minTimeSpan, picks=picks,
                            baseline=None, preload=True)

        tfr_epochs = tfr_morlet(epochs, freqs, n_cycles=freqs, decim=1, average=False, return_itc=False, n_jobs=1)
        # Baseline power
        tfr_epochs.apply_baseline(mode='logratio', baseline=(-.100, 0))

        event_segLog.setdefault(eType_key, [])
        data = np.transpose(tfr_epochs.data,[0,3,2,1])
        dShape = data.shape
        for epochIndex in range(dShape[0]):
            curData=data[epochIndex]
            curSegDic = {'Data': curData, 'SegLog': []}
            curTime = 0
            while curTime + windowSize <= dShape[1]:
                curSegDic['SegLog'].append(curTime)
                curTime += step
            event_segLog[eType_key].append(curSegDic)
    # Equal Data
    keyLogDic = equalSegDicData(event_segLog)  # {key,[{SegLog:,Index:}]} Index is in the event_segLog[key]
    sub_data = []
    sub_label = []
    for log_key, log_value in keyLogDic.items():
        for sl in log_value:
            curData = event_segLog[log_key][sl['Index']]['Data']
            sub_data.append(curData[sl['SegLog']:sl['SegLog'] + windowSize])
            sub_label.append(log_key)
    hc = hot_code(list(eventTypeDic.keys()))
    sub_label = hc.one_hot_encode(sub_label)
    return sub_data,sub_label

def getdata_morlet_trainEvent(yml,raw, events,chanDic,eventTypeDic):
    channelNames = [v['Name'] for v in chanDic.values()]
    picks = mne.pick_channels(raw.info["ch_names"], channelNames)
    event_segLog = {}  # eventId,{SegLog:[],Data:initData (epochs number),}
    minTimeSpan = 10000000
    for key, value in eventTypeDic.items():
        if value['TimeSpan'] != 0 and value['TimeSpan'] < minTimeSpan:
            minTimeSpan = value['TimeSpan']
    # minTimeSpan=(minTimeSpan-1)/yml['Meta']['frequency']
    minTimeSpan = (minTimeSpan-1) / yml['Meta']['frequency']
    windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
    step = yml['Meta']['step']
    freqs = np.arange(yml['Meta']['minFreq'], yml['Meta']['maxFreq'], 1, dtype='float32')
    train_eventId=yml['Meta']['train_eventId']
    for eType_key, eType_value in eventTypeDic.items():
        if eType_key not in train_eventId:
            continue
        # epoch data ##################################################################
        epochs = mne.Epochs(raw, events, {eType_value['Name']: eType_key}, 0, minTimeSpan, picks=picks,
                            baseline=None, preload=True)

        tfr_epochs = tfr_morlet(epochs, freqs, n_cycles=freqs, decim=1, average=False, return_itc=False, n_jobs=1)
        # Baseline power
        tfr_epochs.apply_baseline(mode='logratio', baseline=(-.100, 0))

        event_segLog.setdefault(eType_key, [])
        data = np.transpose(tfr_epochs.data,[0,3,2,1])
        dShape = data.shape
        for epochIndex in range(dShape[0]):
            curData=data[epochIndex]
            curSegDic = {'Data': curData, 'SegLog': []}
            curTime = 0
            while curTime + windowSize <= dShape[1]:
                curSegDic['SegLog'].append(curTime)
                curTime += step
            event_segLog[eType_key].append(curSegDic)
    # Equal Data
    keyLogDic = equalSegDicData(event_segLog)  # {key,[{SegLog:,Index:}]} Index is in the event_segLog[key]
    sub_data = []
    sub_label = []
    for log_key, log_value in keyLogDic.items():
        for sl in log_value:
            curData = event_segLog[log_key][sl['Index']]['Data']
            sub_data.append(curData[sl['SegLog']:sl['SegLog'] + windowSize])
            sub_label.append(log_key)
    hc = hot_code(list(eventTypeDic.keys()))
    sub_label = hc.one_hot_encode(sub_label)[:, 0:len(train_eventId)]
    return sub_data,sub_label

def getdata_permutation(yml,raw, events,chanDic,eventTypeDic):
    channelNames = [v['Name'] for v in chanDic.values()]
    picks = mne.pick_channels(raw.info["ch_names"], channelNames)
    event_segLog = {}  # eventId,{SegLog:[],Data:initData (epochs number),}
    minTimeSpan = 10000000
    for key, value in eventTypeDic.items():
        if value['TimeSpan'] != 0 and value['TimeSpan'] < minTimeSpan:
            minTimeSpan = value['TimeSpan']
    minTimeSpan = minTimeSpan / yml['Meta']['frequency']
    windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
    step = yml['Meta']['step']
    freqs = np.arange(yml['Meta']['minFreq'], yml['Meta']['maxFreq'], 1, dtype='float32')
    for key, value in eventTypeDic.items():
        # epoch data ##################################################################
        epochs = mne.Epochs(raw, events, {value['Name']: key}, 0, minTimeSpan, picks=picks,
                            baseline=None, preload=True)

        tfr_epochs = tfr_morlet(epochs, freqs, n_cycles=freqs, decim=1, average=False, return_itc=False, n_jobs=1)
        # tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=True, decim=3, n_jobs=1)
        # Baseline power
        tfr_epochs.apply_baseline(mode='logratio', baseline=(-.100, 0))
        threshold = 2.5
        n_permutations = 100  # Warning: 100 is too small for real-world analysis.
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(tfr_epochs.data, n_permutations=n_permutations,threshold=threshold, tail=0)

        event_segLog.setdefault(key, [])
        data = np.transpose(T_obs)[1:, ...]
        dShape = data.shape
        curSegDic = {'Data': data, 'SegLog': []}
        curTime = 0
        while curTime + windowSize <= dShape[0]:
            curSegDic['SegLog'].append(curTime)
            curTime += step
        event_segLog[key].append(curSegDic)
    # Equal Data
    keyLogDic = equalSegDicData(event_segLog)  # {key,[{SegLog:,Index:}]} Index is in the event_segLog[key]
    sub_data = []
    sub_label = []
    for key, value in keyLogDic.items():
        for sl in value:
            curData = event_segLog[key][sl['Index']]['Data']
            sub_data.append(curData[sl['SegLog']:sl['SegLog'] + windowSize])
            sub_label.append(key)
    hc = hot_code(list(eventTypeDic.keys()))
    sub_label = hc.one_hot_encode(sub_label)
    return sub_data,sub_label

def getdata_1D(yml,raw, events,chanDic,eventTypeDic):
    eventData = np.array(events, dtype='int32')
    initData = raw._data.T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        initData = initData * expand
    windowSize,step=get_windowSize_step(yml)
    dataShape = initData.shape
    cutDataLog = []
    labelLog = []
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime, curTime + windowSize])
            labelLog.append(eventData[i][2])
            curTime = curTime + step
    eventLen = len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
        cutDataLog.append([curTime, curTime + windowSize])
        labelLog.append(eventData[eventLen - 1][2])
        # labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
        curTime = curTime + step

    labelLog = np.array(labelLog, dtype='int32')

    # Equal Example
    keys = np.unique(eventData[:, 2])
    cutDataLog, labelLog = equalData(keys, cutDataLog, labelLog,doShuffle=False)
    dataList = []

    for i in range(len(cutDataLog)):
        seg_data = initData[cutDataLog[i][0]:cutDataLog[i][1]]
        t_data = np.zeros([windowSize, len(chanDic.keys())])
        for t in range(windowSize):
            for key, value in chanDic.items():
                t_data[t, value['Loca'][0] - 1] = seg_data[t][key - 1]
        #t_data = np.array([initData[cutDataLog[i][0]:cutDataLog[i][1]][:, k-1] for k in chanDic.keys()],dtype='float64').T
        if doNormalize:
            scaler = StandardScaler()
            scaler = scaler.fit(t_data)
            t_data = scaler.transform(t_data)
        dataList.append(t_data)

    hc = hot_code(list(eventTypeDic.keys()))
    labelLog = hc.one_hot_encode(labelLog)
    return dataList, labelLog

def getdata_1D_Forcast(yml,raw, events,chanDic,eventTypeDic):
    eventData = np.array(events, dtype='int32')
    initData = raw._data.T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        initData = initData * expand
    windowSize, step = get_windowSize_step(yml)
    dataShape = initData.shape
    cutDataLog_2Data = []
    labelLog = []
    train_eventId = yml['Meta']['train_eventId']
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    initDatalen=dataShape[0]
    splitIndex_event = 0
    splitIndex_data = 0
    otherEventData=yml['Meta']['otherEventCode']

    curEventTime = get_forcast(yml) #If not select event ids, should remove  eventData[0][0]
    curDataTime = 0#If not select event ids, should remove  eventData[0][0]
    eventData=np.insert(eventData,0,[0,eventData[0][0],otherEventData],axis=0) #If not select event ids, should add this code
    eventData=np.append(eventData,[[eventData[-1][0]+eventData[-1][1],initDatalen-eventData[-1][0]-eventData[-1][1],otherEventData]],axis=0)
    splitTime=eventData[:,0]
    eventLen = len(eventData)

    while curEventTime + windowSize < initDatalen and curDataTime + windowSize < initDatalen:
        try:
            tempIndex = findEventIndex(splitTime, curEventTime, windowSize, splitIndex_event, eventLen)
            if tempIndex > -1:
                splitIndex_event = tempIndex
                tempIndex = findEventIndex(splitTime, curDataTime, windowSize, splitIndex_data, eventLen)
                if tempIndex > -1:
                    splitIndex_data = tempIndex
                    if (splitTime[splitIndex_event] <= curEventTime and curEventTime + windowSize <= splitTime[splitIndex_event + 1] and splitTime[splitIndex_data] <= curDataTime and curDataTime + windowSize <= splitTime[splitIndex_data + 1]):
                        cutDataLog_2Data.append([curDataTime,curEventTime])
                        labelLog.append(eventData[splitIndex_event][2])
            curDataTime += step
            curEventTime += step
        except Exception as err:
            raise err

    labelLog = np.array(labelLog, dtype='int32')
    # Equal Example
    keys = np.unique(eventData[:, 2])
    cutDataLog_2Data, labelLog = equalData(keys, cutDataLog_2Data, labelLog,doShuffle=False)
    dataList = []


    for i in range(len(cutDataLog_2Data)):
        curData=np.array([initData[cutDataLog_2Data[i][0]:cutDataLog_2Data[i][0]+windowSize][:, k] for k in chanDic.keys()],dtype='float64').T
        forData = np.array([initData[cutDataLog_2Data[i][1]:cutDataLog_2Data[i][1]+windowSize][:, k] for k in chanDic.keys()],dtype='float64').T
        if doNormalize:
            curScaler = StandardScaler()
            curScaler = curScaler.fit(curData)
            curData = curScaler.transform(curData)
            forScaler=StandardScaler()
            forScaler=forScaler.fit(forData)
            forData=forScaler.transform(forData)
        twoData=np.pad(curData,forData)
        dataList.append(twoData)
    hc = hot_code(list(eventTypeDic.keys()))
    labelLog = hc.one_hot_encode(labelLog)[:,0:len(yml['Meta']['train_eventId'])]#!!! If train_eventId  is not ranged in front, Should not run the current code.
    return dataList, labelLog

def getdata_1D_SelEveId(yml,raw, events,chanDic,eventTypeDic):
    eventData = np.array(events, dtype='int32')
    initData = raw._data.T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        initData = initData * expand
    windowSize,step = get_windowSize_step(yml)
    dataShape = initData.shape
    cutDataLog = []
    labelLog = []
    train_eventId = yml['Meta']['train_eventId']
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        if eventData[i][2] not in train_eventId:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime, curTime + windowSize])
            labelLog.append(eventData[i][2])
            curTime = curTime + step
    eventLen = len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    if eventData[eventLen - 1][2] in train_eventId:
        while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime, curTime + windowSize])
            labelLog.append(eventData[eventLen - 1][2])
            # labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
            curTime = curTime + step

    labelLog = np.array(labelLog, dtype='int32')

    # Equal Example
    keys = np.unique(eventData[:, 2])
    cutDataLog, labelLog = equalData(keys, cutDataLog, labelLog,doShuffle=False)
    dataList = []

    for i in range(len(cutDataLog)):
        seg_data = initData[cutDataLog[i][0]:cutDataLog[i][1]]
        t_data = np.zeros([windowSize, len(chanDic.keys())])
        for t in range(windowSize):
            for key, value in chanDic.items():
                t_data[t, value['Loca'][0] - 1] = seg_data[t][key - 1]
        #t_data = np.array([initData[cutDataLog[i][0]:cutDataLog[i][1]][:, k-1] for k in chanDic.keys()],dtype='float64').T
        if doNormalize:
            scaler = StandardScaler()
            scaler = scaler.fit(t_data)
            t_data = scaler.transform(t_data)
        dataList.append(t_data)
    hc = hot_code(list(eventTypeDic.keys()))
    labelLog = hc.one_hot_encode(labelLog)[:,0:len(yml['Meta']['train_eventId'])]#!!! If train_eventId  is not ranged in front, Should not run the current code.
    return dataList, labelLog

def getdata_1D_SelEveIdForcast(yml,raw, events,chanDic,eventTypeDic):
    eventData = np.array(events, dtype='int32')
    initData = raw._data.T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        initData = initData * expand
    windowSize, step = get_windowSize_step(yml)
    dataShape = initData.shape
    cutDataLog_2Data = []
    labelLog = []
    train_eventId = yml['Meta']['train_eventId']
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    initDatalen=dataShape[0]
    splitIndex_event = 0
    splitIndex_data = 0
    otherEventData=yml['Meta']['otherEventCode']

    curEventTime = eventData[0][0]+get_forcast(yml) #If not select event ids, should remove  eventData[0][0]
    curDataTime = eventData[0][0] #If not select event ids, should remove  eventData[0][0]
   #eventData=np.insert(eventData,0,[0,eventData[0][0],otherEventData],axis=0) #If not select event ids, should add this code
    eventData=np.append(eventData,[[eventData[-1][0]+eventData[-1][1],initDatalen-eventData[-1][0]-eventData[-1][1],otherEventData]],axis=0)
    splitTime=eventData[:,0]
    eventLen = len(eventData)

    while curEventTime + windowSize < initDatalen and curDataTime + windowSize < initDatalen:
        try:
            tempIndex = findEventIndex(splitTime, curEventTime, windowSize, splitIndex_event, eventLen)
            if tempIndex > -1:
                splitIndex_event = tempIndex
                if eventData[splitIndex_event][2] in train_eventId:
                    tempIndex = findEventIndex(splitTime, curDataTime, windowSize, splitIndex_data, eventLen)
                    if tempIndex > -1:
                        splitIndex_data = tempIndex
                        if (splitTime[splitIndex_event] <= curEventTime and curEventTime + windowSize <= splitTime[splitIndex_event + 1] and splitTime[splitIndex_data] <= curDataTime and curDataTime + windowSize <= splitTime[splitIndex_data + 1]):
                            cutDataLog_2Data.append([curDataTime,curEventTime])
                            labelLog.append(eventData[splitIndex_event][2])
            curDataTime += step
            curEventTime += step
        except Exception as err:
            raise err

    labelLog = np.array(labelLog, dtype='int32')
    # Equal Example
    keys = np.unique(eventData[:, 2])
    cutDataLog_2Data, labelLog = equalData(keys, cutDataLog_2Data, labelLog,doShuffle=False)
    dataList = []

    for i in range(len(cutDataLog_2Data)):
        curData=np.array([initData[cutDataLog_2Data[i][0]:cutDataLog_2Data[i][0]+windowSize][:, k] for k in chanDic.keys()],dtype='float64').T
        forData = np.array([initData[cutDataLog_2Data[i][1]:cutDataLog_2Data[i][1]+windowSize][:, k] for k in chanDic.keys()],dtype='float64').T
        if doNormalize:
            curScaler = StandardScaler()
            curScaler = curScaler.fit(curData)
            curData = curScaler.transform(curData)
            forScaler=StandardScaler()
            forScaler=forScaler.fit(forData)
            forData=forScaler.transform(forData)
        twoData=np.concatenate((curData,forData),axis=1)
        dataList.append(twoData)
    hc = hot_code(list(eventTypeDic.keys()))
    labelLog = hc.one_hot_encode(labelLog)[:,0:len(yml['Meta']['train_eventId'])]#!!! If train_eventId  is not ranged in front, Should not run the current code.
    return dataList, labelLog

def getdata_1D_SelEveId_ExpForcast(yml,raw, events,chanDic,eventTypeDic):
    #windowSize+Forcast time size is the data shape[1].
    eventData = np.array(events, dtype='int32')
    initData = raw._data.T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        initData = initData * expand
    windowSize, step = get_windowSize_step(yml,conBindForcast=True)
    dataShape = initData.shape
    cutDataLog = []
    labelLog = []
    train_eventId = yml['Meta']['train_eventId']
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        if eventData[i][2] not in train_eventId:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime, curTime + windowSize])
            labelLog.append(eventData[i][2])
            curTime = curTime + step
    eventLen = len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    if eventData[eventLen - 1][2] in train_eventId:
        while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime, curTime + windowSize])
            labelLog.append(eventData[eventLen - 1][2])
            # labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
            curTime = curTime + step

    labelLog = np.array(labelLog, dtype='int32')

    # Equal Example
    keys = np.unique(eventData[:, 2])
    cutDataLog, labelLog = equalData(keys, cutDataLog, labelLog,doShuffle=False)
    dataList = []

    for i in range(len(cutDataLog)):
        seg_data=initData[cutDataLog[i][0]:cutDataLog[i][1]]
        t_data = np.zeros([windowSize,len(chanDic.keys())])
        for t in range(windowSize):
            for key,value in chanDic.items():
                t_data[t,value['Loca'][0]-1]=seg_data[t][key-1]
        #t_data = np.array([initData[cutDataLog[i][0]:cutDataLog[i][1]][:, k-1] for k in chanDic.keys()],dtype='float64').T

        if doNormalize:
            scaler = StandardScaler()
            scaler = scaler.fit(t_data)
            t_data = scaler.transform(t_data)
        dataList.append(t_data)
    hc = hot_code(list(eventTypeDic.keys()))
    labelLog = hc.one_hot_encode(labelLog)[:,0:len(yml['Meta']['train_eventId'])]#!!! If train_eventId  is not ranged in front, Should not run the current code.
    return dataList, labelLog

def get_windowSize_step(yml,conBindForcast=False):
    downSampleRate=1
    if 'downSampling' in yml['Meta']:
        downSampleRate=yml['Meta']['downSampling']/yml['Meta']['frequency']
    if conBindForcast:
        windowSize= int(yml['Meta']['frequency'] * yml['Meta']['seconds']*downSampleRate) + int(yml['Meta']['forcast']*downSampleRate)
    else:
        windowSize= int(yml['Meta']['frequency'] * yml['Meta']['seconds']*downSampleRate)
    step =int(yml['Meta']['step']*downSampleRate)
    return windowSize,step

def get_forcast(yml):
    downSampleRate = 1
    if 'downSampling' in yml['Meta']:
        downSampleRate = yml['Meta']['downSampling'] / yml['Meta']['frequency']
    return int(yml['Meta']['forcast']*downSampleRate)

def getdata_1D_SelEveId_Hemisphere(yml,raw, events,chanDic,eventTypeDic):
    eventData = np.array(events, dtype='int32')
    initData = raw._data.T
    expand = 1
    if 'expandData' in yml['Meta']:
        expand = yml['Meta']['expandData']
    if expand != 1:
        initData = initData * expand
    windowSize = int(yml['Meta']['frequency'] * yml['Meta']['seconds'])
    step = yml['Meta']['step']
    dataShape = initData.shape
    labelLog = []
    train_eventId = yml['Meta']['train_eventId']
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    initDatalen=dataShape[0]
    otherEventData=yml['Meta']['otherEventCode']

    eventData=np.append(eventData,[[eventData[-1][0]+eventData[-1][1],initDatalen-eventData[-1][0]-eventData[-1][1],otherEventData]],axis=0)

    cutDataLog=[]
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        if eventData[i][2] not in train_eventId:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime])
            labelLog.append(eventData[i][2])
            curTime = curTime + step
    eventLen = len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    if eventData[eventLen - 1][2] in train_eventId:
        while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime])
            labelLog.append(eventData[eventLen - 1][2])
            # labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
            curTime = curTime + step

    labelLog = np.array(labelLog, dtype='int32')

    # Equal Example
    keys = np.unique(eventData[:, 2])
    cutDataLog, labelLog = equalData(keys, cutDataLog, labelLog,doShuffle=False)
    dataList = []
    leftDic={8:{'Name':'FC5'},33:{'Name':'FC3'},9:{'Name':'FC1'},35: {'Name': 'C5'},13:{'Name':'C3'},36:{'Name':'C1'},14:{'Name':'Cz'},
18: {'Name': 'CP5'},39:{'Name':'CP3'},19:{'Name':'CP1'},40:{'Name':'CPz'}}
    rightDic={10:{'Name':'FC2'},34:{'Name':'FC4'},11:{'Name':'FC6'},14:{'Name':'Cz'},37:{'Name':'C2'},15:{'Name':'C4'},38:{'Name':'C6'},40:{'Name':'CPz'},20:{'Name':'CP2'},41:{'Name':'CP4'},21:{'Name':'CP6'}}
    for i in range(len(cutDataLog)):
        left_data = np.array([initData[cutDataLog[i][0]:cutDataLog[i][0]+windowSize][:, k-1] for k in leftDic.keys()],dtype='float64').T
        right_data=np.array([initData[cutDataLog[i][0]:cutDataLog[i][0]+windowSize][:, k-1] for k in rightDic.keys()],dtype='float64').T
        if doNormalize:
            scaler = StandardScaler()
            scaler = scaler.fit(left_data)
            left_data = scaler.transform(left_data)
            scaler = StandardScaler()
            scaler = scaler.fit(right_data)
            right_data=scaler.transform(right_data)
        t_data = np.concatenate((left_data, right_data), axis=1)
        dataList.append(t_data)

    hc = hot_code(list(eventTypeDic.keys()))
    labelLog = hc.one_hot_encode(labelLog)[:,0:len(yml['Meta']['train_eventId'])]#!!! If train_eventId  is not ranged in front, Should not run the current code.
    return dataList, labelLog

def getdata_5d(curData,cShape):
    curData.reshape(cShape[0], cShape[1], int(sqrt(cShape[2])), int(sqrt(cShape[2])))
    return curData

def getLearnDataPath_sess(yml,subId,key,sess,split_index=-1):
    dataPath = yml['Meta']['initDataFolder']
    dataSegType = yml['Meta']['segmentName']

    if split_index==-1:
        learnDataPath = '%s/learnData/%s_s%d_%s_%s.pkl' % (dataPath, dataSegType,subId,  key, sess)
    else:
        learnDataPath = '%s/learnData/%s_s%d_%s_%s_%d.pkl' % (dataPath,dataSegType,subId,  key,sess,split_index)
    return learnDataPath

def existMulLearnDataPath(yml,subId,key,sess):
    dataPath = yml['Meta']['initDataFolder']
    dataSegType = yml['Meta']['segmentName']
    pkl_num=1
    if 'pkl_num' in yml['Meta']:
        pkl_num=yml['Meta']['pkl_num']
    if pkl_num>1:
        for i in range(pkl_num):
            learnDataPath = '%s/learnData/%s_s%d_%s_%s_%d.pkl' % (dataPath,dataSegType,subId,  key,sess, i)
            if not os.path.exists(learnDataPath):
                return False
        return True
    else:
        learnDataPath ='%s/learnData/%s_s%d_%s_%s.pkl' % (dataPath, dataSegType,subId,  key, sess)
        return os.path.exists(learnDataPath)

def getLearnDataPath(yml,isTrain,subId):
    dataPath = yml['Meta']['initDataFolder']
    dataSegType=yml['Meta']['segmentName']
    if isTrain:
        learnDataPath = '%s/learnData/MI_Train_%s_s%d.pkl' % (dataPath, dataSegType, subId)
    else:
        learnDataPath = '%s/learnData/MI_Test_%s_s%d.pkl' % (dataPath, dataSegType, subId)
    return learnDataPath

def loadData_getFilePath(yml,subId,isTrain,channelDic,eventTypeDic,func,*args):
    learnDataPath=getLearnDataPath(yml,isTrain,subId)
    if os.path.exists(learnDataPath):
        subData=getRawDataFromFile(learnDataPath)
        return learnDataPath,len(subData['Label'])
    else:
        _, label_2d=loadData(yml,subId, isTrain, channelDic, eventTypeDic, func, *args)
        return learnDataPath, len(label_2d)





def getData_3DChan(yml,initData,eventData,windowSize,eventTypeDic,chanDic):
    dataShape=initData.shape
    cutDataLog=[]
    labelLog=[]
    step=yml['Meta']['step']
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime,curTime + windowSize])
            labelLog.append(eventData[i][2])
            #labelLog.extend([eventData[i][2] for k in range(0,windowSize)])
            curTime = curTime + step
    eventLen=len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
        cutDataLog.append([curTime,curTime + windowSize])
        labelLog.append(eventData[eventLen - 1][2])
        #labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
        curTime = curTime + step

    labelLog=np.array(labelLog,dtype='int32')

    # Equal Example
    keys=np.unique(eventData[:,2])
    cutDataLog,labelLog=equalData(keys,cutDataLog,labelLog)

    dataList=[]
    channelShape=yml['Meta']['channelShape'][0],yml['Meta']['channelShape'][1],yml['Meta']['channelShape'][2]
    for i in range(len(cutDataLog)):
        currData=initData[cutDataLog[i][0]:cutDataLog[i][1]]
        reshapeData=[]
        for w in range(windowSize):
            eachData=np.zeros(channelShape)
            for key,value in chanDic.items():
                loca=value['Loca']
                eachData[loca[0],loca[1],loca[2]]=currData[w][key-1] ##should be checked.!!! for 'key-1'
            reshapeData.append(eachData)
        reshapeData = np.array(reshapeData, dtype='float64')
        dataList.append(reshapeData)
        #dataList.append(np.array([initData[cutDataLog[i][0]:cutDataLog[i][1]][:,k] for k in chanDic.keys()],dtype='float64').T)

    hc = hot_code(list(eventTypeDic.keys()))
    label_2d=hc.one_hot_encode(labelLog)
    #return np.array(dataList, dtype='float64'), label_2d
    return dataList,label_2d

def getData_1DChan(yml,initData,eventData,windowSize,eventTypeDic,chanDic):
    dataShape=initData.shape
    cutDataLog=[]
    labelLog=[]
    step = yml['Meta']['step']
    doNormalize = False
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime,curTime + windowSize])
            labelLog.append(eventData[i][2])
            #labelLog.extend([eventData[i][2] for k in range(0,windowSize)])
            curTime = curTime + step
    eventLen=len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
        cutDataLog.append([curTime,curTime + windowSize])
        labelLog.append(eventData[eventLen - 1][2])
        #labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
        curTime = curTime + step

    labelLog=np.array(labelLog,dtype='int32')

    # Equal Example
    keys=np.unique(eventData[:,2])
    cutDataLog,labelLog=equalData(keys,cutDataLog,labelLog)

    dataList=[]

    for i in range(len(cutDataLog)):
        t_data=np.array([initData[cutDataLog[i][0]:cutDataLog[i][1]][:, k] for k in chanDic.keys()], dtype='float64').T
        if doNormalize:
            scaler=StandardScaler()
            scaler=scaler.fit(t_data)
            t_data=scaler.transform(t_data)
        dataList.append(t_data)

    hc = hot_code(list(eventTypeDic.keys()))
    label_2d=hc.one_hot_encode(labelLog)
    #return np.array(dataList, dtype='float64'), label_2d
    return dataList,label_2d
#(yml,raw, events,channelDic,eventTypeDic)
def getData_1DChan_2Class(yml,initData,eventData,windowSize,eventTypeDic,chanDic):
    dataShape=initData.shape
    cutDataLog=[]
    labelLog=[]
    step = yml['Meta']['step']
    doNormalize = False
    train_eventId=yml['Meta']['train_eventId']
    if 'input_normalize' in yml['Meta']:
        doNormalize = yml['Meta']['input_normalize']
    for i in range(len(eventData) - 1):
        if eventData[i][0] == eventData[i + 1][0]:
            continue
        if eventData[i][2] not in train_eventId:
            continue
        curTime = int(eventData[i][0])
        while curTime + windowSize <= eventData[i][0] + eventData[i][1] and curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime,curTime + windowSize])
            labelLog.append(eventData[i][2])

            curTime = curTime + step
    eventLen=len(eventData)

    curTime = int(eventData[eventLen - 1][0])
    if eventData[eventLen - 1][2] in train_eventId:
        while curTime + windowSize < dataShape[0]:  # curTime + windowSize <= eventData[i+1][0]
            cutDataLog.append([curTime,curTime + windowSize])
            labelLog.append(eventData[eventLen - 1][2])
            #labelLog.extend([eventData[eventLen - 1][2] for k in range(0, windowSize)])
            curTime = curTime + step

    labelLog=np.array(labelLog,dtype='int32')

    # Equal Example
    keys=np.unique(eventData[:,2])
    cutDataLog,labelLog=equalData(keys,cutDataLog,labelLog)

    dataList=[]

    for i in range(len(cutDataLog)):
        t_data=np.array([initData[cutDataLog[i][0]:cutDataLog[i][1]][:, k] for k in chanDic.keys()], dtype='float64').T
        if doNormalize:
            scaler=StandardScaler()
            scaler=scaler.fit(t_data)
            t_data=scaler.transform(t_data)
        dataList.append(t_data)

    hc = hot_code(list(eventTypeDic.keys()))
    label_2d=hc.one_hot_encode(labelLog)
    #return np.array(dataList, dtype='float64'), label_2d
    return dataList,label_2d

def equalData(keys,datas,labels_1d,doShuffle=True,minCount = 100000000000):
    # Equal Example
    groupEventIndex = {}
    for key in keys:
        try:
            temp = np.where(labels_1d[:] == key)
            if len(temp[0])>0:
                groupEventIndex.setdefault(key, temp[0])
                if minCount > len(temp[0]):
                    minCount = len(temp[0])
        except Exception as err:
            print(key)
    removeIndex = []
    for key, value in groupEventIndex.items():
        if len(value) > minCount:
            valueIndex = random.sample(range(0, len(value)), len(value) - minCount)
            removeIndex.extend(value[i] for i in valueIndex)

    datas = np.delete(datas, removeIndex, axis=0)
    labels_1d = np.delete(labels_1d, removeIndex, axis=0)
    if doShuffle:
        datas,labels_1d=shuffle(datas,labels_1d)
    return datas,labels_1d

def equalSegDicData(segDic):
    keyLogDic={}
    for key,value in segDic.items():
        keyLogDic.setdefault(key,[])
        for index in range(len(value)):
            for log in value[index]['SegLog']:
                keyLogDic[key].append({'SegLog':log,'Index':index})

    keySegCountList=[len(v) for v in keyLogDic.values()]
    minCount = min(keySegCountList)
    for key,value in keyLogDic.items():
        if len(value)>minCount:
            valueIndex = random.sample(range(0, len(value)), len(value) - minCount)
            keyLogDic[key]=np.delete(value,valueIndex)

    return keyLogDic

#添加getEvent等方法只是为了test OpenBMIEvents
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

def get_26MI_1D():
    return {'FC5':1,'FC3':2,'FC1':3,'FCz':4,'FC2':5,'FC4':6,'FC6':7,'FT7':8,'FT8':9,'C5':10,'C3':11,'C1':12,'Cz':13,'C2':14,
            'C4':15,'C6':16,'T7':17,'T8':18,'CP5':19,'CP3':20,'CP1':21,'CP2':22,'CP4':23,'CP6':24,'TP7':25,'TP8':26}

def get_20MI_1D():#OpenBMI左右对称
    return {'FC5':7,'FC3':32,'FC1':8,'FC2':9,'FC4':33,'FC6':10,'C5':34,'C3':12,'C1':35,'Cz':13,'C2':36,'C4':14,'C6':37,'CP5':17,
            'CP3':38,'CP1':18,'CPz':39,'CP2':19,'CP4':40,'CP6':20}

def get_31MI_1D():#OpenBMI31通道
    return {'FC5': 7, 'FC3': 32, 'FC1': 8, 'FC2': 9, 'FC4': 33, 'FC6': 10,'T7':11, 'C5': 34, 'C3': 12, 'C1': 35, 'Cz': 13,
            'C2': 36, 'C4': 14, 'C6': 37, 'T8':15,'TP7': 47,'CP5':17,'CP3': 38, 'CP1': 18, 'CPz': 39, 'CP2': 19, 'CP4': 40, 'CP6': 20,
            'TP8':52,'P7':22,'P3':23,'P1':41,'Pz':24,'P2':42,'P4':25,'P8':26}