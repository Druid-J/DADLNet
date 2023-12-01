# from keras.layers import (Dense, Dropout,Flatten,Input,Conv1D,MaxPooling1D,Concatenate,GlobalAveragePooling1D,Permute,
# 						  Reshape,Lambda,Conv2D,Multiply,MaxPool2D,SpatialDropout2D,Add,Softmax,BatchNormalization,Activation,DepthwiseConv2D)
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Attention
from tensorflow.keras.constraints import max_norm
from meya.basicModel import modelComFun
from tensorflow.keras import regularizers, constraints
from tensorflow.keras import backend as K
from meya.plot import plotModel
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as bk
# from keras_self_attention import SeqSelfAttention

# bk.set_image_data_format("channels_first")

class ML_Model(modelComFun):
    '''
    dataShape doesn't need to set when testing
    '''
    def __init__(self, yml, eventDic,dataShape=None,params=None):
        self.head_cols = None
        self.model = None
        self.cross_time = 0
        layerType = yml['ML']['layerType']
        # if 'downSampling' in yml['Meta']:
        #     dataShape = [yml['ML']['batch_size'], int(yml['Meta']['downSampling'] * yml['Meta']['seconds'])]
        # else:
        #     dataShape = [yml['ML']['batch_size'], int(yml['Meta']['frequency'] * yml['Meta']['seconds'])]
        # dataShape.extend(yml['Meta']['channelShape'])
        self.dataShape = dataShape
        class_num = len(eventDic)
        if 'train_eventId' in yml['Meta']:
            class_num = len(yml['Meta']['train_eventId'])
        self.class_num = class_num
        # self.custom_objects={'Attention':Attention}
        if layerType.find('two1DInput') > -1:
            channelShape = yml['Meta']['channelShape']
            self.head_cols = [0, channelShape[0], 2 * channelShape[0]]
            self.testDataFun = self.mulInputs_1D
        elif layerType.find('split1DInput_') > -1:
            if 'downSampling' in yml['Meta']:
                self.cross_time = int(yml['Meta']['forcast'] * yml['Meta']['downSampling'] / yml['Meta']['frequency'])
            else:
                self.cross_time = yml['Meta']['forcast']
            self.testDataFun = self.splitInputs_crossTime
        elif layerType.find('mulInput') > -1:
            self.split_shapeIndex = 2
            self.testDataFun = self.split2MulInput
            self.save_step_batch = 10
            if 'save_step_batch' in yml['ML']:
                self.save_step_batch = yml['ML']['save_step_batch']

        self.head_groups = 1
        if 'head_groups' in yml['Meta']:
            self.head_groups = yml['Meta']['head_groups']
            self.testDataFun = self.splitInputs_group


    def EEGNet(self, yml, classNum, Chans=64, Samples=128,
               dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        Chans = self.dataShape[0]
        Samples = self.dataShape[1]
        dropoutRate = yml['ML']['dropout_rate']
        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
        input1 = Input(shape=self.dataShape)
        #频谱特征
        reshape1 = Reshape((Chans, Samples, 1))(input1)
        # input1 = Input(shape=(1, Chans, Samples))
        if 'kernLength' in yml['ML']:
            kernLength=yml['ML']['kernLength']
        if 'F1' in yml['ML']:
            F1=yml['ML']['F1']
        #F1 8  kernlength:64
        ##################################################################
        block1 = Conv2D(F1, (1, kernLength), padding='same',
                        input_shape=(Chans, Samples,1),
                        use_bias=False)(reshape1)
        block1 = BatchNormalization(axis=-1)(block1)
        #时域特征
        block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
        # block1 = DepthwiseConv2D((1, 1), use_bias=False, depth_multiplier=8, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization(axis=-1)(block1)
        block1 = Activation('elu')(block1)
        # block1 = AveragePooling2D((1, Chans))(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(dropoutRate)(block1)
        #添加key-value attention


        #可分离卷积网络F2:16
        block2 = SeparableConv2D(F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization(axis=-1)(block2)
        block2 = Activation('elu')(block2)
        # block2 = AveragePooling2D((1, 1))(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(dropoutRate)(block2)

        flatten = Flatten(name='flatten')(block2)

        if yml['ML']['loss'] == 'binary_crossentropy':
            dense = Dense(1, activation=yml['ML']['activation'])(flatten)
            model = Model(inputs=input1, outputs=dense)
        else:
            dense = Dense(classNum, name='dense',
                          kernel_constraint=max_norm(norm_rate))(flatten)
            softmax = Activation('softmax', name='softmax')(dense)
            model = Model(inputs=input1, outputs=softmax)
        model.compile(loss=self.getLoss(yml), optimizer=self.getOptimize(yml), metrics=[self.metrics])
        plotModel(yml, model)
        self.model = model


    def DADLNet_NNI(self, yml, classNum, input_channel_1=4, input_channel_2=9, input_timestep=400,params=None,fileName=None):
        """
        # Muti分支感受野最小 0.125比例, 第一层卷积之后空间注意力和时间注意力   SATNet
        """

        ###Spatial-Temporal Attention
        def channel_wise_mean(x):
            mid = K.mean(x, axis=-1)
            return mid

        def spatial_attention(input_tensor, index):
            x = Lambda(channel_wise_mean)(input_tensor)  # input[1] = 空间维度1 input[2] = 空间维度2
            x = Reshape([K.int_shape(input_tensor)[1], K.int_shape(
                input_tensor)[2], K.int_shape(input_tensor)[3], 1], name='AttReshape%d' % index)(x)

            nbSpatial = K.int_shape(input_tensor)[2] * K.int_shape(input_tensor)[3]
            spatial = AveragePooling3D(
                pool_size=[K.int_shape(input_tensor)[1], 1, 1], name='SpaPool%d' % index)(x)
            spatial = Flatten(name='SapFlatten%d' % index)(spatial)
            spatial = Dense(nbSpatial)(spatial)
            spatial = Reshape(
                [1, K.int_shape(input_tensor)[2], K.int_shape(input_tensor)[3], 1], name='SpaRe%d' % index)(spatial)

            return spatial

        def SE_Layer(input_tensor, ratio=params['ratio']):

            channel = K.int_shape(input_tensor)[-1]
            squeeze = GlobalAveragePooling3D()(input_tensor)

            excitation = Dense(units=channel // ratio)(squeeze)
            excitation = Activation('relu')(excitation)
            excitation = Dense(units=channel)(excitation)
            se_feature = Reshape((1, 1, 1, channel))(excitation)
            return se_feature

        def combine(input_feature, index, ratio=params['ratio']):
            channel_feature = SE_Layer(input_feature, ratio)
            spatial_feature = spatial_attention(input_feature, index)
            combined = Add()([channel_feature, spatial_feature])
            att = Activation('sigmoid')(combined)
            feature = Multiply()([input_feature, att])
            feature = Add()([input_feature, feature])
            return feature

        input_shape = (input_timestep, input_channel_1, input_channel_2)
        input = Input(shape=input_shape)
        inputs = Reshape((input_timestep, input_channel_1, input_channel_2, 1))(input)
        conv3d_1 = Conv3D(filters=params['features_1'], kernel_size=(50, 2, 2), strides=(1, 1, 1),  kernel_initializer='glorot_uniform',
                          padding='same', kernel_regularizer=regularizers.l2(l=params['l2']))(inputs)
        conv3d_1 = BatchNormalization()(conv3d_1)
        conv3d_1 = Activation('elu')(conv3d_1)
        Pool3D_1 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_1)  # 334 500
        Pool3D_1 = Dropout(params['drop'])(Pool3D_1)
        Pool3D_1 = combine(Pool3D_1, 0)

        conv3d_2 = Conv3D(filters=params['features_2'], kernel_size=(15, 2, 2), strides=(1, 2, 2), kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=params['l2']))(Pool3D_1)  # 294  314 470
        conv3d_2 = BatchNormalization()(conv3d_2)
        conv3d_2 = Activation('elu')(conv3d_2)
        Pool3D_2 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_2)  # 98  105  235
        Pool3D_2 = Dropout(params['drop'])(Pool3D_2)

        conv3d_3 = Conv3D(filters=params['features_3'], kernel_size=(5, 1, 2), strides=(1, 1, 2), kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=params['l2']))(Pool3D_2)  # 86 100 221
        conv3d_3 = BatchNormalization()(conv3d_3)
        conv3d_3 = Activation('elu')(conv3d_3)
        Pool3D_3 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_3)  # 34 111
        Pool3D_3 = Dropout(params['drop'])(Pool3D_3)

        conv3d_4 = Conv3D(filters=params['features_4'], kernel_size=(1, 2, 2), strides=(1, 2, 2), kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=params['l2']), padding='same')(Pool3D_3)  # 86 100 221
        conv3d_4 = BatchNormalization()(conv3d_4)
        conv3d_4 = Activation('elu')(conv3d_4)
        Pool3D_4 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_4)  # 34 111
        Pool3D_4 = Dropout(params['drop'])(Pool3D_4)

        reshape1 = Reshape((4, 1, -1))(Pool3D_4)

        GAP_1 = GlobalAveragePooling2D()(reshape1)  # 53

        # 增加Dense层
        if yml['ML']['loss'] == 'binary_crossentropy':
            dense_3 = Dense(1, name='dense3')(GAP_1)
            sigmoid = Activation('sigmoid', name='sigmoid')(dense_3)
            model = Model(inputs=input, outputs=sigmoid)
        else:
            dense_3 = Dense(classNum, name='dense3')(GAP_1)
            softmax = Activation('softmax', name='softmax')(dense_3)
            model = Model(inputs=input, outputs=softmax)

        model.compile(loss=self.getLoss(yml), optimizer=self.getOptimize(yml), metrics=self.getMetrics(yml))
        plotModel(yml,model,fileName)
        self.model = model

    def DADLNet(self, yml, classNum, input_channel_1=4, input_channel_2=9, input_timestep=400,fileName=None):
        """
        # Muti分支感受野最小 0.125比例, 第一层卷积之后空间注意力和时间注意力   SATNet
        """

        ##Spatial-Temporal Attention
        def channel_wise_mean(x):
            mid = K.mean(x, axis=-1)
            return mid

        def spatial_attention(input_tensor, index):
            x = Lambda(channel_wise_mean)(input_tensor)  # input[1] = 空间维度1 input[2] = 空间维度2
            x = Reshape([K.int_shape(input_tensor)[1], K.int_shape(
                input_tensor)[2], K.int_shape(input_tensor)[3], 1], name='AttReshape%d' % index)(x)

            nbSpatial = K.int_shape(input_tensor)[2] * K.int_shape(input_tensor)[3]
            spatial = AveragePooling3D(
                pool_size=[K.int_shape(input_tensor)[1], 1, 1], name='SpaPool%d' % index)(x)
            spatial = Flatten(name='SapFlatten%d' % index)(spatial)
            spatial = Dense(nbSpatial)(spatial)
            spatial = Reshape(
                [1, K.int_shape(input_tensor)[2], K.int_shape(input_tensor)[3], 1], name='SpaRe%d' % index)(spatial)

            return spatial

        def SE_Layer(input_tensor, ratio=16):

            channel = K.int_shape(input_tensor)[-1]
            squeeze = GlobalAveragePooling3D()(input_tensor)

            excitation = Dense(units=channel // ratio)(squeeze)
            excitation = Activation('relu')(excitation)
            excitation = Dense(units=channel)(excitation)
            se_feature = Reshape((1, 1, 1, channel))(excitation)
            return se_feature

        def combine(input_feature, index, ratio=16):
            channel_feature = SE_Layer(input_feature, ratio)
            spatial_feature = spatial_attention(input_feature, index)
            combined = Add()([channel_feature, spatial_feature])
            att = Activation('sigmoid')(combined)
            feature = Multiply()([input_feature, att])
            feature = Add()([input_feature, feature])
            return feature

        input_shape = (input_timestep, input_channel_1, input_channel_2)
        input = Input(shape=input_shape)
        inputs = Reshape((input_timestep, input_channel_1, input_channel_2, 1))(input)
        conv3d_1 = Conv3D(filters=16, kernel_size=(50, 2, 2), strides=(1, 1, 1),  kernel_initializer='glorot_uniform',
                          padding='same', kernel_regularizer=regularizers.l2(l=0.001))(inputs)
        conv3d_1 = BatchNormalization()(conv3d_1)
        conv3d_1 = Activation('elu')(conv3d_1)
        Pool3D_1 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_1)  # 334 500
        Pool3D_1 = Dropout(0.5)(Pool3D_1)
        Pool3D_1 = combine(Pool3D_1, 0)

        conv3d_2 = Conv3D(filters=16, kernel_size=(15, 2, 2), strides=(1, 2, 2), kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=0.001))(Pool3D_1)  # 294  314 470
        conv3d_2 = BatchNormalization()(conv3d_2)
        conv3d_2 = Activation('elu')(conv3d_2)
        Pool3D_2 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_2)  # 98  105  235
        Pool3D_2 = Dropout(0.5)(Pool3D_2)

        conv3d_3 = Conv3D(filters=16, kernel_size=(5, 1, 2), strides=(1, 1, 2), kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=0.001))(Pool3D_2)  # 86 100 221
        conv3d_3 = BatchNormalization()(conv3d_3)
        conv3d_3 = Activation('elu')(conv3d_3)
        Pool3D_3 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_3)  # 34 111
        Pool3D_3 = Dropout(0.5)(Pool3D_3)

        conv3d_4 = Conv3D(filters=16, kernel_size=(1, 2, 2), strides=(1, 2, 2), kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(l=0.001), padding='same')(Pool3D_3)  # 86 100 221
        conv3d_4 = BatchNormalization()(conv3d_4)
        conv3d_4 = Activation('elu')(conv3d_4)
        Pool3D_4 = AveragePooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same')(conv3d_4)  # 34 111
        Pool3D_4 = Dropout(0.5)(Pool3D_4)

        reshape1 = Reshape((2, 1, -1))(Pool3D_4)

        GAP_1 = GlobalAveragePooling2D()(reshape1)  # 53

        # 增加Dense层
        if yml['ML']['loss'] == 'binary_crossentropy':
            dense_3 = Dense(1, name='dense3')(GAP_1)
            sigmoid = Activation('sigmoid', name='sigmoid')(dense_3)
            model = Model(inputs=input, outputs=sigmoid)
        else:
            dense_3 = Dense(classNum, name='dense3')(GAP_1)
            softmax = Activation('softmax', name='softmax')(dense_3)
            model = Model(inputs=input, outputs=softmax)

        model.compile(loss=self.getLoss(yml), optimizer=self.getOptimize(yml), metrics=self.getMetrics(yml))
        # plotModel(yml,model,fileName)
        self.model = model


