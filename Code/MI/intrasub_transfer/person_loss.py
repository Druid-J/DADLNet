import numpy as np
from math import sqrt
from keras.engine.topology import Layer
from keras.losses import MeanAbsoluteError
from keras.layers import Softmax, Dense



def pearson(input):
    v1 = input[:64]
    v2 = input[64:]
    length = v1.shape[1]
    Exy = np.diagonal(np.matmul(v1, v2.T)) / length
    Ex = np.sum(v1, axis=1) / length
    Ey = np.sum(v2, axis=1) / length
    C = Exy - Ex * Ey
    res = C/sqrt(np.var(v1, ddof=1, axis=1)*np.var(v2, ddof=1, axis=1))
    return res


class Dense_with_person(Layer):
    def __init__(self, **kwargs):
        super(Dense_with_person, self).__init__()

    def call(self, inputs, **kwargs):
        v1, v2 = inputs
        Exy = np.diagonal(np.matmul(v1, v2.T)) / v1.shape[1]
        Ex = np.sum(v1, axis=1) / v1.shape[1]
        Ey = np.sum(v2, axis=1) / v1.shape[1]
        res = (Exy - Ex * Ey) / sqrt(np.var(v1, ddof=1, axis=1) * np.var(v2, ddof=1, axis=1))
        return res






