import numpy as np


# 계단 함수
def step_function_(x: np.ndarray):
    return np.array(x > 0, dtype=np.int32)


# 시그모이드 함수
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


# ReLU 함수
def ReLU(x):
    return np.maximum(0, x)


# 항등함수
def identity_function(x):
    return x
