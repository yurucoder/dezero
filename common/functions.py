import numpy as np


# 계단 함수
def step_function_(x: np.ndarray):
    return np.array(x > 0, dtype=np.int32)


# 시그모이드 함수
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


# ReLU 함수
def ReLU(x: np.ndarray):
    return np.maximum(0, x)


# 항등 함수
def identity_function(x: np.ndarray):
    return x


# 소프트맥스 함수
def softmax(a: np.ndarray):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
