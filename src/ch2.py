# 퍼셉트론

import numpy as np


# AND 게이트는 (1, 1)에서만 1을 출력한다.
def AND(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7  # 편향
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


AND(1, 1)  # 1
AND(1, 0)  # 0
AND(0, 1)  # 0
AND(0, 0)  # 0


# NAND 게이트는 AND게이트와 반대인 진리표를 갖는다.
def NAND(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([-0.5, -0.5])  # 가중치(음수)
    b = 0.7  # 편향
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


NAND(1, 1)  # 0
NAND(1, 0)  # 1
NAND(0, 1)  # 1
NAND(0, 0)  # 1


# OR 게이트는 입력 중 하나라도 1이 있다면 1을 출력한다.
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # AND와 가중치만 다르다!
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


OR(1, 1)  # 1
OR(1, 0)  # 1
OR(0, 1)  # 1
OR(0, 0)  # 0


# XOR(배타적 논리합)은 선형적으로 영역을 나눌 수 없어 다층 퍼셉트론으로 구현한다.
def XOR(x1, x2):  # 0층 (x1, x2)
    s1 = NAND(x1, x2)  # 1층 (s1, s2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)  # 2층 (y)
    return y  # NAND와 OR의 교집합(AND)이 XOR이 된다.


XOR(0, 0)  # 0
XOR(1, 0)  # 1
XOR(0, 1)  # 1
XOR(1, 1)  # 0
