# 신경망 학습 1 - 손실 함수

import sys, os, numpy as np

sys.path.append(os.getcwd())
from dataset.mnist import load_mnist


# 손실함수란: 신경망의 성능을 분석하기 위한 지표


# 손실함수 테스트용 데이터
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 정답은 2
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]


# 오차제곱합: 고딩 때 배운 분산과 유사
def sum_squares_error(y, t):
    # y가 신경망의 추정값, t가 정답(테스트)레이블이다.
    return 0.5 * np.sum((y - t) ** 2)


# y2의 결과가 더 높게 나온다! (오차)
print(
    "오차제곱합 계산: y1이 정답에 가까움",
    sum_squares_error(np.array(y1), np.array(t)),
    sum_squares_error(np.array(y2), np.array(t)),
    sep="\n",
)


# 교차 엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7  # y가 0이 되어 -inf가 되지 않도록 방지함
    return -np.sum(t * np.log(y + delta))


# y2의 결과가 더 높게 나온다! (오차)
print(
    "교차 엔트로피 오차 계산: y1이 정답에 가까움",
    cross_entropy_error(np.array(y1), np.array(t)),
    cross_entropy_error(np.array(y2), np.array(t)),
    sep="\n",
)
