# 신경망 4 - 손글씨 숫자 인식

import sys, os

sys.path.append(os.getcwd())
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist


# 이미 학습된 매개변수를 사용하여 학습 과정 없이 추론 과정만 구현
# 이 추론 과정을 '순전파' (forward propagation)이라고 한다


# 넘파이로 저장된 이미지를 PIL 객체로 변환해야 함 (flatten=True)
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 훈련된 이미지는 넘파이 객체로 피클 파일에 저장
# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,) - 1차원으로 압축된 이미지 데이터
img = img.reshape(28, 28)  # 원래 이미지의 모양으로 변형
print(img.shape)  # (28, 28) = 2차원 이미지 데이터

# img_show(img)
