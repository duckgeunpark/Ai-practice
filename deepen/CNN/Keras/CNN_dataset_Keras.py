"""
[Step 1/5] CIFAR-10 데이터 불러오기 & 전처리
실행: python CNN_dataset_Keras.py
결과: data/data.npz 저장
"""

import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 데이터 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)  # (50000, 32, 32, 3) — 5만 장, 컬러
print(X_test.shape)   # (10000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)

# 클래스 이름
class_names = [
    "비행기", "자동차", "새", "고양이", "사슴",
    "개", "개구리", "말", "배", "트럭",
]

# 샘플 시각화
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(class_names[y_train[i][0]])
    ax.axis("off")
plt.suptitle("CIFAR-10 샘플 이미지")
plt.tight_layout()
plt.show()

# 정규화 — 픽셀값 0~255 → 0~1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 레이블 1차원으로
y_train = y_train.flatten()
y_test = y_test.flatten()

print(X_train.min(), X_train.max())  # 0.0  1.0

# 전처리된 데이터 저장
np.savez(os.path.join(DATA_DIR, "data.npz"),
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test)
print(f"\n✔ 데이터 저장 완료 → {DATA_DIR}/data.npz")
