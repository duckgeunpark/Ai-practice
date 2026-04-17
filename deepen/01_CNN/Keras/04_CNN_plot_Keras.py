"""
[Step 4/5] 학습 곡선 시각화
실행: python CNN_plot_Keras.py
필요: data/history.pkl (Step 3에서 생성)
"""

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Step 3 결과 확인
if not os.path.exists(os.path.join(DATA_DIR, "history.pkl")):
    print("❌ data/history.pkl 이 없습니다. 먼저 CNN_compile_Keras.py 를 실행하세요.")
    exit(1)

# 히스토리 로드
with open(os.path.join(DATA_DIR, "history.pkl"), "rb") as f:
    hist = pickle.load(f)

print("✔ 학습 히스토리 로드 완료")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 정확도
ax1.plot(hist["accuracy"], label="훈련 정확도")
ax1.plot(hist["val_accuracy"], label="검증 정확도", linestyle="--")
ax1.set_title("정확도 변화")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# 손실
ax2.plot(hist["loss"], label="훈련 손실")
ax2.plot(hist["val_loss"], label="검증 손실", linestyle="--")
ax2.set_title("손실 변화")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
