"""
[Step 4/5] 학습 곡선 시각화
실행: python CNN_plot_PyTorch.py
필요: data/history.pth (Step 3에서 생성)
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(os.path.join(DATA_DIR, "history.pth")):
    print("❌ data/history.pth 가 없습니다. 먼저 CNN_train_PyTorch.py 를 실행하세요.")
    exit(1)

hist = torch.load(os.path.join(DATA_DIR, "history.pth"), weights_only=True)
print("✔ 학습 히스토리 로드 완료")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(hist["train_losses"], label="훈련 손실", color="blue")
ax1.set_title("손실 변화")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(hist["val_accuracies"], label="검증 정확도", color="green")
ax2.set_title("정확도 변화")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
