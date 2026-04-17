"""
[Step 1/5] CIFAR-10 데이터 다운로드 & 샘플 확인
실행: python CNN_dataset_PyTorch.py
결과: data/ 폴더에 CIFAR-10 다운로드
"""

import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

# 전처리 정의 (시각화용 — 정규화 없이)
transform_raw = transforms.Compose([transforms.ToTensor()])

# 데이터셋 다운로드
raw_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform_raw, download=True)
test_raw = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform_raw, download=True)

class_names = [
    "비행기", "자동차", "새", "고양이", "사슴",
    "개", "개구리", "말", "배", "트럭",
]

# 샘플 시각화
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    img, label = raw_dataset[i]
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(class_names[label])
    ax.axis("off")
plt.suptitle("CIFAR-10 샘플 이미지")
plt.tight_layout()
plt.show()

print(f"훈련 데이터: {len(raw_dataset)}장")
print(f"테스트 데이터: {len(test_raw)}장")
print(f"\n✔ 데이터 다운로드 완료 → {DATA_DIR}")
