"""
[Step 2/5] CNN 모델 구성
실행: python CNN_model_PyTorch.py
결과: data/model_init.pth 저장
"""

import os
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 블록 1: 입력 채널 3 → 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3,32,32) → (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → (32,16,16)
            nn.Dropout2d(0.25),
        )

        # 블록 2: 32 → 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → (64,8,8)
            nn.Dropout2d(0.25),
        )

        # 블록 3: 64 → 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → (128,4,4)
            nn.Dropout2d(0.25),
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 128×4×4 = 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),  # CrossEntropyLoss가 Softmax 포함
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


model = CIFAR10_CNN().to(device)

# 파라미터 수 확인
total = sum(p.numel() for p in model.parameters())
print(f"전체 파라미터 수: {total:,}")

# 초기 가중치 저장
torch.save(model.state_dict(), os.path.join(DATA_DIR, "model_init.pth"))
print("\n✔ 모델 저장 완료 → data/model_init.pth")
