"""
[Step 5/5] 예측 & 평가
실행: python CNN_prediction_PyTorch.py
필요: data/ (Step 1), data/model_best.pth (Step 3)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── 모델 클래스 정의 (동일 구조 필요) ──────────────────────
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# ── 데이터 & 모델 로드 ───────────────────────────────────
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

class_names = [
    "비행기", "자동차", "새", "고양이", "사슴",
    "개", "개구리", "말", "배", "트럭",
]


def main():
    for f, step in [("data", "CNN_dataset_PyTorch.py"),
                    (os.path.join("data", "model_best.pth"), "CNN_train_PyTorch.py")]:
        full = os.path.join(BASE_DIR, f)
        if not os.path.exists(full):
            print(f"❌ {f} 가 없습니다. 먼저 {step} 를 실행하세요.")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = CIFAR10_CNN().to(device)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "model_best.pth"), map_location=device, weights_only=True))
    model.eval()
    print("✔ 데이터 & 학습된 모델 로드 완료")

    # ── 전체 테스트 정확도 ───────────────────────────────────
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"최종 테스트 정확도: {(all_preds == all_labels).mean():.4f}")

    # ── 예측 결과 시각화 ─────────────────────────────────────
    test_images, test_lbls = next(iter(test_loader))
    with torch.no_grad():
        preds = model(test_images.to(device)).argmax(dim=1).cpu()

    # 역정규화 (시각화를 위해 원래 픽셀값으로 복원)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        img = test_images[i].permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0, 1)  # 역정규화
        ax.imshow(img)
        pred = class_names[preds[i]]
        true = class_names[test_lbls[i]]
        color = "blue" if preds[i] == test_lbls[i] else "red"
        ax.set_title(f"예측: {pred}\n정답: {true}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("파란색: 정답  /  빨간색: 오답")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
