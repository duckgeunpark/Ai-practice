"""
[Step 3/5] 모델 학습
실행: python CNN_train_PyTorch.py
필요: data/ (Step 1), data/model_init.pth (Step 2)
결과: data/model_best.pth, data/history.pth 저장
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── 모델 클래스 정의 (동일 구조 필요) ──────────────────────
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ── 데이터 로더 ──────────────────────────────────────────
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]
)


def main():
    # Step 1~2 결과 확인
    if not os.path.exists(DATA_DIR):
        print("❌ data/ 폴더가 없습니다. 먼저 CNN_dataset_PyTorch.py 를 실행하세요.")
        return
    if not os.path.exists(os.path.join(DATA_DIR, "model_init.pth")):
        print(
            "❌ data/model_init.pth 가 없습니다. 먼저 CNN_model_PyTorch.py 를 실행하세요."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"훈련 배치 수: {len(train_loader)}")
    print(f"테스트 배치 수: {len(test_loader)}")

    # ── 모델 로드 & 학습 설정 ────────────────────────────────
    model = CIFAR10_CNN().to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(DATA_DIR, "model_init.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    print("✔ 모델 로드 완료")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 학습률 스케줄러 — 10 epoch마다 학습률 × 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 학습 루프 ────────────────────────────────────────────
    train_losses = []
    val_accuracies = []
    best_acc = 0.0

    for epoch in range(50):
        # 학습
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ── 검증 ─────────────────────────────────────
        model.eval()
        correct = 0
        total_cnt = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_cnt += y_batch.size(0)

        acc = correct / total_cnt
        val_accuracies.append(acc)

        scheduler.step()

        # 최고 모델 저장
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "model_best.pth"))

        print(
            f"Epoch {epoch+1:2d}/50 | Loss: {avg_loss:.4f} | "
            f"Val Acc: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # 히스토리 저장
    torch.save(
        {"train_losses": train_losses, "val_accuracies": val_accuracies},
        os.path.join(DATA_DIR, "history.pth"),
    )

    print(f"\n최고 테스트 정확도: {best_acc:.4f}")
    print("✔ 학습 완료!")
    print(f"  → 모델: data/model_best.pth")
    print(f"  → 히스토리: data/history.pth")


if __name__ == "__main__":
    main()
