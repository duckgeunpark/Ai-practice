## 0. 시리즈

| 심화편 | 제목 | 난이도 | 역할 |
| :-- | :-- | :-- | :-- |
| [**심화 1편**](https://duckport.pages.dev/posts/CNN_Deep)⬅️ | CNN — 이미지 분류 완벽 정리 | ⭐⭐ | 이미지 처리 |
| [심화 2편](https://duckport.pages.dev/posts/RNN) | RNN / LSTM — 시계열·텍스트 처리 | ⭐⭐⭐ | 순서 데이터 |
| [심화 3편](https://duckport.pages.dev/posts/Transformer) | Transformer 구조 완벽 정리 | ⭐⭐⭐⭐ | 현대 AI 핵심 |
| [심화 4편](https://duckport.pages.dev/posts/BERT) | BERT / GPT 원리와 활용 | ⭐⭐⭐⭐ | 언어 모델 |
| [심화 5편](https://duckport.pages.dev/posts/Finetuning_Deep) | HuggingFace로 LLM Fine-tuning | ⭐⭐⭐⭐⭐ | LLM 실전 |


***

## 1. CNN이란?

### 1-1. 왜 이미지에는 일반 신경망이 부족한가?

트랙 2 5·6편에서 사용한 `Dense` 레이어는 픽셀을 **1차원으로 펼쳐서(Flatten)** 처리합니다.

```
28×28 이미지 → 784개 픽셀을 1줄로 나열 → Dense 입력
```

이 방식에는 두 가지 치명적인 문제가 있습니다.

- **공간 정보 손실** — "눈"과 "코"가 서로 어디에 위치하는지, 픽셀 간 관계를 모두 잃음
- **파라미터 폭발** — 224×224 컬러 이미지라면 입력만 `224×224×3 = 150,528`개, Dense 연결 시 수억 개의 파라미터 필요


### 1-2. CNN이 해결하는 방식

**CNN(Convolutional Neural Network, 합성곱 신경망)** 은 이미지를 **펼치지 않고**, 작은 필터(커널)로 이미지를 훑으며 패턴을 추출합니다.

```
원본 이미지 (2D)
      ↓  필터가 슬라이딩하며 훑음
특징 맵 (Feature Map) — 엣지, 곡선, 텍스처 등
      ↓  풀링으로 크기 축소
더 압축된 특징 맵
      ↓  Flatten → Dense
최종 분류 결과
```

- **파라미터 공유** — 하나의 필터를 이미지 전체에 재사용 → 파라미터 수 대폭 절감
- **위치 불변성** — 고양이가 왼쪽에 있든 오른쪽에 있든 같은 필터로 탐지


### 1-3. 트랙 2와의 연결

| 트랙 2 (Dense) | 트랙 3 (CNN) |
| :-- | :-- |
| `Flatten` → `Dense` | `Conv2D` → `Pooling` → `Flatten` → `Dense` |
| 픽셀을 1줄로 나열 | 이미지 공간 구조 유지하며 처리 |
| MNIST 97~98% | CIFAR-10 75~85% (훨씬 어려운 데이터) |


***

## 2. 핵심 개념 — 합성곱(Convolution)

### 2-1. 합성곱 연산이란?

작은 **필터(커널)** 가 이미지 위를 슬라이딩하며, 겹치는 영역의 픽셀과 필터 값을 **원소별 곱셈 후 합산** 합니다.

```
    입력 이미지 (5×5)              필터 (3×3)
┌────────────┐        ┌──────┐
│    1  2  3  0  1   │        │ 1  0  1  │
│    4  5  6  1  2   │   ×    │ 0  1  0  │  →  특징 맵 (3×3)
│    7  8  9  2  3   │        │ 1  0  1  │
│    1  3  5  4  0   │        └──────┘
│    2  4  6  1  2   │
└────────────┘

왼쪽 상단 (3×3) 영역 계산:
1×1 + 2×0 + 3×1
+ 4×0 + 5×1 + 6×0   = 1+3+5+3+8 = ... → 특징 맵 [0,0] 값
+ 7×1 + 8×0 + 9×1
```

하나의 필터는 **하나의 패턴(엣지, 곡선 등)** 을 감지합니다. Conv2D 레이어에 필터를 32개, 64개씩 쌓으면 그 수만큼 다양한 패턴을 동시에 탐지합니다.

### 2-2. 특징 맵(Feature Map)이란?

필터가 이미지를 훑은 결과물입니다. 필터가 32개면 특징 맵도 32장이 생깁니다.

```
입력: (28, 28, 1)  →  Conv2D(32, 3×3)  →  출력: (26, 26, 32)
                                               ↑         ↑
                                             공간 축소   필터 수만큼
```


### 2-3. 패딩(Padding)과 스트라이드(Stride)

```python
# padding="same"  → 출력 크기 = 입력 크기 유지 (주변을 0으로 채움)
# padding="valid" → 출력 크기 < 입력 크기 (기본값, 패딩 없음)

Conv2D(32, (3, 3), padding="same")   # 28×28 → 28×28
Conv2D(32, (3, 3), padding="valid")  # 28×28 → 26×26

# strides=2 → 필터가 2칸씩 이동 → 출력 크기 절반
Conv2D(32, (3, 3), strides=2)  # 28×28 → 13×13
```

| 파라미터 | 역할 | 기본값 |
| :-- | :-- | :-- |
| `filters` | 필터(커널) 개수 = 출력 채널 수 | — |
| `kernel_size` | 필터 크기 (3,3) / (5,5) | — |
| `padding` | `"same"` / `"valid"` | `"valid"` |
| `strides` | 필터 이동 보폭 | `1` |
| `activation` | 활성화 함수 | `None` |


***

## 3. 핵심 개념 — 풀링(Pooling)

### 3-1. Max Pooling이란?

특징 맵을 **작은 영역으로 나눠서 최대값만 추출** 합니다.

```
특징 맵 (4×4)            Max Pooling (2×2, stride=2)
┌─────────┐            ┌─────┐
│  1  3  2  4   │            │  3  4  │
│  5  6  1  2   │   →→→    │  6  8  │
│  3  2  7  8   │            └─────┘
│  1  4  2  3   │
└─────────┘
→ 각 2×2 블록에서 최대값 추출 → (2×2) 출력
```


### 3-2. 왜 풀링이 필요한가?

- **공간 크기 축소** — 연산량과 파라미터 수 감소
- **위치 불변성 강화** — 패턴이 약간 이동해도 같은 값 출력
- **과적합 방지** — 불필요한 세부 정보 제거

```python
# Keras
from tensorflow.keras.layers import MaxPooling2D
MaxPooling2D(pool_size=(2, 2))  # 기본값, 크기 절반

# PyTorch
import torch.nn as nn
nn.MaxPool2d(kernel_size=2, stride=2)
```


***

## 4. CNN 전체 구조

### 4-1. Conv2D → Pooling → Flatten → Dense 흐름

```
입력 이미지
(32, 32, 3)
     │
     ▼
┌────────────┐
│  Conv2D(32, 3×3)   │  → (32, 32, 32)  특징 추출 1차
│  ReLU              │
│  MaxPooling(2×2)   │  → (16, 16, 32)  크기 절반
└────────────┘
     │
     ▼
┌────────────┐
│  Conv2D(64, 3×3)   │  → (16, 16, 64)  특징 추출 2차 (더 복잡한 패턴)
│  ReLU              │
│  MaxPooling(2×2)   │  → (8, 8, 64)    크기 절반
└────────────┘
     │
     ▼
┌────────────┐
│  Conv2D(128, 3×3)  │  → (8, 8, 128)  특징 추출 3차
│  ReLU              │
│  MaxPooling(2×2)   │  → (4, 4, 128)
└────────────┘
     │
     ▼
  Flatten             →  4×4×128 = 2,048 (1차원)
     │
     ▼
  Dense(256, relu)
  Dropout(0.5)
     │
     ▼
  Dense(10, softmax)  →  최종 분류 결과 (클래스 수)
```


### 4-2. 레이어 깊이별 학습 내용

```
얕은 층 (초반 Conv)  →  엣지, 선, 색상 경계 등 단순한 패턴
중간 층             →  눈, 코, 바퀴 등 의미 있는 부분 패턴
깊은 층 (후반 Conv)  →  얼굴, 자동차 등 고수준 개념
```


***

## 5. Keras로 CNN 구현하기

### 5-1. 데이터 준비 — CIFAR-10

MNIST(흑백 28×28)보다 어려운 **컬러 이미지(32×32×3, 10개 클래스)** 데이터셋입니다.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)  # (50000, 32, 32, 3) — 5만 장, 컬러
print(X_test.shape)   # (10000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)

# 클래스 이름
class_names = ["비행기", "자동차", "새", "고양이", "사슴",
               "개", "개구리", "말", "배", "트럭"]

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
X_test  = X_test.astype("float32")  / 255.0

# 레이블 1차원으로
y_train = y_train.flatten()
y_test  = y_test.flatten()

print(X_train.min(), X_train.max())  # 0.0  1.0
```
![Figure_1.png](/api/assets/c39ff866-9a09-4e71-86aa-cd0db2417067)

### 5-2. 모델 설계

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)

model = Sequential([
    # ── 블록 1 ──────────────────────────────────
    Conv2D(32, (3, 3), padding="same", activation="relu",
           input_shape=(32, 32, 3)),
    BatchNormalization(),          # 배치 정규화 — 학습 안정화
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # 32×32 → 16×16
    Dropout(0.25),

    # ── 블록 2 ──────────────────────────────────
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # 16×16 → 8×8
    Dropout(0.25),

    # ── 블록 3 ──────────────────────────────────
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # 8×8 → 4×4
    Dropout(0.25),

    # ── 분류기 ──────────────────────────────────
    Flatten(),                       # 4×4×128 = 2048
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax")  # 10개 클래스
])

model.summary()
```

```
출력 예시:
Layer (type)            Output Shape         Param #
────────────────────────────────────────────────────
conv2d (Conv2D)         (None, 32, 32, 32)   896
batch_norm ...          (None, 32, 32, 32)   128
conv2d_1 (Conv2D)       (None, 32, 32, 32)   9,248
...
dense_1 (Dense)         (None, 10)           5,130
────────────────────────────────────────────────────
Total params: 1,276,234
```


### 5-3. 컴파일 및 학습

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 콜백 설정
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,      # 개선 없으면 학습률을 절반으로
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

> 💡 **ReduceLROnPlateau** — 검증 손실이 개선되지 않으면 학습률을 자동으로 줄입니다. EarlyStopping과 함께 쓰면 과적합을 효과적으로 잡을 수 있습니다.

### 5-4. 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 정확도
ax1.plot(history.history["accuracy"],     label="훈련 정확도")
ax1.plot(history.history["val_accuracy"], label="검증 정확도", linestyle="--")
ax1.set_title("정확도 변화")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# 손실
ax2.plot(history.history["loss"],     label="훈련 손실")
ax2.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
ax2.set_title("손실 변화")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```
![Figure_2.png](/api/assets/5b4f8330-5f7e-4c3a-94d2-c4b71e34c58e)

### 5-5. 평가 및 예측 시각화

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 최종 정확도
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도: {test_acc:.4f}")  # 예: 0.7850 (78.5%)

# 예측
y_pred = np.argmax(model.predict(X_test), axis=1)

# 분류 리포트
print(classification_report(y_test, y_pred, target_names=class_names))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("CIFAR-10 혼동 행렬")
plt.tight_layout()
plt.show()

# 예측 결과 시각화 (맞춘 것 / 틀린 것)
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i])
    pred = class_names[y_pred[i]]
    true = class_names[y_test[i]]
    color = "blue" if y_pred[i] == y_test[i] else "red"
    ax.set_title(f"예측: {pred}\n정답: {true}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("파란색: 정답  /  빨간색: 오답")
plt.tight_layout()
plt.show()
```
![Figure_3.png|50%](/api/assets/95bec721-ccd3-434a-89ff-e5a5f183978c)
"""
|               |  precision    recall  f1-score   support

|         비행기 |     0.87      0.87      0.87      1000
|         자동차 |     0.93      0.92      0.93      1000
|            새 |     0.80      0.77      0.78      1000
|         고양이 |     0.73      0.68      0.70      1000
|          사슴 |      0.79      0.88      0.83      1000
|           개  |     0.82      0.77      0.79      1000
|         개구리 |     0.83      0.92      0.88      1000
|           말  |     0.91      0.86      0.88      1000
|           배  |     0.93      0.91      0.92      1000
|          트럭 |     0.89      0.93      0.91      1000

|    accuracy   |                         0.85     10000
|   macro avg   |     0.85      0.85      0.85     10000
|weighted avg   |     0.85      0.85      0.85     10000
"""
![Figure_4.png](/api/assets/12574cd6-9669-43fc-8216-cb0168c1437e)


***

## 6. PyTorch로 CNN 구현하기

### 6-1. 데이터 준비

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

# 전처리 정의
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 무작위 좌우 반전 (데이터 증강)
    transforms.RandomCrop(32, padding=4), # 무작위 자르기 (데이터 증강)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 채널별 평균
        std=(0.2023, 0.1994, 0.2010)     # CIFAR-10 채널별 표준편차
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# 데이터셋 다운로드
train_dataset = datasets.CIFAR10(root="./data", train=True,
                                  transform=transform_train, download=True)
test_dataset  = datasets.CIFAR10(root="./data", train=False,
                                  transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64,
                          shuffle=False, num_workers=2)

class_names = ["비행기", "자동차", "새", "고양이", "사슴",
               "개", "개구리", "말", "배", "트럭"]

print(f"훈련 배치 수: {len(train_loader)}")  # 625 (40000/64)
print(f"테스트 배치 수: {len(test_loader)}")
```
![Figure_1.png](/api/assets/4976dfa9-0346-4994-92b9-4f9f00b8b43f)

> 💡 **데이터 증강(Data Augmentation)** — `RandomHorizontalFlip`, `RandomCrop` 등으로 훈련 데이터를 인위적으로 다양하게 만들어 과적합을 줄입니다. PyTorch의 `transforms`로 간단히 적용할 수 있습니다.

### 6-2. 모델 설계

```python
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 블록 1: 입력 채널 3 → 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3,32,32) → (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (32,16,16)
            nn.Dropout2d(0.25)
        )

        # 블록 2: 32 → 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (64,8,8)
            nn.Dropout2d(0.25)
        )

        # 블록 3: 64 → 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (128,4,4)
            nn.Dropout2d(0.25)
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.Flatten(),                # 128×4×4 = 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)           # CrossEntropyLoss가 Softmax 포함
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
print(f"전체 파라미터 수: {total:,}")  # 예: 1,276,234
```


### 6-3. 학습 루프

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 학습률 스케줄러 — 10 epoch마다 학습률 × 0.5
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

train_losses = []
val_accuracies = []
best_acc = 0.0

for epoch in range(50):
    # ── 학습 ─────────────────────────────────────
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

    scheduler.step()  # 스케줄러 업데이트

    # 최고 모델 저장
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_cifar10_cnn.pth")

    print(f"Epoch {epoch+1:2d}/50 | Loss: {avg_loss:.4f} | "
          f"Val Acc: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

print(f"\n최고 테스트 정확도: {best_acc:.4f}")
```


### 6-4. 학습 곡선 시각화

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label="훈련 손실", color="blue")
ax1.set_title("손실 변화")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(val_accuracies, label="검증 정확도", color="green")
ax2.set_title("정확도 변화")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```
![Figure_2.png](/api/assets/446d295c-e209-46f6-a5c0-9b6bf7b66eed)

### 6-5. 저장된 모델 불러와서 평가

```python
# 최고 가중치 복원
model.load_state_dict(torch.load("best_cifar10_cnn.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# 최종 정확도
print(f"최종 테스트 정확도: {(all_preds == all_labels).mean():.4f}")

# 예측 결과 시각화
test_images, test_lbls = next(iter(test_loader))
model.eval()
with torch.no_grad():
    preds = model(test_images.to(device)).argmax(dim=1).cpu()

# 역정규화 (시각화를 위해 원래 픽셀값으로 복원)
mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2023, 0.1994, 0.2010])

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
```
"""
최종 테스트 정확도: 0.8786
"""
![Figure_3.png](/api/assets/5aca5ca2-38cd-41f7-beeb-2ea580a2512b)

***

## 7. Keras vs PyTorch 코드 핵심 차이

| 항목 | Keras | PyTorch |
| :-- | :-- | :-- |
| Conv 레이어 | `Conv2D(32, (3,3), padding="same")` | `nn.Conv2d(in_ch, 32, 3, padding=1)` |
| 입력 채널 명시 | ❌ (자동 추론) | ✅ 직접 지정 필수 |
| BatchNorm | `BatchNormalization()` | `nn.BatchNorm2d(채널수)` |
| Dropout | `Dropout(0.25)` | `nn.Dropout2d(0.25)` |
| 학습 | `model.fit(...)` 한 줄 | 직접 루프 작성 |
| 데이터 증강 | `ImageDataGenerator` | `transforms.Compose([...])` |
| 학습률 스케줄러 | `ReduceLROnPlateau` 콜백 | `lr_scheduler.StepLR` |
| 모델 저장 | `model.save("file.keras")` | `torch.save(model.state_dict(), "file.pth")` |
| 모델 불러오기 | `load_model("file.keras")` | `model.load_state_dict(torch.load(...))` |

> ⚠️ PyTorch `Conv2d`에서 `in_channels`는 **이전 레이어의 출력 채널 수** 와 반드시 일치해야 합니다.
> 첫 레이어라면 RGB 이미지 = `3`, 흑백 이미지 = `1`.

***

## 8. 마무리

### 8-1. 오늘 배운 것 한눈에 정리

| 개념 | 핵심 내용 |
| :-- | :-- |
| Conv2D | 필터가 이미지를 훑으며 패턴 추출, 공간 정보 유지 |
| MaxPooling | 영역 최대값 추출, 크기 축소 + 위치 불변성 강화 |
| BatchNormalization | 레이어 출력 정규화 → 학습 안정화, 수렴 빠름 |
| Dropout2d | 채널 단위로 랜덤 비활성화 → 과적합 방지 |
| 데이터 증강 | 뒤집기·자르기로 훈련 데이터 다양화 → 일반화 향상 |
| 학습률 스케줄러 | 학습 진행에 따라 학습률 자동 감소 |
| 모델 저장 | 최고 성능 가중치 저장 후 복원 |

***