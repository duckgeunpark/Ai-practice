[미검증]
## 0. 시리즈

| 심화편 | 제목 | 난이도 | 역할 |
| :-- | :-- | :-- | :-- |
| [심화 1편](https://duckport.pages.dev/posts/CNN_Deep) | CNN — 이미지 분류 완벽 정리 | ⭐⭐ | 이미지 처리 |
| [**심화 2편**](https://duckport.pages.dev/posts/RNN)⬅️ | RNN / LSTM — 시계열·텍스트 처리 | ⭐⭐⭐ | 순서 데이터 |
| [심화 3편](https://duckport.pages.dev/posts/Transformer) | Transformer 구조 완벽 정리 | ⭐⭐⭐⭐ | 현대 AI 핵심 |
| [심화 4편](https://duckport.pages.dev/posts/BERT) | BERT / GPT 원리와 활용 | ⭐⭐⭐⭐ | 언어 모델 |
| [심화 5편](https://duckport.pages.dev/posts/Finetuning_Deep) | HuggingFace로 LLM Fine-tuning | ⭐⭐⭐⭐⭐ | LLM 실전 |


***

## 1. RNN이란?

### 1-1. CNN(1편)과 무엇이 다른가?

| 비교 항목 | CNN (심화 1편) | RNN (심화 2편) |
| :-- | :-- | :-- |
| 처리 대상 | 공간 데이터 (이미지) | 시간/순서 데이터 (텍스트, 주가) |
| 입력 형태 | `(batch, H, W, C)` | `(batch, timesteps, features)` |
| 핵심 아이디어 | 필터로 공간 패턴 추출 | 이전 상태를 기억하며 순서 처리 |
| 주요 레이어 | `Conv2D`, `MaxPooling2D` | `LSTM`, `GRU` |

### 1-2. 순서가 있는 데이터란?

Dense나 CNN은 입력 순서를 고려하지 않습니다. 하지만 현실에는 **순서 자체가 의미** 인 데이터가 많습니다.

```
텍스트  : "나는 밥을 먹었다" → 단어 순서가 바뀌면 의미가 달라짐
주가    : 어제 → 오늘 → 내일 의 흐름이 중요
날씨    : 기온 변화 패턴이 중요
음성    : 소리의 시간적 흐름이 중요
```


### 1-3. RNN의 동작 원리 — 은닉 상태(Hidden State)

RNN은 **이전 시점의 출력(은닉 상태)을 다음 시점의 입력으로 함께 사용** 합니다.

```
일반 신경망:   입력 → [레이어] → 출력
                         (이전을 기억 못 함)

RNN:    x₁ → [레이어] → h₁ ─┐
                               ↓
        x₂ → [레이어] → h₂ ─┐  (h₁을 기억해서 같이 처리)
                               ↓
        x₃ → [레이어] → h₃ → 출력
```

```python
# RNN 수식 (한 시점)
h_t = tanh(W_x * x_t + W_h * h_(t-1) + b)
# x_t      : 현재 시점의 입력
# h_(t-1)  : 이전 시점의 은닉 상태 (기억)
# h_t      : 현재 시점의 은닉 상태 (출력)
```


### 1-4. RNN의 치명적 한계 — 기울기 소실(Vanishing Gradient)

```
"오늘 날씨가 맑아서 ... (수십 단어) ... 우산이 필요 없다"
                                              ↑
                              "맑다"는 정보가 여기까지 전달이 안 됨!
```

역전파 시 기울기가 시간을 거슬러 올라갈수록 **점점 작아져 0에 수렴** 합니다. 결국 멀리 떨어진 시점의 정보를 학습하지 못합니다. 이 문제를 해결한 것이 **LSTM** 입니다.

***

## 2. LSTM — RNN의 한계를 넘다

### 2-1. LSTM이란?

**LSTM(Long Short-Term Memory)** 은 1997년에 제안된 RNN의 개선 구조입니다. "무엇을 기억하고, 무엇을 잊을지"를 **게이트(Gate)** 로 직접 제어합니다.

```
RNN  : 은닉 상태(h) 1개 → 단기 기억만 가능
LSTM : 은닉 상태(h) + 셀 상태(C) → 장기 + 단기 기억 모두 가능
```


### 2-2. 셀 상태(Cell State) — 장기 기억 통로

```
 C_(t-1) ──────────────────────────────────→ C_t
              ↑ forget    ↑ input
              (버릴 것)   (추가할 것)
```

셀 상태는 컨베이어 벨트처럼 **정보를 시간 축으로 그대로 흘려보내는 통로** 입니다. 게이트들이 정보를 추가하거나 삭제하면서 조절합니다.

### 2-3. 게이트 3종 세트

```python
# ① Forget Gate — 이전 기억 중 버릴 것 결정 (0: 완전히 삭제, 1: 완전히 유지)
f_t = sigmoid(W_f · [h_(t-1), x_t] + b_f)

# ② Input Gate — 새로운 정보 중 저장할 것 결정
i_t = sigmoid(W_i · [h_(t-1), x_t] + b_i)  # 얼마나 저장할지
g_t = tanh(W_g · [h_(t-1), x_t] + b_g)     # 어떤 값을 저장할지

# 셀 상태 업데이트
C_t = f_t * C_(t-1) + i_t * g_t
#      ↑ 버릴 것 지우고   ↑ 새 정보 추가

# ③ Output Gate — 최종적으로 출력할 정보 결정
o_t = sigmoid(W_o · [h_(t-1), x_t] + b_o)
h_t = o_t * tanh(C_t)
```


### 2-4. GRU — LSTM의 경량화 버전

**GRU(Gated Recurrent Unit)** 는 LSTM에서 셀 상태를 없애고 게이트를 2개로 줄인 버전입니다.


| 비교 항목 | LSTM | GRU |
| :-- | :-- | :-- |
| 게이트 수 | 3개 (Forget, Input, Output) | 2개 (Reset, Update) |
| 상태 | 셀 상태 + 은닉 상태 | 은닉 상태만 |
| 파라미터 수 | 많음 | 적음 (약 25% 절감) |
| 학습 속도 | 느림 | 빠름 |
| 성능 | 대부분 우세 | 데이터 작을 때 유리 |

```python
# Keras — 레이어 이름만 바꾸면 됨
from tensorflow.keras.layers import LSTM, GRU

LSTM(64, return_sequences=True)  # LSTM
GRU(64, return_sequences=True)   # GRU (나머지 코드 동일)

# PyTorch — 동일하게 클래스만 교체
nn.LSTM(input_size, hidden_size, ...)  # LSTM
nn.GRU(input_size, hidden_size, ...)   # GRU
```


***

## 3. 시퀀스 데이터 전처리

### 3-1. 슬라이딩 윈도우 — 시계열 데이터를 X/y로 만드는 방법

```
원본 데이터: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
window_size = 3 으로 슬라이딩 윈도우 적용:

X (입력)          y (정답)
[10, 20, 30]  →   40
[20, 30, 40]  →   50
[30, 40, 50]  →   60
[40, 50, 60]  →   70
...
```

```python
import numpy as np

def create_sequences(data, window_size):
    """
    data        : 1D numpy array (정규화된 시계열 데이터)
    window_size : 몇 개의 과거 시점으로 다음을 예측할지
    반환        : X shape (samples, window_size, 1)
                  y shape (samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])  # 과거 window_size 개
        y.append(data[i + window_size])       # 다음 1개
    X = np.array(X)
    y = np.array(y)
    # LSTM 입력은 3D: (samples, timesteps, features)
    X = X.reshape(X.shape0], X.shape[1], 1)
    return X, y
```


### 3-2. 스케일링 (MinMaxScaler)

LSTM은 `tanh` 활성화 함수를 사용하므로 입력값을 **0~1 범위로 정규화** 해야 학습이 안정적입니다.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# 반드시 훈련 데이터로만 fit!
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_scaled  = scaler.transform(test_data.reshape(-1, 1))

# 예측 후 원래 스케일로 복원
y_pred_original = scaler.inverse_transform(y_pred)
```


### 3-3. 3차원 입력 형태 이해

RNN/LSTM의 입력은 반드시 **3차원** 이어야 합니다.

```
(samples, timesteps, features)
    ↑          ↑          ↑
 데이터 수  시간 길이   변수 개수

예시:
- (1000, 30, 1)  : 1000개 샘플, 과거 30일, 1개 변수(주가)
- (500, 10, 5)   : 500개 샘플, 과거 10일, 5개 변수(기온/습도/강수량/기압/풍속)

# Dense 입력 (2D) vs LSTM 입력 (3D)
Dense : (1000, 30)       # 2D
LSTM  : (1000, 30, 1)    # 3D — 마지막 차원(features)을 반드시 붙여야 함
```


***

## 4. Keras로 LSTM 구현하기 — 시계열 예측

### 4-1. 데이터 준비 (항공 승객 수 예측)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# seaborn 내장 데이터셋 사용 (인터넷 없이 실행 가능)
import seaborn as sns
df = sns.load_dataset("flights")  # 1949~1960년 월별 항공 승객 수

print(df.head(10))
print(f"데이터 크기: {df.shape}")  # (144, 3)
print(f"컬럼: {df.columns.tolist()}")  # ['year', 'month', 'passengers']

# 승객 수만 추출
data = df["passengers"].values.astype("float32")
print(f"최소: {data.min()}, 최대: {data.max()}")  # 104 ~ 622

# 시각화
plt.figure(figsize=(12, 4))
plt.plot(data, color="steelblue")
plt.title("월별 항공 승객 수 (1949~1960)")
plt.xlabel("월 (0=1949년 1월)")
plt.ylabel("승객 수 (천 명)")
plt.grid(True)
plt.show()
```
"""
   year month  passengers
0  1949   Jan         112
1  1949   Feb         118
2  1949   Mar         132
3  1949   Apr         129
4  1949   May         121
데이터 크기: (144,), 범위: 104~622
X_train: (103, 12, 1), X_test: (17, 12, 1)
"""

![Figure_1.png](/api/assets/e5a7c443-55da-4a23-a8aa-e7bafa72538f)


```python
# 훈련 / 테스트 분리 (80% / 20%)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]   # 0~115
test_data  = data[train_size:]   # 116~143

print(f"훈련 데이터: {len(train_data)}개")  # 115
print(f"테스트 데이터: {len(test_data)}개")  # 29

# 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_scaled  = scaler.transform(test_data.reshape(-1, 1)).flatten()
```

```python
# 슬라이딩 윈도우 적용
WINDOW_SIZE = 12  # 과거 12개월로 다음 달 예측

X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
X_test,  y_test  = create_sequences(test_scaled,  WINDOW_SIZE)

print(f"X_train shape: {X_train.shape}")  # (103, 12, 1)
print(f"y_train shape: {y_train.shape}")  # (103,)
print(f"X_test shape:  {X_test.shape}")   # (17, 12, 1)
```


### 4-2. 단층 LSTM 모델

```python
def build_lstm(units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, input_shape=(WINDOW_SIZE, 1), return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)  # 회귀 → 활성화 함수 없음
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

model_lstm = build_lstm(units=64)
model_lstm.summary()
```

"""
Model: "sequential"
Layer (type)         Output Shape       Param #
────────────────────────────────────────────────
lstm (LSTM)          (None, 12, 64)    16,896
dropout (Dropout)    (None, 12, 64)         0
lstm_1 (LSTM)        (None, 32)        12,416
dropout_1 (Dropout)  (None, 32)             0
dense (Dense)        (None, 32)         1,056
dense_1 (Dense)      (None, 1)             33
────────────────────────────────────────────────
Total params: 30,401 (118.75 KB)
Trainable params: 30,401 (118.75 KB)
Non-trainable params: 0 (0.00 B)
"""

### 4-3. 다층 LSTM 모델

```python
def build_deep_lstm(units=64, dropout=0.2):
    model = Sequential([
        # return_sequences=True → 다음 LSTM 레이어에 시퀀스 전달
        LSTM(units, input_shape=(WINDOW_SIZE, 1), return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

model_deep = build_deep_lstm(units=64)
```

> 💡 **return_sequences** — LSTM 레이어를 여러 개 쌓을 때 중간 레이어는 반드시 `return_sequences=True`로 설정해야 합니다. 마지막 LSTM만 `False`(기본값)로 설정합니다.

```
return_sequences=True  → 모든 timestep의 출력 반환 (다음 LSTM에 전달)
                          출력 shape: (batch, timesteps, units)

return_sequences=False → 마지막 timestep의 출력만 반환
                          출력 shape: (batch, units)
```


### 4-4. 학습

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model_deep.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 학습 곡선
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"],     label="훈련 손실")
plt.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
plt.title("LSTM 학습 손실 변화")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()
```
"""
Epoch 1/200
6/6 ━━━━━━━━━━━━━━━━━━━━ 2s 57ms/step - loss: 0.1337 - mae: 0.3060 - val_loss: 0.2358 - val_mae: 0.4660
Epoch 2/200
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0461 - mae: 0.1602 - val_loss: 0.0279 - val_mae: 0.1135
....
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.0046 - mae: 0.0528 - val_loss: 0.0151 - val_mae: 0.1029
Epoch 143/200
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.0052 - mae: 0.0551 - val_loss: 0.0101 - val_mae: 0.0889
Epoch 143: early stopping
"""
![Figure_2.png](/api/assets/5718b916-ca01-4d3a-9a5a-c0ad73985857)

### 4-5. 예측 및 시각화

```python
# 예측
y_pred_scaled = model_deep.predict(X_test)

# 역정규화 (원래 승객 수로 복원)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 성능 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"MAE:  {mae:.2f} (천 명)")
print(f"RMSE: {rmse:.2f} (천 명)")

# 전체 데이터 위에 예측 결과 시각화
plt.figure(figsize=(14, 5))
plt.plot(range(len(data)), data, label="실제 승객 수", color="steelblue")
plt.plot(
    range(train_size + WINDOW_SIZE, train_size + WINDOW_SIZE + len(y_pred)),
    y_pred,
    label="LSTM 예측", color="red", linestyle="--", linewidth=2
)
plt.axvline(x=train_size, color="gray", linestyle=":", label="훈련/테스트 경계")
plt.title("항공 승객 수 예측 — Keras LSTM")
plt.xlabel("월")
plt.ylabel("승객 수 (천 명)")
plt.legend()
plt.grid(True)
plt.show()
```
"""
MAE: 65.97 (천 명)  |  RMSE: 78.06 (천 명)
"""
![Figure_3.png](/api/assets/d36b4128-cf92-4ead-ab63-1122928bb4bc)


### 4-6. RNN vs LSTM vs GRU 성능 비교

```python
def build_rnn(units=64):
    model = Sequential([
        SimpleRNN(units, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(units=64):
    model = Sequential([
        GRU(units, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

results = {}

for name, model in [("SimpleRNN", build_rnn()),
                    ("LSTM",      build_deep_lstm()),
                    ("GRU",       build_gru())]:
    model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
    pred = scaler.inverse_transform(model.predict(X_test)).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    results[name] = rmse
    print(f"{name:10s} — RMSE: {rmse:.2f}")

# 시각화
plt.figure(figsize=(8, 4))
plt.bar(results.keys(), results.values(),
        color=["skyblue", "coral", "lightgreen"], edgecolor="black")
plt.title("RNN vs LSTM vs GRU 성능 비교 (RMSE)")
plt.ylabel("RMSE (낮을수록 좋음)")
plt.grid(axis="y")
for i, (name, val) in enumerate(results.items()):
    plt.text(i, val + 0.5, f"{val:.2f}", ha="center", fontsize=11)
plt.show()
```
![Figure_4.png](/api/assets/e6526f37-dd37-4b6b-ba7e-d2aff767653e)
>  "LSTM > GRU > RNN이어야 하는데 왜 뒤집혔지?"라고 의심하지 않아도 됩니다. 샘플이 103개면 큰 모델이 지는 게 당연합니다. 실습 의도는 "GRU가 작은 데이터에 유리하다"를 체감하는 겁니다.

***

## 5. PyTorch로 LSTM 구현하기 — 시계열 예측

flights 데이터는 **우상향 추세 + 연단위 계절성**이 강하고, 샘플 수가 적습니다(144개). 그대로 LSTM에 넣으면 훈련 구간 값 범위를 벗어나는 테스트 구간에서 외삽(extrapolation)이 무너지기 쉽고, 과적합도 쉽게 발생합니다. 아래 구성은 네 가지 원칙으로 정리합니다.

1. **시드 고정** — 재현성 확보
2. **차분(Differencing) 학습** — 수준(level) 대신 "전월 대비 변화량"을 학습해 추세 제거
3. **훈련셋 내부에 검증셋 분리** — EarlyStopping이 테스트셋을 들여다보지 않게
4. **소형 모델** — hidden 32, num_layers 1 (샘플 ~100개 규모에 맞춤)
5. **StandardScaler** — 차분값은 음수 포함이라 MinMax보다 평균 0 / 표준편차 1 스케일링이 적합

### 5-1. 차분 전처리와 train/val/test 분리

```python
import random
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader

SEED = 42
WINDOW_SIZE = 12

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 원본 데이터
df = sns.load_dataset("flights")
data = df["passengers"].values.astype("float32")          # (144,)
train_size = int(len(data) * 0.8)                         # 115

# 차분 — diff[i] = data[i+1] - data[i]
diff = np.diff(data)                                      # (143,)

# train/test 차분 분리 (test 경계 포함)
train_diff_full = diff[: train_size - 1]                  # 114
test_diff       = diff[train_size - 1 :]                  # 29

# 훈련셋 내부에서 뒤 20%를 val로 분리 — EarlyStopping 기준
val_size = int(len(train_diff_full) * 0.2)
tr_diff  = train_diff_full[:-val_size]                    # 92
va_diff  = train_diff_full[-val_size:]                    # 22

# StandardScaler (fit은 train에만)
scaler = StandardScaler()
tr_scaled = scaler.fit_transform(tr_diff.reshape(-1, 1)).flatten()
va_scaled = scaler.transform(va_diff.reshape(-1, 1)).flatten()
te_scaled = scaler.transform(test_diff.reshape(-1, 1)).flatten()
```

> 💡 **왜 차분인가?** — 원시 데이터는 100대에서 600대로 6배 넘게 커집니다. 모델이 본 적 없는 600대 값을 테스트에서 맞춰야 하는 외삽 문제가 됩니다. 차분을 취하면 학습 목표가 "증가량"으로 바뀌고 값 범위가 평균 0 근처로 안정되므로, 테스트 구간에서도 동일 분포 안에서 예측하게 됩니다.

> 💡 **왜 훈련셋 내부 val인가?** — 테스트셋을 검증으로 함께 쓰면 EarlyStopping이 테스트를 보고 멈춘 시점을 고르게 됩니다(데이터 누수). 훈련 내부에서 val을 떼어내야 테스트 성능을 정직하게 측정할 수 있습니다.

### 5-2. Dataset / DataLoader

```python
class SeqDataset(Dataset):
    """1D 시퀀스에 슬라이딩 윈도우 적용"""
    def __init__(self, series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i : i + window_size])
            y.append(series[i + window_size])
        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = SeqDataset(tr_scaled, WINDOW_SIZE)
val_ds   = SeqDataset(va_scaled, WINDOW_SIZE)
test_ds  = SeqDataset(te_scaled, WINDOW_SIZE)

g = torch.Generator(); g.manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                          num_workers=0, generator=g)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)
```

### 5-3. 소형 LSTM 모델

```python
class LSTMModel(nn.Module):
    """단층/소형 LSTM — 파라미터 약 4.8K"""
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq, hidden)
        out = out[:, -1, :]         # 마지막 timestep만 사용
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)
```

> 💡 **왜 소형 모델인가?** — 훈련 윈도우 샘플이 ~80개인데 다층 LSTM(수만 파라미터)을 쓰면 바로 과적합됩니다. 파라미터/샘플 비가 작아야 일반화가 좋습니다.

> 💡 **`batch_first=True`** — PyTorch LSTM 기본 입력은 `(seq, batch, feature)`. `batch_first=True`로 두면 `(batch, seq, feature)`가 되어 직관적입니다.

### 5-4. 학습 루프 (train/val 분리)

```python
def train_model(model, train_loader, val_loader, device,
                epochs=300, lr=1e-3, patience=30):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train": [], "val": []}
    best_val, best_weights, patience_cnt = float("inf"), None, 0

    for epoch in range(epochs):
        # train
        model.train()
        tr_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                va_loss += criterion(model(X), y).item()
        va_loss /= len(val_loader)

        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    return model, history


model = LSTMModel(hidden_size=32, num_layers=1, dropout=0.2)
model, history = train_model(model, train_loader, val_loader, device)
```

### 5-5. 예측 및 원 단위 복원

차분으로 학습했으므로, 예측값은 "다음 달 증가량"입니다. 원래 승객 수로 돌리려면 **직전 실측치에 더해주는** 과정이 필요합니다.

```python
# 테스트 예측 (차분 스케일)
model.eval()
pred_scaled, true_scaled = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred_scaled.extend(model(X).cpu().numpy())
        true_scaled.extend(y.numpy())

# 차분 역스케일
pred_diff = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()

# 원 단위 복원 — 직전 실측치 + 예측 증가량
#  test_ds의 i번째 샘플은 te_scaled[i+W]를 예측하고,
#  이는 data의 index (train_size - 1 + i + W)에 해당하는 "증가량"
offset = train_size - 1 + WINDOW_SIZE
base_vals = data[offset : offset + len(pred_diff)]        # 각 예측의 직전 실측치
y_pred    = base_vals + pred_diff
y_true    = data[offset + 1 : offset + 1 + len(pred_diff)]  # 실제 다음 달 값

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"PyTorch LSTM (diff) — RMSE: {rmse:.2f}, MAE: {mae:.2f}")
```

> ⚠️ **복원 시 자주 하는 실수** — 예측 증가량을 "직전 예측값"에 더해서 누적하면 오차가 계속 쌓입니다. 1-step-ahead 평가에서는 **직전 실측치**에 더해야 정직한 측정이 됩니다. 누적 예측(long-horizon)이 필요하면 별도 루프에서 예측값을 피드백하는 방식으로 분리하세요.


***

## 6. Keras로 LSTM 구현하기 — 텍스트 분류 (감성 분석)

### 6-1. IMDB 영화 리뷰 데이터셋

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# 상위 10,000개 단어만 사용
VOCAB_SIZE  = 10000
MAX_LEN     = 200   # 최대 200 단어로 패딩

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print(f"훈련 샘플 수: {len(X_train)}")   # 25,000
print(f"테스트 샘플 수: {len(X_test)}")  # 25,000
print(f"클래스: 0=부정, 1=긍정")

# 샘플 길이 분포 확인
lengths = [len(x) for x in X_train]
print(f"리뷰 길이 — 최소: {min(lengths)}, 최대: {max(lengths)}, "
      f"평균: {np.mean(lengths):.0f}")

# 단어 인덱스로 원문 확인 (선택 사항)
word_index = imdb.get_word_index()
idx_to_word = {v+3: k for k, v in word_index.items()}
idx_to_word.update({0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"})
sample_review = " ".join(idx_to_word.get(i, "?") for i in X_train[0][:30])
print(f"리뷰 샘플: {sample_review}...")
print(f"정답: {'긍정' if y_train[0] == 1 else '부정'}")
```


### 6-2. 패딩(Padding) — 길이 통일

```python
# 모든 리뷰를 MAX_LEN으로 길이 통일
# 짧은 리뷰 → 앞에 0을 채움 (pre-padding)
# 긴 리뷰   → 앞을 잘라냄 (pre-truncating)
X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
X_test_pad  = pad_sequences(X_test,  maxlen=MAX_LEN, padding="pre", truncating="pre")

print(f"X_train_pad shape: {X_train_pad.shape}")  # (25000, 200)
print(f"패딩 전 리뷰 길이: {len(X_train[0])}")
print(f"패딩 후 리뷰 길이: {len(X_train_pad[0])}")  # 200
```

```
패딩 예시:
원본  : [14, 22, 16, 43, 530]
패딩  : [0, 0, 0, ..., 14, 22, 16, 43, 530]
         ↑ 앞에 0으로 채움          ↑ 원본
```


### 6-3. Embedding Layer란?

```
단어 인덱스 → 의미 있는 실수 벡터로 변환

"고양이" → 14 → [0.23, -0.17, 0.88, ..., 0.41]  (128차원 벡터)
"강아지" → 38 → [0.19, -0.14, 0.91, ..., 0.39]  (128차원 벡터)
                  ↑ 의미가 비슷한 단어 → 벡터가 유사함

Embedding(vocab_size, embed_dim)
= vocab_size개의 단어를 각각 embed_dim 차원 벡터로 표현
= 학습을 통해 최적의 단어 표현을 찾아냄
```


### 6-4. 모델 설계

```python
# 단방향 LSTM
def build_text_lstm():
    model = Sequential([
        # (batch, 200) → (batch, 200, 128) : 각 단어를 128차원 벡터로
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")  # 이진 분류 → sigmoid
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# 양방향 LSTM (Bidirectional) — 앞→뒤 + 뒤→앞 동시 처리
def build_bidirectional_lstm():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
        # Bidirectional: hidden_size가 2배 (앞방향 + 뒷방향)
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

model_text = build_bidirectional_lstm()
model_text.summary()
```

```
Bidirectional(LSTM(64)) → 출력 shape: (batch, 200, 128)
                                               ↑
                         앞방향 64 + 뒷방향 64 = 128
```


### 6-5. 학습 및 평가

```python
early_stop = EarlyStopping(monitor="val_accuracy", patience=3,
                           restore_best_weights=True, verbose=1)

history = model_text.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 평가
test_loss, test_acc = model_text.evaluate(X_test_pad, y_test, verbose=0)
print(f"테스트 정확도: {test_acc:.4f}")  # 예: 0.8720

# 학습 곡선
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(history.history["accuracy"],     label="훈련 정확도")
ax1.plot(history.history["val_accuracy"], label="검증 정확도", linestyle="--")
ax1.set_title("정확도 변화")
ax1.legend(); ax1.grid(True)

ax2.plot(history.history["loss"],     label="훈련 손실")
ax2.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
ax2.set_title("손실 변화")
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.show()

# 직접 예측해보기
def predict_sentiment(model, review_idx_list):
    padded = pad_sequences([review_idx_list], maxlen=MAX_LEN,
                           padding="pre", truncating="pre")
    prob = model.predict(padded, verbose=0)[0][0]
    label = "긍정 😊" if prob >= 0.5 else "부정 😞"
    print(f"예측: {label} (확률: {prob:.4f})")

# 테스트 샘플로 확인
predict_sentiment(model_text, X_test[0])
print(f"실제 정답: {'긍정' if y_test[0] == 1 else '부정'}")
```


***

## 7. PyTorch로 LSTM 구현하기 — 텍스트 분류

### 7-1. 데이터 준비

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Keras에서 불러온 X_train_pad, X_test_pad 그대로 활용
X_tr = torch.tensor(X_train_pad, dtype=torch.long)
y_tr = torch.tensor(y_train,     dtype=torch.float32)
X_te = torch.tensor(X_test_pad,  dtype=torch.long)
y_te = torch.tensor(y_test,      dtype=torch.float32)

train_ds = TensorDataset(X_tr, y_tr)
test_ds  = TensorDataset(X_te, y_te)

train_loader_text = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader_text  = DataLoader(test_ds,  batch_size=128, shuffle=False)

print(f"X_tr shape: {X_tr.shape}")  # (25000, 200)
print(f"y_tr shape: {y_tr.shape}")  # (25000,)
```


### 7-2. 모델 클래스 설계

```python
class TextLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128,
                 hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = 0          # 패딩 토큰(0)은 학습에서 제외
        )
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True        # 양방향 LSTM
        )
        # bidirectional=True → hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)            # (batch, seq_len, embed_dim)

        out, (hn, cn) = self.lstm(embedded)
        # out: (batch, seq_len, hidden*2)

        # 마지막 timestep
        out = out[:, -1, :]                     # (batch, hidden*2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))          # (batch, 32)
        out = self.sigmoid(self.fc2(out))       # (batch, 1)
        return out.squeeze(-1)                  # (batch,)


model_text_pt = TextLSTM(vocab_size=VOCAB_SIZE).to(device)
total = sum(p.numel() for p in model_text_pt.parameters())
print(f"파라미터 수: {total:,}")
```


### 7-3. 학습 루프

```python
criterion_text = nn.BCELoss()   # 이진 분류 → Binary Cross Entropy
optimizer_text = torch.optim.Adam(model_text_pt.parameters(), lr=0.001)

best_acc_text = 0.0
best_weights_text = None

for epoch in range(10):
    # ── 학습 ─────────────────────────────────
    model_text_pt.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader_text:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_text.zero_grad()
        output = model_text_pt(X_batch)
        loss   = criterion_text(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_text_pt.parameters(), 1.0)
        optimizer_text.step()
        running_loss += loss.item()

    # ── 검증 ─────────────────────────────────
    model_text_pt.eval()
    correct = 0
    total_cnt = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader_text:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = (model_text_pt(X_batch) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total_cnt += y_batch.size(0)

    acc = correct / total_cnt
    avg_loss = running_loss / len(train_loader_text)

    if acc > best_acc_text:
        best_acc_text = acc
        best_weights_text = {k: v.clone() for k, v in
                             model_text_pt.state_dict().items()}

    print(f"Epoch {epoch+1:2d}/10 | Loss: {avg_loss:.4f} | "
          f"Test Acc: {acc:.4f}")

model_text_pt.load_state_dict(best_weights_text)
print(f"\n최고 테스트 정확도: {best_acc_text:.4f}")
```


***

## 8. Keras vs PyTorch 핵심 코드 비교

| 항목 | Keras | PyTorch |
| :-- | :-- | :-- |
| LSTM 레이어 | `LSTM(64, return_sequences=True)` | `nn.LSTM(input, 64, batch_first=True)` |
| 입력 채널 명시 | ❌ 자동 추론 | ✅ `input_size` 직접 지정 |
| 다층 LSTM | 레이어 여러 개 쌓기 | `num_layers=2` |
| 양방향 | `Bidirectional(LSTM(...))` | `bidirectional=True` |
| Embedding | `Embedding(vocab, dim)` | `nn.Embedding(vocab, dim, padding_idx=0)` |
| 은닉 상태 초기화 | 자동 | `h0, c0` 직접 생성 |
| 기울기 클리핑 | `clipnorm=1.0` (compile 옵션) | `clip_grad_norm_(model.parameters(), 1.0)` |
| 학습 루프 | `model.fit(...)` | 직접 작성 |
| 이진 분류 손실 | `binary_crossentropy` | `nn.BCELoss()` |

> ⚠️ PyTorch `nn.LSTM`의 출력 `out`은 `(batch, seq_len, hidden_size * directions)` 형태입니다. **양방향** (`bidirectional=True`)이면 `hidden_size * 2`가 됩니다. 다음 Linear 레이어의 `in_features`를 반드시 맞춰야 합니다.

***

## 9. 마무리

### 9-1. 오늘 배운 것 한눈에 정리

| 개념 | 핵심 내용 |
| :-- | :-- |
| RNN | 이전 은닉 상태를 현재 입력과 함께 처리, 기울기 소실 문제 |
| LSTM | Forget·Input·Output 게이트로 장기 기억 해결 |
| GRU | LSTM 경량화, 게이트 2개, 파라미터 적음 |
| 슬라이딩 윈도우 | 시계열 → `(samples, timesteps, features)` 3D 변환 |
| return_sequences | 다음 LSTM에 전달 시 `True`, 마지막 레이어만 `False` |
| Embedding | 정수 토큰 → 의미 있는 실수 벡터 변환 |
| Bidirectional | 앞뒤 방향 동시 처리, hidden_size 2배 |
| clip_grad_norm | 기울기 폭발 방지, RNN 계열 필수 |

***