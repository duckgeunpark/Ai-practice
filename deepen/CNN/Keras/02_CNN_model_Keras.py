"""
[Step 2/5] CNN 모델 구성
실행: python CNN_model_Keras.py
필요: data/data.npz (Step 1에서 생성)
결과: data/model_untrained.keras 저장
"""

import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Step 1 결과 확인
if not os.path.exists(os.path.join(DATA_DIR, "data.npz")):
    print("❌ data/data.npz 가 없습니다. 먼저 CNN_dataset_Keras.py 를 실행하세요.")
    exit(1)

data = np.load(os.path.join(DATA_DIR, "data.npz"))
print(f"✔ 데이터 로드 완료 — 훈련: {data['X_train'].shape}, 테스트: {data['X_test'].shape}")

model = Sequential(
    [
        # ── 블록 1 ──────────────────────────────────
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
        BatchNormalization(),  # 배치 정규화 — 학습 안정화
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
        Flatten(),  # 4×4×128 = 2048
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation="softmax"),  # 10개 클래스
    ]
)

model.summary()

# 미학습 모델 저장
model.save(os.path.join(DATA_DIR, "model_untrained.keras"))
print("\n✔ 모델 저장 완료 → data/model_untrained.keras")
