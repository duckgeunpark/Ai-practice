"""
[Step 3/5] 모델 컴파일 & 학습
실행: python CNN_compile_Keras.py
필요: data/data.npz, data/model_untrained.keras (Step 1~2)
결과: data/model_trained.keras, data/history.pkl 저장
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Step 1~2 결과 확인
for f, step in [("data.npz", "CNN_dataset_Keras.py"),
                ("model_untrained.keras", "CNN_model_Keras.py")]:
    if not os.path.exists(os.path.join(DATA_DIR, f)):
        print(f"❌ data/{f} 가 없습니다. 먼저 {step} 를 실행하세요.")
        exit(1)

# 데이터 & 모델 로드
data = np.load(os.path.join(DATA_DIR, "data.npz"))
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
print(f"✔ 데이터 로드 완료 — 훈련: {X_train.shape}")

model = load_model(os.path.join(DATA_DIR, "model_untrained.keras"))
print("✔ 모델 로드 완료")

# 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 콜백 설정
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,  # 개선 없으면 학습률을 절반으로
    patience=5,
    min_lr=1e-6,
    verbose=1,
)

# 학습
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# 학습된 모델 저장
model.save(os.path.join(DATA_DIR, "model_trained.keras"))

# 학습 히스토리 저장
with open(os.path.join(DATA_DIR, "history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print("\n✔ 학습 완료!")
print(f"  → 모델: data/model_trained.keras")
print(f"  → 히스토리: data/history.pkl")
