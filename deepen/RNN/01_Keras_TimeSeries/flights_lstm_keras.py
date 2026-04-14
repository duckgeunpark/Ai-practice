"""
[RNN 실습 1] Keras LSTM — 시계열 예측 (항공 승객 수)
실행: python flights_lstm_keras.py

- 데이터: seaborn 내장 flights (1949~1960 월별 승객 수)
- 모델: 단층 LSTM / 다층 LSTM / SimpleRNN / GRU 성능 비교
- 참고: RNN.md 섹션 3~4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

WINDOW_SIZE = 12


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def build_lstm(units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, input_shape=(WINDOW_SIZE, 1), return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_deep_lstm(units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, input_shape=(WINDOW_SIZE, 1), return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_rnn(units=64):
    model = Sequential([
        SimpleRNN(units, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(units=64):
    model = Sequential([
        GRU(units, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    # 1. 데이터 로드 & 시각화
    df = sns.load_dataset("flights")
    print(df.head())
    data = df["passengers"].values.astype("float32")
    print(f"데이터 크기: {data.shape}, 범위: {data.min():.0f}~{data.max():.0f}")

    plt.figure(figsize=(12, 4))
    plt.plot(data, color="steelblue")
    plt.title("월별 항공 승객 수 (1949~1960)")
    plt.xlabel("월 (0=1949년 1월)"); plt.ylabel("승객 수 (천 명)")
    plt.grid(True); plt.tight_layout(); plt.show()

    # 2. 훈련/테스트 분리 + 정규화
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

    # 3. 슬라이딩 윈도우
    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 4. 다층 LSTM 학습
    model_deep = build_deep_lstm(units=64)
    model_deep.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=1,
    )
    history = model_deep.fit(
        X_train, y_train,
        epochs=200, batch_size=16,
        validation_split=0.1, callbacks=[early_stop], verbose=1,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="훈련 손실")
    plt.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
    plt.title("LSTM 학습 손실 변화"); plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 5. 예측 + 역정규화 + 시각화
    y_pred_scaled = model_deep.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.2f} (천 명)  |  RMSE: {rmse:.2f} (천 명)")

    plt.figure(figsize=(14, 5))
    plt.plot(range(len(data)), data, label="실제 승객 수", color="steelblue")
    plt.plot(
        range(train_size + WINDOW_SIZE, train_size + WINDOW_SIZE + len(y_pred)),
        y_pred, label="LSTM 예측", color="red", linestyle="--", linewidth=2,
    )
    plt.axvline(x=train_size, color="gray", linestyle=":", label="훈련/테스트 경계")
    plt.title("항공 승객 수 예측 — Keras LSTM")
    plt.xlabel("월"); plt.ylabel("승객 수 (천 명)")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 6. RNN vs LSTM vs GRU 비교
    results = {}
    for name, model in [
        ("SimpleRNN", build_rnn()),
        ("LSTM", build_deep_lstm()),
        ("GRU", build_gru()),
    ]:
        model.fit(
            X_train, y_train, epochs=100, batch_size=16,
            validation_split=0.1, verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        )
        pred = scaler.inverse_transform(model.predict(X_test)).flatten()
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        results[name] = rmse
        print(f"{name:10s} — RMSE: {rmse:.2f}")

    plt.figure(figsize=(8, 4))
    plt.bar(results.keys(), results.values(),
            color=["skyblue", "coral", "lightgreen"], edgecolor="black")
    plt.title("RNN vs LSTM vs GRU 성능 비교 (RMSE)")
    plt.ylabel("RMSE (낮을수록 좋음)"); plt.grid(axis="y")
    for i, (name, val) in enumerate(results.items()):
        plt.text(i, val + 0.5, f"{val:.2f}", ha="center", fontsize=11)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
