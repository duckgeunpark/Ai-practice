"""
[RNN 실습 3] Keras Bidirectional LSTM — 텍스트 분류 (IMDB 감성 분석)
실행: python imdb_bilstm_keras.py

- 데이터: Keras 내장 IMDB (영화 리뷰 25,000건 x 2)
- 모델: Embedding + Bidirectional LSTM 2층 → 이진 분류(긍/부정)
- 참고: RNN.md 섹션 6
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

VOCAB_SIZE = 10000
MAX_LEN = 200


def build_text_lstm():
    """단방향 LSTM (비교용)"""
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_bidirectional_lstm():
    """양방향 LSTM"""
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    # 1. 데이터 로드
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    print(f"훈련: {len(X_train)}건 / 테스트: {len(X_test)}건")

    lengths = [len(x) for x in X_train]
    print(f"리뷰 길이 — min:{min(lengths)}, max:{max(lengths)}, avg:{np.mean(lengths):.0f}")

    # 원문 샘플 확인
    word_index = imdb.get_word_index()
    idx_to_word = {v + 3: k for k, v in word_index.items()}
    idx_to_word.update({0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"})
    sample = " ".join(idx_to_word.get(i, "?") for i in X_train[0][:30])
    print(f"샘플: {sample}...")
    print(f"정답: {'긍정' if y_train[0] == 1 else '부정'}")

    # 2. 패딩
    X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
    X_test_pad = pad_sequences(X_test, maxlen=MAX_LEN, padding="pre", truncating="pre")
    print(f"X_train_pad: {X_train_pad.shape}")

    # 3. 모델 생성 & 학습
    model = build_bidirectional_lstm()
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=3,
        restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        X_train_pad, y_train,
        epochs=10, batch_size=128,
        validation_split=0.2, callbacks=[early_stop], verbose=1,
    )

    # 4. 평가
    test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"테스트 정확도: {test_acc:.4f}")

    # 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history.history["accuracy"], label="훈련 정확도")
    ax1.plot(history.history["val_accuracy"], label="검증 정확도", linestyle="--")
    ax1.set_title("정확도 변화"); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history["loss"], label="훈련 손실")
    ax2.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
    ax2.set_title("손실 변화"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()

    # 5. 개별 예측 테스트
    def predict_sentiment(review_idx_list):
        padded = pad_sequences([review_idx_list], maxlen=MAX_LEN,
                               padding="pre", truncating="pre")
        prob = model.predict(padded, verbose=0)[0][0]
        label = "긍정 😊" if prob >= 0.5 else "부정 😞"
        print(f"예측: {label} (확률: {prob:.4f})")

    predict_sentiment(X_test[0])
    print(f"실제 정답: {'긍정' if y_test[0] == 1 else '부정'}")


if __name__ == "__main__":
    main()
