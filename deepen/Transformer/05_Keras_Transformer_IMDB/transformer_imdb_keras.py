"""
[Transformer 실습 5] Keras Transformer — IMDB 감성 분석 (텍스트 분류)
실행: python transformer_imdb_keras.py

- TransformerBlock 직접 작성 + 학습 가능한 위치 임베딩
- GlobalAveragePooling → 이진 분류
- 참고: Transformer.md 섹션 9
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

VOCAB_SIZE = 20000
MAX_LEN = 200
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 2
DROPOUT = 0.1


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.att(x, x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


def build_transformer_classifier():
    inputs = layers.Input(shape=(MAX_LEN,))

    # 토큰 임베딩 + 학습 가능한 위치 임베딩
    x = layers.Embedding(VOCAB_SIZE, D_MODEL)(inputs)
    positions = tf.range(start=0, limit=MAX_LEN, delta=1)
    pos_emb = layers.Embedding(MAX_LEN, D_MODEL)(positions)
    x = x + pos_emb
    x = layers.Dropout(DROPOUT)(x)

    # Transformer 블록 N개
    for _ in range(NUM_LAYERS):
        x = TransformerBlock(D_MODEL, NUM_HEADS, D_FF, DROPOUT)(x)

    # 분류 헤드
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)


def main():
    # 1. 데이터
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
    X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding="pre", truncating="pre")
    print(f"훈련: {X_train.shape} / 테스트: {X_test.shape}")

    # 2. 모델
    model = build_transformer_classifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. 학습
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3,
        restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # 4. 평가
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTransformer 테스트 정확도: {test_acc:.4f}")
    print("참고 — Bi-LSTM(실습 3): ~87%")

    # 5. 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history.history["accuracy"], label="훈련 정확도")
    ax1.plot(history.history["val_accuracy"], label="검증 정확도", linestyle="--")
    ax1.set_title("Transformer 정확도")
    ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)

    ax2.plot(history.history["loss"], label="훈련 손실")
    ax2.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
    ax2.set_title("Transformer 손실")
    ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
