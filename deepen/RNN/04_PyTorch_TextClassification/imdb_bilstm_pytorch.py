"""
[RNN 실습 4] PyTorch Bidirectional LSTM — 텍스트 분류 (IMDB 감성 분석)
실행: python imdb_bilstm_pytorch.py

- 데이터: Keras 내장 IMDB (tensorflow.keras.datasets 필요)
- 모델: Embedding + 양방향 LSTM 2층 → 이진 분류
- 참고: RNN.md 섹션 7
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

VOCAB_SIZE = 10000
MAX_LEN = 200


class TextLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128,
                 hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)              # (batch, seq, embed)
        out, _ = self.lstm(embedded)              # (batch, seq, hidden*2)
        out = out[:, -1, :]                       # 마지막 timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    # 1. 데이터 로드 + 패딩
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
    X_test_pad = pad_sequences(X_test, maxlen=MAX_LEN, padding="pre", truncating="pre")
    print(f"훈련: {X_train_pad.shape} / 테스트: {X_test_pad.shape}")

    # 2. 텐서 변환
    X_tr = torch.tensor(X_train_pad, dtype=torch.long)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test_pad, dtype=torch.long)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # 3. 모델
    model = TextLSTM(vocab_size=VOCAB_SIZE).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {total:,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. 학습 루프
    train_losses, test_accs = [], []
    best_acc = 0.0
    best_weights = None

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct, total_cnt = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = (model(X_batch) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total_cnt += y_batch.size(0)
        acc = correct / total_cnt

        train_losses.append(avg_loss)
        test_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

    model.load_state_dict(best_weights)
    print(f"\n최고 테스트 정확도: {best_acc:.4f}")

    # 5. 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(train_losses, color="steelblue"); ax1.set_title("훈련 손실")
    ax1.set_xlabel("Epoch"); ax1.grid(True)
    ax2.plot(test_accs, color="coral"); ax2.set_title("테스트 정확도")
    ax2.set_xlabel("Epoch"); ax2.grid(True)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
