"""
[RNN 실습 2] PyTorch LSTM — 시계열 예측 (항공 승객 수)
실행: python flights_lstm_pytorch.py

"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

SEED = 42
WINDOW_SIZE = 12


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


def train_model(
    model, train_loader, val_loader, device, epochs=300, lr=1e-3, patience=30
):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_weights = None
    patience_cnt = 0

    for epoch in range(epochs):
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
                print(f"  Early stopping @ epoch {epoch+1} (best val={best_val:.6f})")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train {tr_loss:.6f} | Val {va_loss:.6f}")

    model.load_state_dict(best_weights)
    return model, history


def main():
    # 5-1. 차분 전처리와 train/val/test 분리
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device} | seed={SEED}")

    # 1. 원본 데이터
    df = sns.load_dataset("flights")
    data = df["passengers"].values.astype("float32")  # (144,)
    train_size = int(len(data) * 0.8)  # 115
    print(f"원본 길이: {len(data)}, 훈련: {train_size}, 테스트: {len(data)-train_size}")

    # 2. 차분(Differencing) — 전월 대비 변화량으로 상향 추세 제거
    diff = np.diff(data)  # (143,)
    # diff[i] = data[i+1] - data[i]

    # train/test 차분 분리
    #  - train 내부 전이: diff[0 : train_size-1]
    #  - test 구간 전이(경계 포함): diff[train_size-1 : ]
    train_diff_full = diff[: train_size - 1]  # 114
    test_diff = diff[train_size - 1 :]  # 29

    # 3. 훈련셋 내부에서 다시 val 분리 (뒤 20%)
    val_size = int(len(train_diff_full) * 0.2)
    tr_diff = train_diff_full[:-val_size]  # 92
    va_diff = train_diff_full[-val_size:]  # 22
    print(f"차분 분할 — train:{len(tr_diff)} val:{len(va_diff)} test:{len(test_diff)}")

    # 4. 스케일링 (차분은 음수 포함 → StandardScaler)
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(tr_diff.reshape(-1, 1)).flatten()
    va_scaled = scaler.transform(va_diff.reshape(-1, 1)).flatten()
    te_scaled = scaler.transform(test_diff.reshape(-1, 1)).flatten()

    # 5. 슬라이딩 윈도우 Dataset/Loader
    train_ds = SeqDataset(tr_scaled, WINDOW_SIZE)
    val_ds = SeqDataset(va_scaled, WINDOW_SIZE)
    test_ds = SeqDataset(te_scaled, WINDOW_SIZE)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"윈도우 샘플 — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    # 6. 모델 & 학습
    model = LSTMModel(hidden_size=32, num_layers=1, dropout=0.2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params:,}")

    model, history = train_model(
        model, train_loader, val_loader, device, epochs=300, lr=1e-3, patience=30
    )

    # 7. 테스트 예측 (차분 스케일)
    model.eval()
    pred_scaled, true_scaled = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred_scaled.extend(model(X).cpu().numpy())
            true_scaled.extend(y.numpy())

    # 8. 차분 역스케일
    pred_diff = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
    true_diff = scaler.inverse_transform(np.array(true_scaled).reshape(-1, 1)).flatten()

    # 9. 원래 승객 수로 복원
    # test_ds의 i번째 샘플은 te_scaled[i+W]를 예측
    # 이는 diff 배열 전체에서 index (train_size-1 + i + W)
    # 즉 data[train_size + i + W] - data[train_size - 1 + i + W] 를 예측한 것
    # 복원: pred_val[i] = data[train_size - 1 + i + W] + pred_diff[i]
    offset = train_size - 1 + WINDOW_SIZE
    base_vals = data[offset : offset + len(pred_diff)]  # 각 예측의 "직전 실측치"
    y_pred = base_vals + pred_diff
    y_true = data[offset + 1 : offset + 1 + len(pred_diff)]  # 실제 다음 값

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nPyTorch LSTM (diff) — RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # 10. 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history["train"], label="Train loss")
    ax1.plot(history["val"], label="Val loss", linestyle="--")
    ax1.set_title("학습 곡선 (차분 + MSE)")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(y_true, label="실제", color="steelblue", marker="o")
    ax2.plot(y_pred, label="예측", color="red", linestyle="--", marker="x")
    ax2.set_title(f"테스트 구간 예측 (RMSE={rmse:.2f})")
    ax2.set_xlabel("테스트 샘플 index")
    ax2.set_ylabel("승객 수 (천 명)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # 전체 데이터 위 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(data)), data, label="실제 승객 수", color="steelblue")
    start_idx = offset + 1
    plt.plot(
        range(start_idx, start_idx + len(y_pred)),
        y_pred,
        label="예측 (차분 학습)",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.axvline(x=train_size, color="gray", linestyle=":", label="훈련/테스트 경계")
    plt.title("항공 승객 수 예측 — PyTorch LSTM")
    plt.xlabel("월")
    plt.ylabel("승객 수 (천 명)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
