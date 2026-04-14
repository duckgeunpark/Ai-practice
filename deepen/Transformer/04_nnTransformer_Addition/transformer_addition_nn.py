"""
[Transformer 실습 4] PyTorch nn.Transformer로 덧셈 Seq2Seq 재구현
실행: python transformer_addition_nn.py

- 실습 3을 nn.Transformer 내장 모듈로 옮겨 비교
- 마스크 관례 차이 주의: key_padding_mask/tgt_mask 는 True=차단
- 참고: Transformer.md 섹션 8
"""

import math
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

SEED = 42
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
SRC_MAX_LEN = 9
TGT_MAX_LEN = 6


_chars = list("0123456789+")
char2idx = {c: i + 3 for i, c in enumerate(_chars)}
char2idx["<pad>"] = PAD_TOKEN
char2idx["<sos>"] = SOS_TOKEN
char2idx["<eos>"] = EOS_TOKEN
idx2char = {v: k for k, v in char2idx.items()}
VOCAB_SIZE = len(char2idx)


def encode(text):
    return [char2idx[c] for c in text]


def decode(indices):
    result = []
    for idx in indices:
        if idx == EOS_TOKEN:
            break
        if idx not in (PAD_TOKEN, SOS_TOKEN):
            result.append(idx2char.get(idx, "?"))
    return "".join(result)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        max_seq_len=20,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_pad_mask(self, seq, pad_idx=PAD_TOKEN):
        # True = 차단할 패딩 위치
        return seq == pad_idx

    def make_lookahead_mask(self, sz, device):
        # True = 차단
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src, tgt):
        src_kpm = self.make_pad_mask(src)
        tgt_kpm = self.make_pad_mask(tgt)
        tgt_mask = self.make_lookahead_mask(tgt.size(1), tgt.device)

        src_emb = self.pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.fc_out(output)


def generate_data(num_samples=10000, max_val=500):
    data = []
    for _ in range(num_samples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        data.append((f"{a}+{b}", str(a + b)))
    return data


class AdditionDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_str, tgt_str = self.data[idx]

        src = encode(src_str)
        src = (src + [PAD_TOKEN] * (SRC_MAX_LEN - len(src)))[:SRC_MAX_LEN]

        tgt_in = [SOS_TOKEN] + encode(tgt_str)
        tgt_in = (tgt_in + [PAD_TOKEN] * (TGT_MAX_LEN - len(tgt_in)))[:TGT_MAX_LEN]

        tgt_out = encode(tgt_str) + [EOS_TOKEN]
        tgt_out = (tgt_out + [PAD_TOKEN] * (TGT_MAX_LEN - len(tgt_out)))[:TGT_MAX_LEN]

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def get_lr(step, d_model=128, warmup_steps=400):
    if step == 0:
        step = 1
    return (d_model**-0.5) * min(step**-0.5, step * warmup_steps**-1.5)


def train(model, train_loader, val_loader, device, epochs=100):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    # LambdaLR 은 optimizer.lr 에 `lr_lambda(step)` 를 곱합니다.
    # 논문의 warmup 공식은 "절대 LR"이므로 base_lr=1.0 으로 두어야 공식 값이 그대로 들어갑니다.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: get_lr(step)
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            output = model(src, tgt_in)
            loss = criterion(output.view(-1, VOCAB_SIZE), tgt_out.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        tr = total / len(train_loader)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for src, tgt_in, tgt_out in val_loader:
                src, tgt_in, tgt_out = (
                    src.to(device),
                    tgt_in.to(device),
                    tgt_out.to(device),
                )
                output = model(src, tgt_in)
                vtotal += criterion(
                    output.view(-1, VOCAB_SIZE), tgt_out.view(-1)
                ).item()
        va = vtotal / len(val_loader)

        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train {tr:.4f} | Val {va:.4f}")

    return history


@torch.no_grad()
def predict(model, src_str, device, max_len=TGT_MAX_LEN):
    model.eval()
    src = encode(src_str)
    src = (src + [PAD_TOKEN] * (SRC_MAX_LEN - len(src)))[:SRC_MAX_LEN]
    src_t = torch.tensor([src], dtype=torch.long).to(device)

    tgt_tokens = [SOS_TOKEN]
    for _ in range(max_len):
        tgt_t = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        logits = model(src_t, tgt_t)
        next_tok = logits[:, -1, :].argmax(dim=-1).item()
        tgt_tokens.append(next_tok)
        if next_tok == EOS_TOKEN:
            break
    return decode(tgt_tokens)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device} | vocab={VOCAB_SIZE}")

    data = generate_data(num_samples=10000, max_val=500)
    random.shuffle(data)
    train_pairs, val_pairs = data[:9000], data[9000:]

    train_loader = DataLoader(
        AdditionDataset(train_pairs), batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        AdditionDataset(val_pairs), batch_size=128, shuffle=False, num_workers=0
    )

    model = TransformerModel(vocab_size=VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params:,}")

    history = train(model, train_loader, val_loader, device, epochs=100)

    test_cases = [
        ("123+456", "579"),
        ("999+1", "1000"),
        ("42+58", "100"),
        ("0+0", "0"),
        ("300+200", "500"),
        ("77+88", "165"),
    ]
    print("\n" + "=" * 44)
    print(f"{'입력':12s} {'정답':8s} {'예측':8s} 결과")
    print("=" * 44)
    correct = 0
    for src_str, answer in test_cases:
        pred = predict(model, src_str, device)
        mark = "OK " if pred == answer else "X  "
        if pred == answer:
            correct += 1
        print(f"{src_str:12s} {answer:8s} {pred:8s} {mark}")
    print("=" * 44)
    print(f"정확도: {correct}/{len(test_cases)}")

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="훈련 손실")
    plt.plot(history["val_loss"], label="검증 손실", linestyle="--")
    plt.title("nn.Transformer 학습 손실")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
