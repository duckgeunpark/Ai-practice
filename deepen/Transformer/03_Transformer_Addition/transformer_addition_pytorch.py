"""
[Transformer 실습 3] PyTorch Transformer 직접 구현 — 숫자 덧셈 Seq2Seq
실행: python transformer_addition_pytorch.py

- Multi-Head Attention / Positional Encoding / Encoder / Decoder 직접 조립
- 문자 단위 덧셈 학습 ("153+287" → "440") + Greedy 디코딩
- 참고: Transformer.md 섹션 6~7
"""

import math
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

SEED = 42
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
SRC_MAX_LEN = 9
TGT_MAX_LEN = 6


# ────────────────────────────────────────────────────────────
# 토큰화
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# Transformer 구성요소
# ────────────────────────────────────────────────────────────
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        b = Q.size(0)
        Q = self.split_heads(self.W_Q(Q), b)
        K = self.split_heads(self.W_K(K), b)
        V = self.split_heads(self.W_V(V), b)
        out, attn = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.W_O(out), attn


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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=3,
        max_seq_len=20,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=PAD_TOKEN):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=PAD_TOKEN):
        tgt_len = tgt.size(1)
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        lookahead = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return pad_mask & lookahead.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, pad_idx=PAD_TOKEN):
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────────────────────
def warmup_factor(step, warmup_steps=400):
    # 0 → 1 로 선형 증가 후 1 유지 (base_lr 에 곱해짐)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0


def train(model, train_loader, val_loader, device, epochs=30):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_factor(step)
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
            print(
                f"Epoch {epoch+1:2d}/{epochs} | Train {tr:.4f} | Val {va:.4f} "
                f"| LR {scheduler.get_last_lr()[0]:.6f}"
            )

    return history


# ────────────────────────────────────────────────────────────
# Greedy 디코딩
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, src_str, device, max_len=TGT_MAX_LEN):
    model.eval()
    src = encode(src_str)
    src = (src + [PAD_TOKEN] * (SRC_MAX_LEN - len(src)))[:SRC_MAX_LEN]
    src = torch.tensor([src], dtype=torch.long).to(device)

    src_mask = model.make_src_mask(src)
    enc_out = model.encoder(src, src_mask)

    tgt_tokens = [SOS_TOKEN]
    for _ in range(max_len):
        tgt = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        tgt_mask = model.make_tgt_mask(tgt)
        dec_out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits = model.fc_out(dec_out)
        next_tok = logits[:, -1, :].argmax(dim=-1).item()
        tgt_tokens.append(next_tok)
        if next_tok == EOS_TOKEN:
            break
    return decode(tgt_tokens)


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device} | vocab={VOCAB_SIZE}")

    # 1. 데이터
    data = generate_data(num_samples=10000, max_val=500)
    random.shuffle(data)
    train_pairs, val_pairs = data[:9000], data[9000:]

    train_loader = DataLoader(
        AdditionDataset(train_pairs), batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        AdditionDataset(val_pairs), batch_size=128, shuffle=False, num_workers=0
    )
    print("샘플:", train_pairs[:3])

    # 2. 모델
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=3,
        max_seq_len=20,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params:,}")

    # 3. 학습
    history = train(model, train_loader, val_loader, device, epochs=30)

    # 4. 테스트
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

    # 5. 학습 곡선
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="훈련 손실")
    plt.plot(history["val_loss"], label="검증 손실", linestyle="--")
    plt.title("Transformer 학습 손실")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
