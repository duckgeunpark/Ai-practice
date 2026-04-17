"""
[Transformer 실습 2] Sinusoidal Positional Encoding (PyTorch)
실행: python positional_encoding_pytorch.py

- sin/cos로 위치 정보 주입
- 시각화 + 토큰 임베딩에 더하는 전체 흐름
- 참고: Transformer.md 섹션 4
"""

import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False


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
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def visualize_pe(max_seq_len=50, d_model=128):
    pe_layer = PositionalEncoding(d_model, max_seq_len, dropout=0.0)
    pe_matrix = pe_layer.pe.squeeze(0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 히트맵
    im = ax1.pcolormesh(pe_matrix, cmap="RdBu")
    fig.colorbar(im, ax=ax1, label="PE 값")
    ax1.set_title("Positional Encoding 히트맵")
    ax1.set_xlabel("임베딩 차원 (d_model)")
    ax1.set_ylabel("시퀀스 위치 (pos)")

    # 몇몇 차원의 위치별 변화
    for dim in [0, 1, 4, 5, 20, 21]:
        ax2.plot(pe_matrix[:, dim], label=f"dim {dim}")
    ax2.set_title("차원별 위치 변화 (짝=sin, 홀=cos)")
    ax2.set_xlabel("위치")
    ax2.set_ylabel("값")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout(); plt.show()


def demo_embedding_plus_pe():
    """임베딩 + PE 더하기 데모"""
    vocab_size = 1000
    d_model = 128
    max_len = 20

    embedding = nn.Embedding(vocab_size, d_model)
    pe = PositionalEncoding(d_model, max_len, dropout=0.0)

    tokens = torch.randint(1, vocab_size, (2, 10))        # (batch=2, seq=10)
    emb = embedding(tokens) * math.sqrt(d_model)          # √d_model 스케일링
    out = pe(emb)

    print(f"tokens shape : {tokens.shape}")
    print(f"embedding   : {emb.shape}")
    print(f"+ PE        : {out.shape}")
    print(f"임베딩 평균 |값|: {emb.abs().mean().item():.4f}")
    print(f"PE    평균 |값|: {pe.pe.abs().mean().item():.4f}")
    print("→ √d_model 스케일링으로 임베딩이 PE에 묻히지 않음")


def main():
    torch.manual_seed(42)
    demo_embedding_plus_pe()
    print()
    visualize_pe(max_seq_len=50, d_model=128)


if __name__ == "__main__":
    main()
