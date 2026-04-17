"""
[Transformer 실습 1] PyTorch Multi-Head Attention 직접 구현
실행: python mha_pytorch.py

- Scaled Dot-Product Attention + Multi-Head 분리/합치기
- 참고: Transformer.md 섹션 3
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V"""

    def forward(self, Q, K, V, mask=None):
        # Q/K/V: (batch, heads, seq_len, d_k)
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어 떨어져야 합니다."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """(batch, seq_len, d_model) → (batch, heads, seq_len, d_k)"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.split_heads(self.W_Q(Q), batch_size)
        K = self.split_heads(self.W_K(K), batch_size)
        V = self.split_heads(self.W_V(V), batch_size)

        output, attn_weights = self.attention(Q, K, V, mask)

        # (batch, heads, seq, d_k) → (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        return self.W_O(output), attn_weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # Self-Attention: Q=K=V=x
    output, weights = mha(x, x, x)
    print(f"입력   shape: {x.shape}")        # (2, 5, 512)
    print(f"출력   shape: {output.shape}")   # (2, 5, 512)
    print(f"가중치 shape: {weights.shape}")  # (2, 8, 5, 5)

    # 마스크 테스트 — 마지막 토큰 무시
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, -1] = 0
    out_masked, w_masked = mha(x, x, x, mask)
    print(f"\n마스킹 적용 후 마지막 위치 가중치 합: "
          f"{w_masked[..., -1].sum().item():.4f}  (0에 가까워야 함)")

    # 파라미터 수
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"\nMHA 파라미터 수: {n_params:,} (= 4 * d_model^2 + 4 * d_model)")


if __name__ == "__main__":
    main()
