[미검증]
## 0. 시리즈

| 심화편 | 제목 | 난이도 | 역할 |
| :-- | :-- | :-- | :-- |
| [심화 1편](https://duckport.pages.dev/posts/CNN_Deep) | CNN — 이미지 분류 완벽 정리 | ⭐⭐ | 이미지 처리 |
| [심화 2편](https://duckport.pages.dev/posts/RNN) | RNN / LSTM — 시계열·텍스트 처리 | ⭐⭐⭐ | 순서 데이터 |
| [**심화 3편**](https://duckport.pages.dev/posts/Transformer)⬅️ | Transformer 구조 완벽 정리 | ⭐⭐⭐⭐ | 현대 AI 핵심 |
| [심화 4편](https://duckport.pages.dev/posts/BERT) | BERT / GPT 원리와 활용 | ⭐⭐⭐⭐ | 언어 모델 |
| [심화 5편](https://duckport.pages.dev/posts/Finetuning_Deep) | HuggingFace로 LLM Fine-tuning | ⭐⭐⭐⭐⭐ | LLM 실전 |


***

## 1. Transformer가 왜 등장했는가?

### 1-1. LSTM(2편)의 한계

2편에서 배운 LSTM은 순서 데이터를 잘 처리하지만 두 가지 근본적인 한계가 있습니다.

```
한계 1. 병렬 처리 불가
────────────────────────────────────────
LSTM: x₁ → h₁ → x₂ → h₂ → x₃ → h₃ ...
      ↑ 반드시 앞 시점이 끝나야 다음 시점 처리 가능
      → 문장이 길수록 학습 시간 폭발적 증가

한계 2. 장거리 의존성 약화
────────────────────────────────────────
"나는 어제 서울에서 열린 콘서트에 갔는데 정말 재미있었다."
  ↑ "나"                              ↑ "재미있었다"
  → 멀리 떨어진 단어끼리의 관계를
    LSTM은 중간을 거치며 희석됨
```


### 1-2. "Attention is All You Need" (2017)

2017년 구글이 발표한 논문 **"Attention is All You Need"** 는 RNN/LSTM을 완전히 제거하고 **Attention 메커니즘만으로** 언어 모델을 만들 수 있음을 증명했습니다.

```
기존 방식: RNN + Attention (보조 수단으로 Attention 사용)
Transformer: Attention만 사용 (RNN 완전 제거)
```

결과적으로 두 문제를 한 번에 해결했습니다.

- **병렬 처리** — 모든 단어를 동시에 처리 (GPU 활용 극대화)
- **장거리 의존성** — 어떤 단어끼리든 직접 연결하여 관계 파악


### 1-3. Transformer가 현대 AI의 기반이 된 이유

```
Transformer (2017)
       ↓
  BERT (2018) — 구글, 문서 이해
  GPT  (2018) — OpenAI, 문서 생성
       ↓
  GPT-3 / GPT-4 / ChatGPT
  Claude / Gemini / LLaMA
  → 현재 모든 대형 언어 모델(LLM)의 기반 구조
```

BERT, GPT 등 4편에서 다룰 모든 모델이 Transformer를 기반으로 만들어집니다.

***

## 2. Attention 메커니즘

### 2-1. Attention이란? — "어디를 집중해서 볼까?"

사람이 문장을 읽을 때를 생각해보세요.

```
"그 영화는 배우의 연기가 훌륭했지만 스토리가 별로였다."

"훌륭했지만" 을 이해할 때 → "연기"에 집중
"별로였다"  를 이해할 때 → "스토리"에 집중
```

Attention은 이처럼 **현재 처리 중인 단어와 관련이 높은 다른 단어에 더 많은 가중치를 부여** 하는 메커니즘입니다.

### 2-2. Query, Key, Value란?

Attention을 도서관에 비유하면 이해하기 쉽습니다.

```
Query (Q) = 내가 찾고 있는 것 (검색어)
Key   (K) = 책마다 붙어있는 색인 태그 (책 제목/주제)
Value (V) = 실제 책의 내용

과정:
1. 검색어(Q)와 모든 책의 태그(K)를 비교
2. 가장 관련 있는 책을 찾음 (Attention Score)
3. 관련도에 따라 책 내용(V)을 가중 합산
```

수식으로 표현하면 아래와 같습니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $QK^T$ : Q와 K의 내적 → 유사도 점수 계산
- $\sqrt{d_k}$ : 차원 수의 제곱근으로 나누기 → 점수가 너무 커지는 것 방지
- $\text{softmax}$ : 점수를 0~1 확률로 변환 (합계 = 1)
- $\times V$ : 확률로 V를 가중 합산


### 2-3. Attention Score 계산 과정 (단계별)

```
예시 문장: "나는 밥을 먹었다" (3개 단어)
각 단어를 4차원 벡터로 표현했다고 가정

Step 1. 입력 임베딩 → Q, K, V 생성 (선형 변환)
─────────────────────────────────────────────────
입력 X → Q = X @ W_Q
         K = X @ W_K
         V = X @ W_V
(W_Q, W_K, W_V 는 학습되는 가중치 행렬)

Step 2. Q × K^T → 유사도 행렬 계산
─────────────────────────────────────────────────
         나는  밥을  먹었다
나는   [ 1.2   0.3   0.8 ]   ← "나는"이 각 단어와 얼마나 관련?
밥을   [ 0.2   1.5   0.9 ]
먹었다 [ 0.7   1.1   1.3 ]

Step 3. √d_k 로 나누기 (스케일링)
─────────────────────────────────────────────────
d_k = 4 (key 차원)  →  √4 = 2로 나눔
→ 값이 너무 커서 softmax가 포화되는 현상 방지

Step 4. Softmax → 확률 분포
─────────────────────────────────────────────────
"나는"  행: [0.60, 0.10, 0.30]   ← "나는" 자신에 가장 집중
"밥을"  행: [0.10, 0.55, 0.35]
"먹었다"행: [0.20, 0.35, 0.45]

Step 5. × V → 최종 Attention 출력
─────────────────────────────────────────────────
각 단어의 최종 표현 = 확률에 따라 V를 가중 합산한 벡터
```


### 2-4. Self-Attention이란?

Transformer에서는 Q, K, V **모두 같은 입력** 에서 만들어집니다. 즉 문장이 자기 자신과 Attention을 계산합니다.

```
입력: "나는 밥을 먹었다"
   ↓
Q, K, V 모두 이 문장에서 생성
   ↓
"먹었다" 입장에서 "나는", "밥을", "먹었다" 와의 관계 파악
→ "먹다"는 "나는"(주어)과 "밥을"(목적어) 모두와 관련있음을 학습
```

이것이 **Self-Attention(자기 주의)** 입니다. Transformer의 핵심 개념입니다.

***

## 3. Multi-Head Attention

### 3-1. 왜 Head를 여러 개 쓰는가?

Attention을 한 번만 하면 한 가지 관점에서만 관계를 파악합니다. **Head를 여러 개** 사용하면 여러 관점에서 동시에 관계를 파악할 수 있습니다.

```
Head 1 → 문법적 관계에 집중 ("주어-동사")
Head 2 → 의미적 관계에 집중 ("먹다-음식")
Head 3 → 위치적 관계에 집중 (인접 단어)
...
Head N → 또 다른 관점

→ 모든 Head의 출력을 이어 붙여(concat) 선형 변환
```


### 3-2. Multi-Head Attention 구조

```
입력 X
  ↓ (3번 복사)
[Q] [K] [V]
  ↓        ↓        ↓
Head1: Attention(Q₁,K₁,V₁) → 출력₁
Head2: Attention(Q₂,K₂,V₂) → 출력₂
Head3: Attention(Q₃,K₃,V₃) → 출력₃
Head4: Attention(Q₄,K₄,V₄) → 출력₄
  ↓
[출력₁ | 출력₂ | 출력₃ | 출력₄]  → Concat
  ↓
선형 변환 (W_O 행렬)
  ↓
최종 출력 (입력과 같은 차원)
```

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)
$$

### 3-3. PyTorch로 Multi-Head Attention 직접 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q: (batch, heads, seq_len, d_k)
        # K: (batch, heads, seq_len, d_k)
        # V: (batch, heads, seq_len, d_v)

        d_k = Q.size(-1)  # Key 차원

        # Step 1. Q × K^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: (batch, heads, seq_len, seq_len)

        # Step 2. 마스크 적용 (padding 위치 or 미래 토큰 차단)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 3. Softmax → 확률 분포
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: (batch, heads, seq_len, seq_len)

        # Step 4. × V → 가중 합산
        output = torch.matmul(attn_weights, V)
        # output: (batch, heads, seq_len, d_v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model   : 전체 임베딩 차원 (예: 512)
        num_heads : Head 수 (예: 8)
        d_k = d_model // num_heads (예: 64)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어 떨어져야 합니다."

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # 각 Head의 차원

        # Q, K, V 선형 변환 (각 Head용 한꺼번에)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 최종 출력 선형 변환
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        각 Head가 독립적으로 Attention 계산하도록 차원 분리
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 선형 변환
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # Head 분리
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot-Product Attention
        output, attn_weights = self.attention(Q, K, V, mask)
        # output: (batch, heads, seq_len, d_k)

        # Head 합치기: (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        # 최종 선형 변환
        output = self.W_O(output)  # (batch, seq_len, d_model)
        return output, attn_weights


# 동작 확인
if __name__ == "__main__":
    batch_size = 2
    seq_len    = 5
    d_model    = 512
    num_heads  = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x   = torch.randn(batch_size, seq_len, d_model)  # 임의 입력

    output, weights = mha(x, x, x)  # Self-Attention: Q=K=V=x
    print(f"입력  shape: {x.shape}")       # (2, 5, 512)
    print(f"출력  shape: {output.shape}")  # (2, 5, 512)
    print(f"가중치 shape: {weights.shape}")# (2, 8, 5, 5)
```
"""
입력   shape: torch.Size([2, 5, 512])
출력   shape: torch.Size([2, 5, 512])
가중치 shape: torch.Size([2, 8, 5, 5])

마스킹 적용 후 마지막 위치 가중치 합: 0.0000  (0에 가까워야 함)

MHA 파라미터 수: 1,050,624 (= 4 * d_model^2 + 4 * d_model)
"""

***

## 4. Positional Encoding

### 4-1. 왜 위치 정보가 필요한가?

LSTM은 단어를 순서대로 처리하므로 위치 정보가 자동으로 포함됩니다. 하지만 Transformer는 모든 단어를 **동시에 병렬로** 처리하기 때문에 위치 정보가 없습니다.

```
Transformer 입장에서는 아래 두 문장이 동일하게 보임:
"나는 밥을 먹었다"
"밥을 나는 먹었다"
→ 단어 집합이 같기 때문!

해결책: 각 단어의 임베딩에 위치 정보를 더해준다
입력 = 단어 임베딩 + 위치 임베딩 (Positional Encoding)
```


### 4-2. Sinusoidal Positional Encoding

논문에서는 sin/cos 함수를 사용하여 위치 정보를 인코딩합니다.

$$
PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- `pos` : 단어의 위치 (0번째, 1번째, ...)
- `i` : 임베딩 차원의 인덱스
- 짝수 차원 → sin, 홀수 차원 → cos

이렇게 하면 각 위치마다 **고유한 패턴**의 벡터가 생성됩니다.

### 4-3. 코드 구현 및 시각화

```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        d_model     : 임베딩 차원 (예: 512)
        max_seq_len : 처리할 수 있는 최대 시퀀스 길이
        dropout     : 드롭아웃 비율
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE 행렬 초기화: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        # 위치 벡터: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 분모 계산: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 짝수 차원 → sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 차원 → cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_seq_len, d_model) 형태로 변환 (batch 차원 추가)
        pe = pe.unsqueeze(0)

        # 학습되지 않는 버퍼로 등록 (모델 저장 시 포함됨)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # pe는 자동으로 x의 seq_len에 맞게 슬라이싱
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 시각화: 히트맵 + 차원별 위치 변화
def visualize_pe(max_seq_len=50, d_model=128):
    pe_layer = PositionalEncoding(d_model, max_seq_len, dropout=0.0)
    pe_matrix = pe_layer.pe.squeeze(0).numpy()   # (max_seq_len, d_model)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ① 히트맵 — 위치×차원 전체 패턴
    im = ax1.pcolormesh(pe_matrix, cmap="RdBu")
    fig.colorbar(im, ax=ax1, label="PE 값")
    ax1.set_title("Positional Encoding 히트맵")
    ax1.set_xlabel("임베딩 차원 (d_model)")
    ax1.set_ylabel("시퀀스 위치 (pos)")

    # ② 몇몇 차원의 위치별 변화 — 짝=sin, 홀=cos
    for dim in [0, 1, 4, 5, 20, 21]:
        ax2.plot(pe_matrix[:, dim], label=f"dim {dim}")
    ax2.set_title("차원별 위치 변화 (짝=sin, 홀=cos)")
    ax2.set_xlabel("위치"); ax2.set_ylabel("값")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout(); plt.show()

visualize_pe(max_seq_len=50, d_model=128)
```

**히트맵과 라인플롯이 함께 보여주는 것**
- 히트맵: 각 행(위치)마다 고유한 컬러 패턴 → 위치마다 서로 다른 PE 벡터
- 라인플롯: 낮은 차원(dim 0/1)은 주기가 길고, 높은 차원(dim 20/21)은 주기가 짧음 → 여러 주파수의 sin/cos 조합으로 위치를 인코딩


### 4-4. 임베딩에 PE 더하기 — √d_model 스케일링

실제 Transformer 에서는 단어 임베딩에 PE 를 더해서 입력을 만듭니다. 이때 임베딩에 `√d_model` 을 곱하는 **스케일링 단계**가 중요합니다.

```python
def demo_embedding_plus_pe():
    vocab_size = 1000
    d_model    = 128
    max_len    = 20

    embedding = nn.Embedding(vocab_size, d_model)
    pe        = PositionalEncoding(d_model, max_len, dropout=0.0)

    tokens = torch.randint(1, vocab_size, (2, 10))          # (batch=2, seq=10)
    emb    = embedding(tokens) * math.sqrt(d_model)         # ← √d_model 스케일링
    out    = pe(emb)                                        # 임베딩 + PE

    print(f"tokens shape : {tokens.shape}")
    print(f"embedding    : {emb.shape}")
    print(f"+ PE         : {out.shape}")
    print(f"임베딩 평균 |값|: {emb.abs().mean().item():.4f}")
    print(f"PE    평균 |값|: {pe.pe.abs().mean().item():.4f}")
    print("→ √d_model 스케일링으로 임베딩이 PE에 묻히지 않음")

demo_embedding_plus_pe()
```
[]이미지
"""
tokens shape : torch.Size([2, 10])
embedding   : torch.Size([2, 10, 128])
+ PE        : torch.Size([2, 10, 128])
임베딩 평균 |값|: 9.3175
PE    평균 |값|: 0.5657
→ √d_model 스케일링으로 임베딩이 PE에 묻히지 않음
"""

**왜 √d_model 을 곱하는가?**
- `nn.Embedding` 의 기본 초기화는 평균 0, 표준편차 1 근처 → 임베딩 벡터의 크기가 상대적으로 작음
- PE 는 sin/cos 값이라 `[-1, 1]` 범위 → 크기가 약 0.5~0.7 수준
- 임베딩에 `√d_model` (예: √128 ≈ 11.3) 을 곱해주지 않으면, 임베딩 정보가 PE 에 압도당함
- 스케일링 후 임베딩 평균 |값| ≈ 9, PE 평균 |값| ≈ 0.57 → 임베딩이 주 정보로 유지되면서 PE 가 위치 보정 신호로 작용

> 💡 이 스케일링은 Transformer 섹션 6~7 의 Encoder/Decoder 에서 `self.embedding(x) * math.sqrt(self.d_model)` 로 반복 등장합니다.


***

## 5. Transformer 전체 구조

### 5-1. Encoder 블록 구조

Encoder는 동일한 블록을 N번 쌓습니다 (논문 기준 6개).

```
입력 시퀀스
     ↓
 임베딩 + Positional Encoding
     ↓
 ┌────────────────────┐
 │         Encoder Block × N       │
 │                                 │
 │  ┌───────────────┐  │
 │  │  Multi-Head Self-       │  │
 │  │  Attention              │  │
 │  └──────┬────────┘  │
 │              ↓                 │
 │  ┌───────────────┐  │
 │  │  Add & Norm             │  │  ← 잔차 연결 + 레이어 정규화
 │  │  (입력 + Attention 출력) │  │
 │  └──────┬────────┘  │
 │              ↓                 │
 │  ┌───────────────┐  │
 │  │  Feed Forward Network   │  │  ← 위치별 독립 MLP
 │  └──────┬────────┘  │
 │              ↓                 │
 │  ┌───────────────┐  │
 │  │  Add & Norm             │  │
 │  └──────┬────────┘  │
 └────────┼──────────┘
                ↓
          Encoder 출력
```

**Add \& Norm** 이란?

```python
# 잔차 연결(Residual Connection) + 레이어 정규화
output = LayerNorm(x + SubLayer(x))
# x           : 이 블록의 입력 (그대로 더해줌 → 기울기 소실 방지)
# SubLayer(x) : Attention 또는 FFN의 출력
```

**Feed Forward Network(FFN)** 란?

```python
# 각 위치마다 독립적으로 적용되는 2층 MLP
FFN(x) = max(0, x @ W₁ + b₁) @ W₂ + b₂
# 내부 차원(d_ff)은 d_model의 4배 사용 (512 → 2048 → 512)
```


### 5-2. Decoder 블록 구조

Decoder는 Encoder와 달리 3개의 서브레이어를 갖습니다.

```
번역 타겟 시퀀스 (이미 생성된 부분)
     ↓
 임베딩 + Positional Encoding
     ↓
 ┌──────────────────────────┐
 │              Decoder Block × N            │
 │                                           │
 │  ┌─────────────────────┐   │
 │  │  Masked Multi-Head Self-Attention │   │  ← 미래 토큰 차단
 │  └──────────┬──────────┘   │
 │                     ↓                     │
 │  ┌─────────────────────┐   │
 │  │  Add & Norm                       │   │
 │  └──────────┬──────────┘   │
 │                     ↓                     │
 │  ┌─────────────────────┐   │
 │  │  Cross Attention                  │   │  ← Encoder 출력 참조
 │  │  Q: Decoder, K/V: Encoder 출력     │  │
 │  └──────────┬──────────┘   │
 │                     ↓                     │
 │  ┌─────────────────────┐   │
 │  │  Add & Norm                       │   │
 │  └──────────┬──────────┘   │
 │                     ↓                     │
 │  ┌─────────────────────┐   │
 │  │  Feed Forward Network             │   │
 │  └──────────┬──────────┘   │
 │                     ↓                     │
 │  ┌─────────────────────┐   │
 │  │  Add & Norm                       │   │
 │  └──────────┬──────────┘   │
 └──────────────────────────┘
                       ↓
              Linear + Softmax
                       ↓
               다음 단어 예측
```

**Masked Self-Attention** 이란?

```
번역 중 "I love"까지 생성한 상태에서 다음 단어 예측:
→ "you"를 예측할 때 미래의 정답 단어를 보면 안 됨!

Look-ahead Mask:
         I    love   you
I    [ 가능  차단   차단 ]
love [ 가능  가능   차단 ]   ← 현재까지만 볼 수 있음
you  [ 가능  가능   가능 ]
```

**Cross Attention** 이란?

```
Q = Decoder의 현재 출력 ("love"를 처리 중)
K = Encoder의 출력 (원문 "나는 너를 사랑해" 전체)
V = Encoder의 출력 (원문 전체)

→ "love"가 원문의 "사랑해"에 집중하도록 학습
```


### 5-3. 전체 흐름 (번역 예시)

```
[입력] "나는 너를 사랑해"
          ↓
      Encoder × 6
          ↓
  [Encoder 출력] (문장의 의미가 압축된 벡터)
          ↓                   ↑ Cross Attention으로 참조
      Decoder × 6 ───────┘
          ↓
    [<start>] → "I" → "love" → "you" → [<end>]
               ↑ 순차적으로 생성
```


***

## 6. PyTorch로 Transformer 직접 구현하기

### 6-1. Feed Forward Network 구현

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model : 입출력 차원 (예: 512)
        d_ff    : 내부 확장 차원 (d_model * 4 = 2048)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)  # (batch, seq_len, d_model)
```


### 6-2. Encoder 블록 구현

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = FeedForward(d_model, d_ff, dropout)

        # 각 서브레이어마다 LayerNorm 1개씩
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ── 서브레이어 1: Multi-Head Self-Attention + Add & Norm ──
        attn_out, _ = self.self_attn(x, x, x, mask)   # Q=K=V=x (Self)
        x = self.norm1(x + self.dropout(attn_out))      # 잔차 연결 후 정규화

        # ── 서브레이어 2: FFN + Add & Norm ──────────────────
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)                     # 잔차 연결 후 정규화

        return x  # (batch, seq_len, d_model)
```


### 6-3. Decoder 블록 구현

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)  # Masked Self-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # Cross Attention
        self.ffn        = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # ── 서브레이어 1: Masked Multi-Head Self-Attention ───────
        # tgt_mask: 미래 토큰 차단 (look-ahead mask)
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # ── 서브레이어 2: Cross Attention ────────────────────────
        # Q = Decoder의 현재 출력
        # K, V = Encoder의 출력
        # src_mask: 패딩 토큰 무시
        cross_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # ── 서브레이어 3: FFN ───────────────────────────────
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x  # (batch, tgt_seq_len, d_model)
```


### 6-4. Encoder / Decoder 전체 조립

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (batch, src_seq_len) — 단어 인덱스
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        # √d_model 스케일링 — 임베딩 값이 PE보다 작아지지 않도록
        x = self.pos_enc(x)   # Positional Encoding 더하기
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)   # (batch, src_seq_len, d_model)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)   # (batch, tgt_seq_len, d_model)
```


### 6-5. 전체 Transformer 모델 클래스

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads,
                               d_ff, num_layers, max_seq_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads,
                               d_ff, num_layers, max_seq_len, dropout)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화 — 학습 안정화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        """패딩 토큰(0)이 있는 위치를 마스킹"""
        # src: (batch, src_len)
        # 마스크: 패딩이 아닌 위치 = True
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch, 1, 1, src_len) → 브로드캐스팅으로 (batch, heads, q_len, k_len) 적용
        return src_mask

    def make_tgt_mask(self, tgt, pad_idx=0):
        """패딩 마스크 + Look-ahead 마스크 결합"""
        tgt_len = tgt.size(1)

        # 패딩 마스크
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch, 1, 1, tgt_len)

        # Look-ahead 마스크 (하삼각 행렬 — 미래 토큰 차단)
        lookahead_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()
        # (tgt_len, tgt_len)

        # 두 마스크 결합 (둘 다 True인 위치만 허용)
        tgt_mask = pad_mask & lookahead_mask.unsqueeze(0).unsqueeze(0)
        return tgt_mask

    def forward(self, src, tgt, pad_idx=0):
        # src: (batch, src_len) — 원문 토큰 인덱스
        # tgt: (batch, tgt_len) — 번역문 토큰 인덱스

        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        output = self.fc_out(dec_output)  # (batch, tgt_len, tgt_vocab_size)
        return output


# 모델 생성 및 구조 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

model = Transformer(
    src_vocab_size = 1000,
    tgt_vocab_size = 1000,
    d_model    = 256,   # 논문은 512, 실습에서는 256으로 경량화
    num_heads  = 8,
    d_ff       = 1024,  # d_model * 4
    num_layers = 3,     # 논문은 6, 실습에서는 3
    max_seq_len= 100,
    dropout    = 0.1
).to(device)

total = sum(p.numel() for p in model.parameters())
print(f"전체 파라미터 수: {total:,}")

# Forward 테스트
src = torch.randint(1, 1000, (2, 10)).to(device)  # 배치 2, 길이 10
tgt = torch.randint(1, 1000, (2, 8)).to(device)   # 배치 2, 길이 8

output = model(src, tgt)
print(f"입력 src shape: {src.shape}")     # (2, 10)
print(f"입력 tgt shape: {tgt.shape}")     # (2, 8)
print(f"출력      shape: {output.shape}") # (2, 8, 1000)
```

***

## 7. 실습 — 숫자 덧셈 Seq2Seq (처음부터 끝까지)

### 7-1. 문제 정의

```
입력: "153+287"  →  출력: "440"
입력: "999+1"    →  출력: "1000"
입력: "42+58"    →  출력: "100"

→ 문자 단위로 처리 (글자 1개 = 토큰 1개)
→ Transformer가 덧셈 규칙을 데이터에서 스스로 학습
```


### 7-2. 데이터 생성 및 토큰화

```python
import random
import numpy as np

# ── 특수 토큰 정의 ────────────────────────────────
PAD_TOKEN = 0   # 패딩
SOS_TOKEN = 1   # 문장 시작 (<start>)
EOS_TOKEN = 2   # 문장 종료 (<end>)

# 문자 → 인덱스 매핑
chars = list("0123456789+")
char2idx = {c: i+3 for i, c in enumerate(chars)}  # 0,1,2 는 특수 토큰
char2idx["<pad>"] = PAD_TOKEN
char2idx["<sos>"] = SOS_TOKEN
char2idx["<eos>"] = EOS_TOKEN
idx2char = {v: k for k, v in char2idx.items()}

VOCAB_SIZE = len(char2idx)
print(f"Vocabulary: {char2idx}")
print(f"Vocab 크기: {VOCAB_SIZE}")

def encode(text):
    """문자열 → 인덱스 리스트"""
    return [char2idx[c] for c in text]

def decode(indices):
    """인덱스 리스트 → 문자열"""
    result = []
    for idx in indices:
        if idx == EOS_TOKEN:
            break
        if idx not in (PAD_TOKEN, SOS_TOKEN):
            result.append(idx2char.get(idx, "?"))
    return "".join(result)


def generate_data(num_samples=10000, max_val=500):
    """덧셈 데이터셋 생성"""
    data = []
    for _ in range(num_samples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        src = f"{a}+{b}"
        tgt = str(a + b)
        data.append((src, tgt))
    return data

data = generate_data(num_samples=10000)

print("샘플 데이터:")
for src, tgt in data[:5]:
    print(f"  입력: {src:10s}  →  정답: {tgt}")
```


### 7-3. Dataset 및 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

SRC_MAX_LEN = 9   # "999+999" = 최대 7글자, 여유 포함
TGT_MAX_LEN = 6   # "1000"   = 최대 4글자, <sos>/<eos> 포함

class AdditionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_str, tgt_str = self.data[idx]

        # 인코더 입력: 그냥 입력 문자열
        src = encode(src_str)
        src = src + [PAD_TOKEN] * (SRC_MAX_LEN - len(src))
        src = src[:SRC_MAX_LEN]

        # 디코더 입력: <sos> + 정답 (학습 시 teacher forcing)
        tgt_in  = [SOS_TOKEN] + encode(tgt_str)
        tgt_in  = tgt_in + [PAD_TOKEN] * (TGT_MAX_LEN - len(tgt_in))
        tgt_in  = tgt_in[:TGT_MAX_LEN]

        # 디코더 정답: 정답 + <eos> (손실 계산 대상)
        tgt_out = encode(tgt_str) + [EOS_TOKEN]
        tgt_out = tgt_out + [PAD_TOKEN] * (TGT_MAX_LEN - len(tgt_out))
        tgt_out = tgt_out[:TGT_MAX_LEN]

        return (torch.tensor(src,     dtype=torch.long),
                torch.tensor(tgt_in,  dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long))


# 훈련/검증 분리 (9:1)
random.shuffle(data)
train_data = data[:9000]
val_data   = data[9000:]

train_ds = AdditionDataset(train_data)
val_ds   = AdditionDataset(val_data)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

# 샘플 확인
src, tgt_in, tgt_out = train_ds[0]
print(f"src     : {src.tolist()}")
print(f"tgt_in  : {tgt_in.tolist()}")
print(f"tgt_out : {tgt_out.tolist()}")
print(f"디코딩 확인: {decode(tgt_out.tolist())}")
```


### 7-4. 모델 초기화 및 학습

```python
# 모델 생성 (작은 설정으로 빠르게 학습)
model = Transformer(
    src_vocab_size = VOCAB_SIZE,
    tgt_vocab_size = VOCAB_SIZE,
    d_model    = 128,
    num_heads  = 4,
    d_ff       = 512,
    num_layers = 3,
    max_seq_len= 20,
    dropout    = 0.1
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # 패딩은 손실 계산 제외
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,
                             betas=(0.9, 0.98), eps=1e-9)

# 선형 Warm-up 스케줄러 (warmup 이후 base_lr 유지)
def warmup_factor(step, warmup_steps=400):
    """
    처음 warmup_steps 동안 0 → 1 로 선형 증가,
    이후에는 1 유지 (base_lr 에 그대로 곱해짐)
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: warmup_factor(step)
)
```

> ⚠️ **왜 논문의 `get_lr` 공식을 버렸나?**
>
> Vaswani(2017) 원 논문의 warmup 공식은 다음과 같다:
>
> ```
> lr(step) = d_model^-0.5 × min(step^-0.5, step × warmup_steps^-1.5)
> ```
>
> 이 공식은 **3단계**로 동작한다 — ① warmup 중 선형 증가, ② `warmup_steps` 지점에서 피크,
> ③ 이후 `step^-0.5` 로 감쇠. 문제는 **피크 LR 이 `d_model` 과 `warmup_steps` 에 의해
> 자동 결정**된다는 점이다.
>
> | 설정 | d_model | warmup_steps | 피크 LR |
> |---|---|---|---|
> | 원 논문 | 512 | 4000 | ≈ 7e-4 |
> | 이 실습 (초기값) | 128 | 400 | **≈ 4.4e-3** (6배 ↑) |
>
> `d_model` 을 4배 줄이고 `warmup_steps` 를 10배 줄이니 피크 LR 이 논문 대비 6배로
> 과도해져서, epoch 6 근처부터 loss 가 오히려 상승(발산)하고 30 epoch 이 지나도
> val loss 1.6 대에서 수렴하지 못했다 (정확도 0/5).
>
> **수정 후(`warmup_factor` + `lr=5e-4`)**: warmup 400 step 후 `5e-4` 로 안정 유지.
> 30 epoch 만에 val loss **0.01** 수준까지 내려가며 테스트 4/5 통과.
>
> **언제 논문 공식이 유리한가?**
> - 학습 step 이 매우 많을 때 (수만 step+) → 후반 decay 효과가 의미 있음
> - 거대 모델 (d_model ≥ 512) → 피크 LR 자동 스케일링이 적절
>
> **언제 단순 warmup 이 유리한가?**
> - 학습 step 이 적을 때 (< 수천 step) ← **이 실습이 해당**
> - 소형 모델 → 피크 LR 을 `lr=5e-4` 한 줄로 직관적으로 제어
>
> 본 실습은 9000/128 × 30 ≈ 2100 step 이라 decay 구간이 실질적 이득 없이
> 초반 피크만 독이 되었던 셈이다.

```python

# 학습
EPOCHS  = 30
step    = 0
history = {"train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    # ── 학습 ────────────────────────────────────
    model.train()
    total_loss = 0

    for src, tgt_in, tgt_out in train_loader:
        src     = src.to(device)
        tgt_in  = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        optimizer.zero_grad()

        # Forward
        output = model(src, tgt_in, pad_idx=PAD_TOKEN)
        # output: (batch, tgt_len, vocab_size)
        # tgt_out: (batch, tgt_len)

        # 손실 계산 — 차원 맞추기
        output_flat = output.view(-1, VOCAB_SIZE)    # (batch*tgt_len, vocab_size)
        target_flat = tgt_out.view(-1)               # (batch*tgt_len,)
        loss = criterion(output_flat, target_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # ── 검증 ────────────────────────────────────
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in val_loader:
            src     = src.to(device)
            tgt_in  = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            output  = model(src, tgt_in, pad_idx=PAD_TOKEN)
            output_flat = output.view(-1, VOCAB_SIZE)
            target_flat = tgt_out.view(-1)
            val_loss += criterion(output_flat, target_flat).item()

    avg_val_loss = val_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.7f}")

# 학습 곡선
plt.figure(figsize=(10, 4))
plt.plot(history["train_loss"], label="훈련 손실")
plt.plot(history["val_loss"],   label="검증 손실", linestyle="--")
plt.title("Transformer 학습 손실 변화")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)
plt.show()
```


### 7-5. 예측 함수 (Greedy Decoding)

```python
def predict(model, src_str, max_len=TGT_MAX_LEN):
    """
    Greedy Decoding:
    매 시점마다 가장 높은 확률의 토큰을 선택하며 순차적으로 생성
    """
    model.eval()

    # 입력 토큰화 및 패딩
    src = encode(src_str)
    src = src + [PAD_TOKEN] * (SRC_MAX_LEN - len(src))
    src = torch.tensor([src[:SRC_MAX_LEN]], dtype=torch.long).to(device)

    with torch.no_grad():
        # Encoder 1회 실행
        src_mask   = model.make_src_mask(src)
        enc_output = model.encoder(src, src_mask)

        # Decoder 순차 생성 — <sos>부터 시작
        tgt_tokens = [SOS_TOKEN]

        for _ in range(max_len):
            tgt = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            tgt_mask   = model.make_tgt_mask(tgt)

            dec_output = model.decoder(tgt, enc_output, src_mask, tgt_mask)
            output     = model.fc_out(dec_output)  # (1, cur_len, vocab_size)

            # 마지막 시점의 예측 토큰
            next_token = output[:, -1, :].argmax(dim=-1).item()
            tgt_tokens.append(next_token)

            if next_token == EOS_TOKEN:
                break

    return decode(tgt_tokens)


# 테스트
test_cases = [
    ("123+456", "579"),
    ("999+1",   "1000"),
    ("42+58",   "100"),
    ("0+0",     "0"),
    ("300+200", "500"),
]

print("=" * 40)
print(f"{'입력':12s} {'정답':8s} {'예측':8s} {'결과'}")
print("=" * 40)
for src_str, answer in test_cases:
    pred = predict(model, src_str)
    correct = "✅" if pred == answer else "❌"
    print(f"{src_str:12s} {answer:8s} {pred:8s} {correct}")
print("=" * 40)
```

"""
사용 장치: cuda | vocab=14
샘플: [('447+263', '710'), ('266+34', '300'), ('0+169', '169')]
파라미터 수: 1,394,446
Epoch  1/30 | Train 2.4934 | Val 1.9703 | LR 0.000090
Epoch  2/30 | Train 1.9247 | Val 1.7092 | LR 0.000179
....
Epoch 28/30 | Train 0.0619 | Val 0.0168 | LR 0.000500
Epoch 30/30 | Train 0.0428 | Val 0.0124 | LR 0.000500

────────────────────────────────────
 입력           정답       예측       결과
────────────────────────────────────
 123+456      579      579      OK 
 999+1        1000     110      X 
 42+58        100      100      OK 
 0+0          0        00       X 
 300+200      500      500      OK 
 77+88        165      165      OK 
 ────────────────────────────────────
 정확도: 4/6
"""

> 📌 **999+1, 0+0 이 실패하는 이유**
> - `999+1`: 학습 데이터가 `max_val=500` 범위라 `999` 는 분포 밖 — 모델이 본 적 없는 입력
> - `0+0`: 1-digit 답(`"0"`)이 학습 데이터에 드물어 모델이 `"00"` 처럼 2자리로 예측

***

## 8. nn.Transformer — PyTorch 내장 모듈 활용

### 8-1. 직접 구현 vs nn.Transformer

```python
# 직접 구현 (6번) — 내부 구조를 완전히 이해하고 제어
model = Transformer(...)

# nn.Transformer 사용 — 검증된 최적화 구현, 코드 간결
import torch.nn as nn
transformer = nn.Transformer(
    d_model          = 512,
    nhead            = 8,
    num_encoder_layers = 6,
    num_decoder_layers = 6,
    dim_feedforward  = 2048,
    dropout          = 0.1,
    batch_first      = True   # (batch, seq, feature) 순서
)
```


### 8-2. nn.Transformer로 덧셈 실습 재구현

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3,
                 d_ff=512, max_seq_len=20, dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc       = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer = nn.Transformer(
            d_model             = d_model,
            nhead               = nhead,
            num_encoder_layers  = num_encoder_layers,
            num_decoder_layers  = num_decoder_layers,
            dim_feedforward     = d_ff,
            dropout             = dropout,
            batch_first         = True   # ← 반드시 True 설정 권장
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_pad_mask(self, seq, pad_idx=0):
        """패딩 위치 마스크 — True인 위치는 무시"""
        return (seq == pad_idx)  # (batch, seq_len)

    def make_lookahead_mask(self, sz):
        """Look-ahead 마스크 — 미래 토큰 차단"""
        # nn.Transformer 에서 True = 차단
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)

    def forward(self, src, tgt, pad_idx=0):
        src_key_padding_mask = self.make_pad_mask(src, pad_idx)
        tgt_key_padding_mask = self.make_pad_mask(tgt, pad_idx)
        tgt_mask             = self.make_lookahead_mask(tgt.size(1))

        src_emb = self.pos_enc(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.pos_enc(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )

        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask             = tgt_mask,
            src_key_padding_mask = src_key_padding_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = src_key_padding_mask  # Cross Attention 패딩 마스크
        )

        return self.fc_out(output)  # (batch, tgt_len, vocab_size)


# 생성 및 테스트
model_builtin = TransformerModel(vocab_size=VOCAB_SIZE).to(device)
total = sum(p.numel() for p in model_builtin.parameters())
print(f"파라미터 수: {total:,}")

# 동작 확인
src_test = torch.randint(1, VOCAB_SIZE, (2, SRC_MAX_LEN)).to(device)
tgt_test = torch.randint(1, VOCAB_SIZE, (2, TGT_MAX_LEN)).to(device)
out_test = model_builtin(src_test, tgt_test)
print(f"출력 shape: {out_test.shape}")  # (2, TGT_MAX_LEN, VOCAB_SIZE)
```

> 💡 **nn.Transformer의 마스크 주의사항**
> - `tgt_mask` (Look-ahead): `True` = 차단 (float `-inf`로 치환됨)
> - `src/tgt_key_padding_mask`: `True` = 패딩 위치 무시
> - `batch_first=True` 설정 시 입력 shape = `(batch, seq, feature)` — 반드시 확인!

스크립트 전체(`transformer_addition_nn.py`)를 실행하면 §7 과 동일한 학습/디코딩 파이프라인을 `nn.Transformer` 로 돌립니다.

"""
사용 장치: cuda | vocab=14
파라미터 수: 1,394,446
UserWarning: The PyTorch API of nested tensors is in prototype stage ...  ← nn.Transformer 내부 경고(무시 가능)
Epoch  1/30 | Train 2.5432 | Val 1.8859
Epoch  2/30 | Train 1.8977 | Val 1.6943
....
Epoch 28/30 | Train 0.0538 | Val 0.0037
Epoch 30/30 | Train 0.0441 | Val 0.0081

────────────────────────────────────
 입력           정답       예측       결과
────────────────────────────────────
 123+456      579      579      OK 
 999+1        1000     110      X 
 42+58        100      100      OK 
 0+0          0        00       X 
 300+200      500      500      OK 
 77+88        165      165      OK 
 ────────────────────────────────────
 정확도: 4/6
"""

> 📊 **§7(직접 구현) vs §8(nn.Transformer) 비교**
> - 최종 val loss: §7 `0.0124` vs §8 `0.0081` — 내장 구현이 약간 더 안정적
> - 동일한 분포 밖/edge-case 에서 실패 (`999+1`, `0+0`)
> - 학습 시간: 거의 동일 (둘 다 동일 하이퍼파라미터)

***

## 9. Keras로 Transformer 구현하기

### 9-1. MultiHeadAttention 레이어

Keras 2.4+ 부터 `tf.keras.layers.MultiHeadAttention`이 내장되어 있습니다.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# MultiHeadAttention 기본 사용법
mha_layer = layers.MultiHeadAttention(
    num_heads  = 4,
    key_dim    = 64,    # 각 Head의 Q/K 차원
    value_dim  = 64,    # 각 Head의 V 차원
    dropout    = 0.1
)

# 사용 예
x     = tf.random.normal((2, 10, 256))  # (batch, seq_len, d_model)
out, weights = mha_layer(
    query  = x,
    key    = x,
    value  = x,
    return_attention_scores = True
)
print(f"출력 shape: {out.shape}")       # (2, 10, 256)
print(f"가중치 shape: {weights.shape}") # (2, 4, 10, 10)
```


### 9-2. TransformerBlock 클래스

```python
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.att  = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim   = d_model // num_heads,
            dropout   = dropout
        )
        self.ffn  = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout)
        self.drop2   = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-Attention + Add & Norm
        attn_out = self.att(x, x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))

        # FFN + Add & Norm
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x
```


### 9-3. 텍스트 분류 실습 — IMDB 감성 분석

```python
VOCAB_SIZE_IMDB = 20000
MAX_LEN_IMDB    = 200
D_MODEL         = 128
NUM_HEADS       = 4
D_FF            = 512
NUM_LAYERS      = 2
DROPOUT         = 0.1

# 데이터 불러오기 (2편과 동일)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE_IMDB)
X_train = pad_sequences(X_train, maxlen=MAX_LEN_IMDB,
                        padding="pre", truncating="pre")
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN_IMDB,
                        padding="pre", truncating="pre")


def build_transformer_classifier(vocab_size, maxlen, d_model,
                                  num_heads, d_ff, num_layers, dropout):
    inputs  = layers.Input(shape=(maxlen,))

    # Embedding + Positional Encoding (학습 가능한 위치 임베딩 사용)
    x = layers.Embedding(vocab_size, d_model)(inputs)
    # 위치 임베딩: 각 위치에 대해 학습 가능한 벡터
    positions = tf.range(start=0, limit=maxlen, delta=1)
    pos_emb   = layers.Embedding(maxlen, d_model)(positions)
    x = x + pos_emb
    x = layers.Dropout(dropout)(x)

    # Transformer 블록 N개 쌓기
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, d_ff, dropout)(x)

    # 분류기 헤드
    x = layers.GlobalAveragePooling1D()(x)  # 시퀀스 평균 → (batch, d_model)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


model_keras_tf = build_transformer_classifier(
    vocab_size = VOCAB_SIZE_IMDB,
    maxlen     = MAX_LEN_IMDB,
    d_model    = D_MODEL,
    num_heads  = NUM_HEADS,
    d_ff       = D_FF,
    num_layers = NUM_LAYERS,
    dropout    = DROPOUT
)

model_keras_tf.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)

model_keras_tf.summary()
```


### 9-4. 학습 및 LSTM과 성능 비교

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3,
    restore_best_weights=True, verbose=1
)

history_tf = model_keras_tf.fit(
    X_train, y_train,
    epochs          = 10,
    batch_size      = 128,
    validation_split= 0.2,
    callbacks       = [early_stop],
    verbose         = 1
)

test_loss, test_acc = model_keras_tf.evaluate(X_test, y_test, verbose=0)
print(f"\nTransformer 테스트 정확도: {test_acc:.4f}")

# 학습 곡선
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(history_tf.history["accuracy"],     label="훈련 정확도")
ax1.plot(history_tf.history["val_accuracy"], label="검증 정확도", linestyle="--")
ax1.set_title("Transformer 정확도")
ax1.legend(); ax1.grid(True)

ax2.plot(history_tf.history["loss"],     label="훈련 손실")
ax2.plot(history_tf.history["val_loss"], label="검증 손실", linestyle="--")
ax2.set_title("Transformer 손실")
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.show()

# LSTM(2편)과 비교
print("\n2편 LSTM vs 3편 Transformer 성능 비교 (IMDB 감성 분석):")
print(f"  Bi-LSTM     정확도: ~87%")
print(f"  Transformer 정확도: {test_acc*100:.1f}%")
```
"""
(TF 시작 시 oneDNN / GPU 경고 메시지 생략)
훈련: (25000, 200) / 테스트: (25000, 200)

WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.
Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.

Model: "functional_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 200)                 │               0 │
│ embedding (Embedding)                │ (None, 200, 128)            │       2,560,000 │
│ add (Add)                            │ (None, 200, 128)            │               0 │
│ dropout (Dropout)                    │ (None, 200, 128)            │               0 │
│ transformer_block (TransformerBlock) │ (None, 200, 128)            │         198,272 │
│ transformer_block_1                  │ (None, 200, 128)            │         198,272 │
│ global_average_pooling1d             │ (None, 128)                 │               0 │
│ dense_4 (Dense)                      │ (None, 64)                  │           8,256 │
│ dropout_9 (Dropout)                  │ (None, 64)                  │               0 │
│ dense_5 (Dense)                      │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,964,865 (11.31 MB)
 Trainable params: 2,964,865 (11.31 MB)

Epoch 1/10  157/157  79s 481ms/step - acc: 0.6549 - loss: 0.6097 - val_acc: 0.7708 - val_loss: 0.4695
Epoch 2/10  157/157  77s 493ms/step - acc: 0.8661 - loss: 0.3157 - val_acc: 0.8696 - val_loss: 0.3102
Epoch 3/10  157/157  80s 507ms/step - acc: 0.9105 - loss: 0.2275 - val_acc: 0.8762 - val_loss: 0.3117
Epoch 4/10  157/157  79s 501ms/step - acc: 0.9375 - loss: 0.1685 - val_acc: 0.8748 - val_loss: 0.3441
Epoch 5/10  157/157           —   - acc: 0.9545 - loss: 0.1262 - val_acc: 0.8808 - val_loss: 0.3502   ← best
Epoch 6/10  157/157  78s 497ms/step - acc: 0.9685 - loss: 0.0924 - val_acc: 0.8760 - val_loss: 0.4151
Epoch 7/10  157/157  78s 494ms/step - acc: 0.9801 - loss: 0.0613 - val_acc: 0.8672 - val_loss: 0.5096
Epoch 8/10  157/157  78s 498ms/step - acc: 0.9854 - loss: 0.0453 - val_acc: 0.8642 - val_loss: 0.5682
Epoch 8: early stopping
Restoring model weights from the end of the best epoch: 5.

Transformer 테스트 정확도: 0.8619

2편 LSTM vs 3편 Transformer 성능 비교 (IMDB 감성 분석):
  Bi-LSTM     정확도: ~87%
  Transformer 정확도: 86.2%
"""

> 📊 **결과 해석**
> - 훈련 정확도는 epoch 8 에 98.5% 까지 올라가지만, 검증 정확도는 **epoch 5 (88.1%) 가 피크** → 이후 과적합
> - `EarlyStopping` 이 epoch 8 에서 중단 + epoch 5 가중치로 복원 → 최종 테스트 정확도 **86.2%**
> - Bi-LSTM(~87%) 과 거의 동일한 성능 — IMDB 처럼 비교적 짧은 시퀀스(200 토큰)에서는 Transformer 의 병렬성/장거리 의존성 이점이 정확도로 크게 드러나지 않음
> - **Transformer 의 진가는 긴 시퀀스 · 대규모 데이터 · 전이학습** 에서 나타납니다 (BERT, GPT 등)

> ⚠️ **Windows 네이티브 TF 는 GPU 미지원**
> TF ≥ 2.11 은 Windows 네이티브에서 GPU 를 쓸 수 없어 CPU 로 학습됩니다 (에폭당 ~78초 · 총 ~10분). GPU 로 돌리려면:
> - **WSL2 + CUDA** 전환 (권장)
> - 또는 `tensorflow-directml-plugin` 설치


***

## 10. 마무리

### 10-1. 오늘 배운 것 한눈에 정리

| 개념 | 핵심 내용 |
| :-- | :-- |
| Attention | Q·K 유사도로 중요 위치에 가중치 부여, 가중합으로 V 집계 |
| Self-Attention | Q=K=V=입력, 문장 내 단어끼리 관계 직접 파악 |
| Multi-Head Attention | 여러 관점에서 동시에 Attention → 다양한 관계 포착 |
| Positional Encoding | sin/cos으로 위치 정보 부여 (병렬 처리의 단점 보완) |
| Encoder Block | Self-Attention + Add\&Norm + FFN + Add\&Norm |
| Decoder Block | Masked Self-Attention + Cross Attention + FFN |
| Look-ahead Mask | 미래 토큰 차단 (학습 시 답 보는 것 방지) |
| Cross Attention | Q=Decoder, K·V=Encoder 출력 (번역 시 원문 참조) |
| Greedy Decoding | 매 시점 최고 확률 토큰 선택하여 순차 생성 |

***