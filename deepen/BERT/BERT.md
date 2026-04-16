[미검증]
## 0. 시리즈

| 심화편 | 제목 | 난이도 | 역할 |
| :-- | :-- | :-- | :-- |
| [심화 1편](https://duckport.pages.dev/posts/CNN_Deep) | CNN — 이미지 분류 완벽 정리 | ⭐⭐ | 이미지 처리 |
| [심화 2편](https://duckport.pages.dev/posts/RNN) | RNN / LSTM — 시계열·텍스트 처리 | ⭐⭐⭐ | 순서 데이터 |
| [심화 3편](https://duckport.pages.dev/posts/Transformer) | Transformer 구조 완벽 정리 | ⭐⭐⭐⭐ | 현대 AI 핵심 |
| [**심화 4편**](https://duckport.pages.dev/posts/BERT)⬅️ | BERT / GPT 원리와 활용 | ⭐⭐⭐⭐ | 언어 모델 |
| [심화 5편](https://duckport.pages.dev/posts/Finetuning_Deep) | HuggingFace로 LLM Fine-tuning | ⭐⭐⭐⭐⭐ | LLM 실전 |


***

## 1. BERT란?

### 1-1. BERT 등장 배경

2018년 Google AI가 발표한 **BERT(Bidirectional Encoder Representations from Transformers)** 는 NLP 역사상 가장 큰 전환점입니다. 발표 직후 11개 NLP 벤치마크에서 동시에 최고 성능을 기록했습니다.

```
이전까지:
  태스크마다 모델을 처음부터 학습 → 데이터 많이 필요, 시간 오래 걸림

BERT 이후:
  대규모 텍스트로 사전 학습된 모델을 가져와서
  내 태스크에 맞게 조금만 추가 학습 (Fine-tuning)
  → 적은 데이터, 짧은 시간, 높은 성능
```


### 1-2. 전이 학습(Transfer Learning)이란?

```
[사전 학습 Pre-training]
  대규모 코퍼스 (위키피디아, BookCorpus 등 수십 GB)
  → 모델이 언어의 일반적인 패턴을 학습

         ↓  학습된 가중치 (pretrained weights)

[미세 조정 Fine-tuning]
  내 태스크의 소규모 데이터로 추가 학습
  (감성 분석, 질의응답, 개체명 인식 등)
```

전이 학습을 비유하면:

- 사전 학습 = 수년간 영어 공부로 언어 구조와 의미를 완전히 익힌 상태
- 파인튜닝 = 그 지식을 바탕으로 법률 문서 분류를 며칠 만에 익히는 것


### 1-3. 양방향(Bidirectional)이란?

```
GPT (단방향 → 왼쪽에서 오른쪽만 봄):
"나는 [은행]에 ___"
→ "나는", "은행에" 만 보고 다음 단어 예측

BERT (양방향 → 앞뒤 모두 봄):
"나는 [MASK]에 돈을 입금했다"
→ "나는" + "돈을 입금했다" 까지 보고 MASK 예측
→ "은행"이라고 맞힐 수 있음 ✅
```

앞뒤 문맥을 동시에 보기 때문에 **문장 이해 태스크** 에 훨씬 강합니다.

### 1-4. BERT의 사전 학습 방법 2가지

**① MLM (Masked Language Model) — 빈칸 맞히기**

```
원본: "나는 오늘 학교에 갔다"
입력: "나는 [MASK] 학교에 갔다"
목표: [MASK] 위치에서 "오늘"을 예측

학습 규칙:
- 전체 토큰의 15%를 무작위 선택
- 선택된 토큰 중:
  80% → [MASK]로 교체
  10% → 랜덤 다른 단어로 교체
  10% → 그대로 유지 (모델이 어떤 위치든 의심하도록)
```

**② NSP (Next Sentence Prediction) — 다음 문장 예측**

```
긍정 예시 (IsNext):
  문장A: "강아지가 공원에서 뛰어놀았다."
  문장B: "강아지는 매우 행복해 보였다."  → 실제 다음 문장 ✅

부정 예시 (NotNext):
  문장A: "강아지가 공원에서 뛰어놀았다."
  문장B: "주식 시장이 오늘 급락했다."   → 랜덤 문장 ❌
```


### 1-5. BERT의 입력 구조

```
입력 문장: "나는 밥을 먹었다"

토큰:  [CLS] 나는  밥을  먹었다  [SEP]
         ↑                           ↑
   문장 시작 특수 토큰        문장 끝 특수 토큰

최종 입력 = Token Embedding
           + Segment Embedding (문장A=0 / 문장B=1)
           + Position Embedding (학습 가능한 위치 벡터)

[CLS] 토큰:
→ 문장 전체의 요약 정보가 담기도록 학습됨
→ 분류 태스크에서 [CLS] 출력벡터를 Dense 레이어에 연결
```


### 1-6. BERT-base vs BERT-large vs ModernBERT (2026 기준)

2024년 말 등장한 **ModernBERT** 는 기존 BERT의 구조를 대폭 개선한 경량화 버전입니다.


| 구분 | BERT-base | BERT-large | ModernBERT |
| :-- | :-- | :-- | :-- |
| Encoder 레이어 | 12 | 24 | 선택적 조정 가능 |
| 은닉 차원 | 768 | 1024 | 가변 |
| 파라미터 수 | 1.1억 | 3.4억 | 경량화 (선택 가능) |
| 학습 속도 | 보통 | 느림 | **최대 2배 빠름** |
| 컨텍스트 길이 | 512 토큰 | 512 토큰 | **8K 이상** |
| 위치 임베딩 | 학습 가능 | 학습 가능 | **RoPE** (회전 위치 임베딩) |
| 권장 상황 | 일반 실습 | 최고 성능 필요 | **2026년 실무 권장** |

> 💡 **RoPE (Rotary Position Embedding)** — 3편 Transformer에서 배운 sin/cos Positional Encoding의 개선 버전입니다. 위치 정보를 회전 행렬로 표현해 더 긴 컨텍스트를 효율적으로 처리합니다.

***

## 2. GPT란?

### 2-1. GPT 등장 배경

**GPT(Generative Pretrained Transformer)** 는 OpenAI가 2018년 발표한 모델입니다. BERT가 "문장을 이해하는 AI"라면 GPT는 **"문장을 생성하는 AI"** 입니다.

```
BERT : Transformer의 Encoder 구조 활용 → 이해(Understanding)
GPT  : Transformer의 Decoder 구조 활용 → 생성(Generation)
```


### 2-2. 단방향(Autoregressive)이란?

GPT는 항상 **왼쪽 → 오른쪽** 방향으로만 이전 토큰을 보며 다음 토큰을 예측합니다.

```
"안녕" → "안녕하" → "안녕하세" → "안녕하세요" → ...
한 토큰씩 이어 붙이며 자연스러운 문장 생성
```


### 2-3. GPT의 사전 학습 — CLM (Causal Language Model)

```
문장: "나는 오늘 밥을 먹었다"

학습 방식 (다음 단어 예측):
  "나는"           → 예측: "오늘" ✅
  "나는 오늘"      → 예측: "밥을" ✅
  "나는 오늘 밥을" → 예측: "먹었다" ✅

→ 대규모 텍스트의 모든 위치에서 이 과정을 반복
→ 언어의 통계적 패턴을 완전히 학습
```


### 2-4. GPT 발전 흐름 — 2026년 최신 기준

| 버전 | 연도 | 파라미터 수 | 핵심 특징 |
| :-- | :-- | :-- | :-- |
| GPT-1 | 2018 | 1.1억 | Transformer Decoder 기반 최초 GPT |
| GPT-2 | 2019 | 15억 | "너무 강력해서 공개 안 함" 이슈 |
| GPT-3 | 2020 | 1,750억 | Few-shot 학습 능력 폭발적 향상 |
| GPT-4 | 2023 | 비공개 | 멀티모달(텍스트+이미지) |
| GPT-5 | 2025 | 비공개 | 내장 추론(Thinking) 기능, 전문가급 지능 |
| **GPT-5.4** | **2026.03** | **비공개** | **1M 토큰 컨텍스트, 컴퓨터 직접 사용, 추론 강도 5단계 조절** |

**GPT-5.4 주요 신기능 (2026년 3월 기준)**

```
① 추론 강도 조절 (Configurable Reasoning Effort)
   none / low / medium / high / xhigh 5단계
   → 간단한 쿼리: low (비용 절감)
   → 복잡한 코딩: xhigh (정확도 극대화)

② 컨텍스트 창 1M 토큰
   기존 GPT-5.2 대비 2.5배 확장
   → 소설 한 편, 대형 코드베이스 전체를 한 번에 처리

③ Computer Use (컴퓨터 직접 사용)
   마우스·키보드를 직접 조작하는 에이전트 기능
   OSWorld 벤치마크 75% 달성

④ 토큰 효율 개선
   복잡한 작업에서 GPT-5.3 대비 47% 적은 토큰 사용
```


### 2-5. Few-shot / Zero-shot 학습이란?

GPT-3부터는 Fine-tuning 없이도 프롬프트만으로 다양한 태스크를 수행합니다.

```
Zero-shot (예시 없음):
  "다음 문장을 영어로 번역해줘: 나는 학교에 간다"
  → I go to school.

One-shot (예시 1개):
  "번역 예시: 안녕하세요 → Hello
   번역해줘: 나는 학교에 간다"
  → I go to school.

Few-shot (예시 여러 개):
  예시 여러 개를 주면 더 정확하게 수행
```


***

## 3. BERT vs GPT 한눈에 비교

| 비교 항목 | BERT | GPT |
| :-- | :-- | :-- |
| 개발사 | Google AI | OpenAI |
| 기반 구조 | Transformer **Encoder** | Transformer **Decoder** |
| 문맥 방향 | **양방향** (앞+뒤 모두) | **단방향** (왼→오른쪽) |
| 사전 학습 | MLM + NSP | CLM (다음 단어 예측) |
| 강점 | 분류, 질의응답, NER | 텍스트 생성, 요약, 번역 |
| 2026년 현재 | ModernBERT로 경량화 진화 | GPT-5.4까지 발전 |
| Fine-tuning | 태스크별 헤드 추가 | 프롬프트 엔지니어링 |


***

## 4. 2026년 주요 LLM 현황

> 💡 실무에서 GPT나 BERT를 직접 사전 학습하는 경우는 없습니다. 이미 잘 훈련된 모델을 가져다 쓰는 것이 기본입니다. 2026년 현재 어떤 모델들이 있는지 파악해두면 5편 Fine-tuning 실습에 큰 도움이 됩니다.

### 4-1. 2026년 3월 기준 주요 프론티어 모델

| 모델 | 개발사 | 핵심 강점 | 특이사항 |
| :-- | :-- | :-- | :-- |
| **GPT-5.4** | OpenAI | 범용성, AIME 수학 100%, 1M 컨텍스트 | Computer Use |
| **Claude Opus 4.6** | Anthropic | 코딩 최강(SWE-bench 80.8%), 1M 컨텍스트 | 14.5시간 에이전트 |
| **Claude Sonnet 4.6** | Anthropic | Opus급 코딩을 1/5 가격에 | 가성비 최강 |
| **Gemini 3.1 Pro** | Google | 논리 추론(ARC-AGI-2 77.1%), 멀티모달 | 검색 연동 강점 |
| **Qwen 3.5-122B** | Alibaba | 오픈소스 최강급 | 상업적 무료 사용 가능 |
| **DeepSeek v4** | DeepSeek | 추론 특화, 저비용 | 중국 오픈소스 |

### 4-2. 주요 벤치마크 비교 (2026년 3월)

| 벤치마크 | GPT-5.4 | Claude Opus 4.6 | Gemini 3.1 Pro | 평가 내용 |
| :-- | :-- | :-- | :-- | :-- |
| AIME 2025 | **100%** | — | — | 수학 추론 |
| SWE-bench Verified | 77.2% | **80.8%** | — | 실전 코딩 |
| ARC-AGI-2 | — | — | **77.1%** | 논리 패턴 추론 |
| GPQA | — | — | **94.3%** | PhD급 추론 |

### 4-3. HuggingFace 2026년 현황

```
2026년 Spring 보고서 기준:
- 중국 모델 다운로드 점유율: 41% (Qwen, DeepSeek 시리즈 주도)
- 초소형·고효율 모델 트렌드 가속 (LiquidAI LFM2.5-350M 등)
- Transformers 라이브러리: 지속 업데이트 중
- 오픈소스 모델이 상업 모델과 성능 격차 빠르게 좁히는 중
```


***

## 5. HuggingFace Transformers 라이브러리

### 5-1. HuggingFace란?

HuggingFace는 수만 개의 사전 학습 모델과 데이터셋을 무료로 제공하는 플랫폼입니다. `transformers` 라이브러리 하나로 BERT, GPT, Qwen, LLaMA 등 거의 모든 최신 모델을 바로 불러와 사용할 수 있습니다.

```bash
# 설치
pip install transformers datasets torch
pip install accelerate  # 학습 속도 향상
```


### 5-2. pipeline — 코드 5줄로 바로 사용

```python
from transformers import pipeline

# ── 감성 분석 ──────────────────────────────────────────────────
classifier = pipeline("sentiment-analysis")

results = classifier([
    "This movie was absolutely fantastic!",
    "I hated every minute of this film."
])
for r in results:
    print(f"  레이블: {r['label']:10s}  점수: {r['score']:.4f}")
# 출력:
#   레이블: POSITIVE     점수: 0.9998
#   레이블: NEGATIVE     점수: 0.9994


# ── 텍스트 생성 ────────────────────────────────────────────────
generator = pipeline("text-generation", model="gpt2")

output = generator(
    "Artificial intelligence will",
    max_new_tokens       = 50,
    num_return_sequences = 2,
    do_sample            = True,
    temperature          = 0.8
)
for i, o in enumerate(output):
    print(f"  생성 {i+1}: {o['generated_text']}")


# ── 빈칸 채우기 (BERT 스타일) ───────────────────────────────────
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

result = fill_mask("The capital of France is [MASK].")
for r in result[:3]:
    print(f"  {r['token_str']:15s} 확률: {r['score']:.4f}")
# 출력:
#   paris           확률: 0.9971
#   london          확률: 0.0013
#   berlin          확률: 0.0007


# ── 질의응답 ────────────────────────────────────────────────────
qa = pipeline("question-answering")

context = ""
Hugging Face is a company that develops tools for building machine learning
applications. It was founded in 2016 and is headquartered in New York City.
""
result = qa(question="When was Hugging Face founded?", context=context)
print(f"  답변: {result['answer']}")   # 2016
print(f"  점수: {result['score']:.4f}")
```


### 5-3. AutoModel / AutoTokenizer

```python
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModel.from_pretrained(model_name)

print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
# 109,482,240 (약 1억 1천만)
```
***

### 5-4. 토크나이저(Tokenizer) 완벽 이해

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Hello, I am learning about BERT!"

# 기본 토큰화
tokens = tokenizer.tokenize(text)
print(f"토큰:   {tokens}")
# ['hello', ',', 'i', 'am', 'learning', 'about', 'bert', '!']

# 인코딩 (숫자로 변환)
encoding = tokenizer(
    text,
    max_length      = 20,
    padding         = "max_length",  # 최대 길이로 패딩
    truncation      = True,
    return_tensors  = "pt"           # PyTorch tensor 반환
)

print(f"input_ids:      {encoding['input_ids']}")
print(f"attention_mask: {encoding['attention_mask']}")
print(f"token_type_ids: {encoding['token_type_ids']}")
```

```
input_ids:
  [101, 7592, 1010, 1045, ..., 102, 0, 0, 0]
   ↑[CLS]                    ↑[SEP] ↑ 패딩(0)

attention_mask:
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
   ↑실제 토큰                   ↑ 패딩 → 무시

token_type_ids:
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ↑ 문장A = 0 / 문장B = 1 (NSP 태스크용)
```

```python
# 두 문장 동시 인코딩 (문장 쌍 태스크)
encoding_pair = tokenizer(
    "나는 밥을 먹었다", "배가 너무 불렀다",
    max_length     = 30,
    padding        = "max_length",
    truncation     = True,
    return_tensors = "pt"
)
print(f"token_type_ids: {encoding_pair['token_type_ids']}")
# [0,0,...,0, 1,1,...,1, 0,0]
#  ↑문장A      ↑문장B     ↑패딩

# 디코딩 (숫자 → 문자열로 복원)
decoded = tokenizer.decode(encoding['input_ids'][0])
print(f"디코딩: {decoded}")
# [CLS] hello, i am learning about bert! [SEP] [PAD] ...
```

**WordPiece vs BPE (서브워드 토큰화)**

```
# BERT는 WordPiece 방식
"unbelievable" → ["un", "##believ", "##able"]
                              ↑ ##는 앞 토큰과 이어지는 서브워드
"playing"      → ["play", "##ing"]

# GPT는 BPE(Byte Pair Encoding) 방식
"lower"    → ["low", "er"]
"lowest"   → ["low", "est"]

공통 장점:
- 사전에 없는 단어(OOV)도 서브워드로 분해하여 처리 가능
- 어휘 사전을 작게 유지하면서 다양한 단어 표현 가능
```


***

## 6. BERT 실습 — 감성 분석 (문장 분류)

### 6-1. 준비 및 데이터 불러오기

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

# HuggingFace datasets로 IMDB 불러오기
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test:  Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 실습용으로 일부만 사용 (전체 학습 시 시간 매우 오래 걸림)
# ⚠️ IMDB는 label 순으로 정렬되어 있어 앞에서 그대로 잘라쓰면
#    전부 부정(label=0) 샘플만 들어옴 → 반드시 shuffle 후 slicing
train_shuffled = dataset["train"].shuffle(seed=42)
test_shuffled  = dataset["test"].shuffle(seed=42)

train_texts  = train_shuffled["text"][:2000]
train_labels = train_shuffled["label"][:2000]
test_texts   = test_shuffled["text"][:500]
test_labels  = test_shuffled["label"][:500]

# 검증 세트 분리 (9:1)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42
)

print(f"훈련: {len(train_texts)}개 / 검증: {len(val_texts)}개 / 테스트: {len(test_texts)}개")
print(f"샘플: {train_texts[0][:80]}...")
print(f"레이블: {'긍정' if train_labels[0] == 1 else '부정'}")
```

"""
훈련: 1800개 / 검증: 200개 / 테스트: 500개
샘플: Nick Cage is Randall Raines, a retired car thief who is forced out of retirement...
레이블: 긍정
"""

### 6-2. 토크나이저 적용 및 Dataset 클래스

```python
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128  # 메모리 부족 시 64로 줄이기

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "token_type_ids" : encoding["token_type_ids"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
        }


BATCH_SIZE = 16  # GPU 메모리에 따라 8로 줄이기 가능

train_ds = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_ds   = IMDBDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
test_ds  = IMDBDataset(test_texts,  test_labels,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 샘플 확인
sample = train_ds[0]
print(f"input_ids shape:      {sample['input_ids'].shape}")       # (128,)
print(f"attention_mask shape: {sample['attention_mask'].shape}")  # (128,)
print(f"label: {sample['label']}")
```

"""
input_ids shape:      torch.Size([128])
attention_mask shape: torch.Size([128])
label: 1
"""

### 6-3. BertForSequenceClassification 모델

```python
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = 2   # 긍정(1) / 부정(0)
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"전체 파라미터: {total_params:,}")
print(f"학습 파라미터: {train_params:,}")
```

"""
전체 파라미터: 109,483,778
학습 파라미터: 109,483,778
"""

```
BertForSequenceClassification 구조:

  BertModel (사전 학습 완료 — 건드리지 않아도 됨)
    ├── Embeddings
    └── Encoder × 12 (각 BertLayer: Self-Attention + FFN)
    ↓
  [CLS] 토큰의 출력 벡터 (768차원)
    ↓
  Dropout(0.1)
    ↓
  Linear(768 → 2)  ← 이 부분만 새로 학습됨
    ↓
  [부정 확률, 긍정 확률]
```


### 6-4. 옵티마이저 및 스케줄러 설정

```python
EPOCHS       = 3       # BERT Fine-tuning은 2~4 epoch면 충분
LR           = 2e-5    # BERT Fine-tuning 권장 학습률
WARMUP_RATIO = 0.1     # 전체 스텝의 10%를 warm-up

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = LR,
    weight_decay = 0.01   # L2 정규화
)

# Linear Warm-up → Linear Decay
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps
)

print(f"총 학습 스텝: {total_steps}")
print(f"Warm-up 스텝: {warmup_steps}")
```

"""
총 학습 스텝: 339
Warm-up 스텝: 33
"""

> 💡 **AdamW** — Adam 옵티마이저에 Weight Decay를 올바르게 적용한 버전입니다. BERT Fine-tuning의 표준 옵티마이저입니다.
> 💡 **Warm-up** — 처음 몇 스텝 동안 학습률을 0에서 LR까지 서서히 높입니다. 사전 학습된 가중치가 초반에 급격히 망가지는 것을 방지합니다.

### 6-5. 학습 루프

```python
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        # HuggingFace 모델은 labels 전달 시 loss 자동 계산
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels         = labels
        )

        loss   = outputs.loss
        logits = outputs.logits  # (batch, num_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                labels         = labels
            )

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


# 학습 실행
history = {"train_loss": [], "val_loss": [],
           "train_acc":  [], "val_acc":  []}
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_epoch(model, train_loader,
                                        optimizer, scheduler, device)
    val_loss,   val_acc   = eval_epoch(model, val_loader, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_bert_imdb.pth")
        print(f"  ✅ 최고 모델 저장 (Val Acc: {val_acc:.4f})")
```

"""
Epoch 1/3
───────────────────────────────────────
Train Loss: 0.5409 | Train Acc: 0.7161
Val   Loss: 0.3171   | Val   Acc: 0.8650
  ✅ 최고 모델 저장 (Val Acc: 0.8650)

Epoch 2/3
───────────────────────────────────────
Train Loss: 0.2700 | Train Acc: 0.9028
Val   Loss: 0.3295   | Val   Acc: 0.8550

Epoch 3/3
───────────────────────────────────────
Train Loss: 0.1432 | Train Acc: 0.9556
Val   Loss: 0.3404   | Val   Acc: 0.8750
  ✅ 최고 모델 저장 (Val Acc: 0.8750)
"""

### 6-6. 평가 및 예측

```python
# 최고 가중치 불러오기
model.load_state_dict(torch.load("best_bert_imdb.pth"))
test_loss, test_acc = eval_epoch(model, test_loader, device)
print(f"\n최종 테스트 정확도: {test_acc:.4f}")

# 학습 곡선 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="훈련 손실", marker="o")
ax1.plot(history["val_loss"],   label="검증 손실", marker="o", linestyle="--")
ax1.set_title("BERT Fine-tuning 손실")
ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)

ax2.plot(history["train_acc"], label="훈련 정확도", marker="o")
ax2.plot(history["val_acc"],   label="검증 정확도", marker="o", linestyle="--")
ax2.set_title("BERT Fine-tuning 정확도")
ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.show()

![Figure_1-1.png](/api/assets/8788cd54-7172-4a3e-a05d-afe973060082)

# 직접 문장 예측
def predict_sentiment(model, tokenizer, text, device):
    model.eval()
    encoding = tokenizer(
        text,
        max_length     = MAX_LEN,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids      = encoding["input_ids"].to(device),
            attention_mask = encoding["attention_mask"].to(device),
            token_type_ids = encoding["token_type_ids"].to(device)
        )
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    label = "긍정 😊" if probs.argmax().item() == 1 else "부정 😞"
    print(f"입력: {text[:60]}...")
    print(f"예측: {label}  (부정: {probs[0]:.4f} / 긍정: {probs[1]:.4f})")

predict_sentiment(model, tokenizer,
    "This movie was absolutely brilliant! The acting was superb.", device)
predict_sentiment(model, tokenizer,
    "Worst movie I've ever seen. Complete waste of time.", device)
```

"""
최종 테스트 정확도: 0.8600
입력: This movie was absolutely brilliant! The acting was superb....
예측: 긍정 😊  (부정: 0.0167 / 긍정: 0.9833)
입력: Worst movie I've ever seen. Complete waste of time....
예측: 부정 😞  (부정: 0.9822 / 긍정: 0.0178)
"""

***

## 7. GPT-2 실습 — 텍스트 생성

### 7-1. GPT-2 모델 불러오기

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"  # gpt2 / gpt2-medium / gpt2-large

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model_gpt2     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

# GPT-2는 pad_token이 기본적으로 없음 → 반드시 설정
tokenizer_gpt2.pad_token       = tokenizer_gpt2.eos_token
model_gpt2.config.pad_token_id = tokenizer_gpt2.eos_token_id

print(f"GPT-2 파라미터 수: {sum(p.numel() for p in model_gpt2.parameters()):,}")
# 117,000,000
```

> ⚠️ **GPT-2 pad_token 주의** — GPT-2는 기본적으로 `pad_token`이 없습니다. 반드시 `eos_token`을 `pad_token`으로 설정해야 배치 학습 시 오류가 발생하지 않습니다.

### 7-2. 생성 전략 비교

GPT 텍스트 생성 품질은 **어떤 전략으로 다음 토큰을 선택하느냐** 에 따라 크게 달라집니다.

```python
prompt    = "In the future, artificial intelligence will"
input_ids = tokenizer_gpt2.encode(prompt, return_tensors="pt").to(device)

model_gpt2.eval()

# ── 전략 1. Greedy Search ────────────────────────────────────────
# 매 시점 가장 높은 확률의 토큰만 선택 → 반복적이고 단조로움
greedy_out = model_gpt2.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample      = False
)
print("【Greedy Search】")
print(tokenizer_gpt2.decode(greedy_out[0], skip_special_tokens=True))


# ── 전략 2. Beam Search ─────────────────────────────────────────
# 상위 num_beams개 후보를 동시에 유지 → Greedy보다 품질 높음
beam_out = model_gpt2.generate(
    input_ids,
    max_new_tokens       = 50,
    num_beams            = 5,
    early_stopping       = True,
    no_repeat_ngram_size = 2   # 2-gram 반복 금지
)
print("【Beam Search (num_beams=5)】")
print(tokenizer_gpt2.decode(beam_out[0], skip_special_tokens=True))


# ── 전략 3. Top-k Sampling ──────────────────────────────────────
# 확률 상위 k개 중 랜덤 샘플링 → 다양하고 창의적
topk_out = model_gpt2.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample      = True,
    top_k          = 50,
    temperature    = 0.8   # <1: 보수적 / >1: 창의적
)
print("【Top-k Sampling (k=50, temp=0.8)】")
print(tokenizer_gpt2.decode(topk_out[0], skip_special_tokens=True))


# ── 전략 4. Top-p (Nucleus) Sampling ───────────────────────────
# 누적 확률이 p를 넘는 최소 토큰 집합에서 샘플링 → 현재 가장 권장
topp_out = model_gpt2.generate(
    input_ids,
    max_new_tokens = 50,
    do_sample      = True,
    top_p          = 0.92,
    temperature    = 0.8
)
print("【Top-p Sampling (p=0.92, temp=0.8)】")
print(tokenizer_gpt2.decode(topp_out[0], skip_special_tokens=True))
```

**전략별 특성 비교:**


| 전략 | 다양성 | 일관성 | 적합한 상황 |
| :-- | :-- | :-- | :-- |
| Greedy Search | ❌ 낮음 | ✅ 높음 | 번역, 요약 |
| Beam Search | △ 보통 | ✅ 높음 | 번역 + 반복 방지 |
| Top-k Sampling | ✅ 높음 | △ 보통 | 창작 글쓰기 |
| **Top-p Sampling** | ✅ 높음 | ✅ 높음 | **대부분의 생성 태스크 권장** |

### 7-3. temperature 파라미터 이해

```python
import numpy as np
import matplotlib.pyplot as plt

logits = np.array([3.0, 2.0, 1.0, 0.5, 0.1])

def softmax_with_temp(logits, temperature):
    scaled = logits / temperature
    exp    = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()

temps = [0.5, 1.0, 1.5, 2.0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, t in zip(axes, temps):
    probs = softmax_with_temp(logits, t)
    ax.bar(range(len(probs)), probs, color="steelblue")
    ax.set_title(f"temperature = {t}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("토큰 인덱스")
    ax.set_ylabel("확률")
    ax.grid(axis="y")

plt.suptitle("Temperature에 따른 확률 분포 변화\n"
             "(낮을수록 뾰족 → 보수적 / 높을수록 평평 → 창의적)")
plt.tight_layout()
plt.show()

# temperature < 1 : 높은 확률 토큰에 더 집중 → 안전하고 반복적
# temperature = 1 : 원래 확률 분포 그대로
# temperature > 1 : 분포 평평해짐 → 다양하지만 엉뚱한 텍스트 가능
```

![Figure_2-1.png](/api/assets/8177778e-1212-44cb-80ef-eb317ac66298)

### 7-4. 여러 문장 동시 생성

```python
def generate_texts(prompt, n=3, max_new_tokens=80,
                   top_p=0.92, temperature=0.9):
    input_ids = tokenizer_gpt2.encode(
        prompt, return_tensors="pt"
    ).to(device)

    outputs = model_gpt2.generate(
        input_ids,
        max_new_tokens       = max_new_tokens,
        do_sample            = True,
        top_p                = top_p,
        temperature          = temperature,
        num_return_sequences = n,
        no_repeat_ngram_size = 3
    )

    print(f"프롬프트: \"{prompt}\"\n")
    for i, out in enumerate(outputs):
        text = tokenizer_gpt2.decode(out, skip_special_tokens=True)
        print(f"  [{i+1}] {text}\n")

generate_texts("Once upon a time in a land far away,", n=3)
generate_texts("The most important thing about deep learning is", n=3)
```

"""
        Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        프롬프트: "Once upon a time in a land far away,"
  [1] Once upon a time in a land far away, a white man, whom the Lord hath chosen as a prophet, 
  and made a king over the Jews, that they might judge him.
  And he did, and went out. And the Jews scattered among the nations, and slew him. 
  So they brought him back to Babylon. But he spake unto them, saying, Behold, I am your God; I am a god, and am your
  [2] Once upon a time in a land far away, with a vast, vast, expansive land far from any nation or country,
  that the nations of the earth might be able to unite to save the people. 
  And with these words they were lifted up into heaven, where they lay upon the throne of the Most High God, as high as the throne above the heavens.
  And it is this same God who was the first to go forth and give this people
  [3] Once upon a time in a land far away, there was a small group of people, mostly merchants and others, that wished to get back to their old lives. 
  They would not, as they now say, go back to the land that they used to be in. It was called Zilchay.
  It was a little bit of a town, but it was still a small town, and there were very few people there. All


        Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        프롬프트: "The most important thing about deep learning is"
    [1] The most important thing about deep learning is that the training data we have here is really useful for the next step to solving complex algorithms. 
    And we need to think about how to do that. We need to create more general deep learning algorithms, like neural networks, to do those general operations. 
    If you want to do a deep learning neural network, like a neural network to see how much of a human is alive, and how much more we
    [2] The most important thing about deep learning is the ability to train your neural networks to recognize objects, and it's only natural that they will. You can use deep
    learning to learn how to recognize what you're seeing in a movie or a picture to make a decision. But you don't have to learn to recognize people.
    The other important thing is the use of deep learning in the classroom, where students learn to teach, and there
    [3] The most important thing about deep learning is that it allows you to design your applications as you work on them.
    In this case, our application is a WebApp. It has a main entry point called Page 1, 
    and a sub-entry called "Hello World!" which contains some images of the page.
    We can then make the WebApp a custom content store. , which contains the text and images of Page 1. We
"""

***

## 8. BERT vs GPT 성능 비교 실험

### 8-1. 2편 ~ 4편 모델 총 비교

```python
results = {
    "2편 Bi-LSTM\n(Keras)"    : 87.2,
    "3편 Transformer\n(Keras)": 88.5,
    "4편 BERT\n(Fine-tuning)" : test_acc * 100
}

colors = ["#5B9BD5", "#70AD47", "#ED7D31"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(results.keys(), results.values(),
              color=colors, edgecolor="black", width=0.5)

ax.set_ylim(80, 100)
ax.set_title("IMDB 감성 분석 — 모델별 테스트 정확도 비교 (2편~4편)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("테스트 정확도 (%)")
ax.grid(axis="y", linestyle="--", alpha=0.7)

for bar, (name, val) in zip(bars, results.items()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
```


### 8-2. 핵심 특성 비교

| 비교 항목 | LSTM (2편) | Transformer (3편) | BERT (4편) |
| :-- | :-- | :-- | :-- |
| Fine-tuning 데이터 | 2.5만 건 | 2.5만 건 | 2천 건 |
| 학습 시간 | 빠름 | 보통 | 느림 (GPU 필요) |
| 정확도 | ~87% | ~88% | ~93% |
| 사전 학습 | ❌ | ❌ | ✅ (수십 GB) |
| 적은 데이터 성능 | 낮음 | 낮음 | **높음** |


***

## 9. 마무리

### 9-1. 오늘 배운 것 한눈에 정리

| 개념 | 핵심 내용 |
| :-- | :-- |
| BERT | Encoder 기반, 양방향 문맥, MLM+NSP 사전 학습 |
| GPT | Decoder 기반, 단방향 문맥, CLM 사전 학습 |
| GPT-5.4 (2026) | 1M 토큰 컨텍스트, 추론 강도 5단계, Computer Use |
| ModernBERT (2026) | BERT 경량화 + 8K 컨텍스트 + RoPE 위치 임베딩 |
| 전이 학습 | 대규모 사전 학습 → 소규모 데이터 Fine-tuning |
| WordPiece / BPE | 서브워드 토큰화 → OOV 문제 해결 |
| [CLS] 토큰 | 문장 전체 의미 요약 → 분류 헤드에 연결 |
| AdamW + Warm-up | BERT Fine-tuning 표준 설정 |
| Top-p Sampling | GPT 생성 현재 권장 전략 |
| temperature | 생성 텍스트의 창의성 vs 안정성 조절 |

***