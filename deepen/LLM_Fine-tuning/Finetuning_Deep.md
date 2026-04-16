[미검증]
## 0. 시리즈

| 심화편 | 제목 | 난이도 | 역할 |
| :-- | :-- | :-- | :-- |
| [심화 1편](https://duckport.pages.dev/posts/CNN_Deep) | CNN — 이미지 분류 완벽 정리 | ⭐⭐ | 이미지 처리 |
| [심화 2편](https://duckport.pages.dev/posts/RNN) | RNN / LSTM — 시계열·텍스트 처리 | ⭐⭐⭐ | 순서 데이터 |
| [심화 3편](https://duckport.pages.dev/posts/Transformer) | Transformer 구조 완벽 정리 | ⭐⭐⭐⭐ | 현대 AI 핵심 |
| [심화 4편](https://duckport.pages.dev/posts/BERT) | BERT / GPT 원리와 활용 | ⭐⭐⭐⭐ | 언어 모델 |
| [**심화 5편**](https://duckport.pages.dev/posts/Finetuning_Deep)⬅️ | HuggingFace로 LLM Fine-tuning | ⭐⭐⭐⭐⭐ | LLM 실전 |

***

## 1. Fine-tuning이란?

### 1-1. 4편에서 배운 Fine-tuning의 한계

4편 BERT 실습에서 `bert-base-uncased` 전체 파라미터(1.1억 개)를 업데이트했습니다. 이 방식은 **Full Fine-tuning** 으로, 작은 BERT 수준에서는 가능하지만 오늘날 실무에서 쓰는 7B~70B 규모 LLM에는 적용이 거의 불가능합니다.

```
Full Fine-tuning 문제점:

LLaMA 3.3 70B (700억 파라미터) 기준
  - float32: 파라미터당 4바이트 × 700억 = 280GB VRAM 필요
  - float16: 140GB → A100 80GB 기준 2장 필요
  - 그라디언트, 옵티마이저 상태까지 포함 시 → 최소 4~6배
  → 일반 개발자/연구자 환경에서 사실상 불가능
```


### 1-2. 2026년 Fine-tuning 환경

2026년 현재는 **소비자 GPU 한 장으로 수십 억 파라미터 모델을 파인튜닝** 할 수 있습니다.

```
2026년 현재 상황:
- RTX 4090 (24GB VRAM) 한 장으로 7B QLoRA 파인튜닝 가능
- Qwen 3.5, LLaMA 3.3, DeepSeek v4 등 고성능 오픈소스 선택지 다양
- Unsloth 도구로 표준 대비 속도 2배, VRAM 70% 절감
- ORPO로 SFT + 선호도 학습을 단일 루프에서 처리
```


### 1-3. 4가지 접근법 비교

| 방식 | 설명 | 비용 | 성능 |
| :-- | :-- | :-- | :-- |
| 프롬프트 엔지니어링 | 모델 그대로, 지시문만 잘 작성 | 없음 | △ |
| RAG | 외부 문서 검색 후 컨텍스트 주입 | 낮음 | △~◯ |
| **PEFT (LoRA/QLoRA)** | 파라미터 일부만 학습 | 낮음 | ◯~◎ |
| Full Fine-tuning | 전체 파라미터 업데이트 | 매우 높음 | ◎ |

> 💡 2026년 실무에서는 **PEFT(LoRA/QLoRA)** 가 비용과 성능의 최적 균형점으로 사실상 표준입니다.

***

## 2. PEFT — 파라미터 효율적 Fine-tuning

### 2-1. PEFT란?

**PEFT(Parameter-Efficient Fine-Tuning)** 는 전체 파라미터 중 극히 일부만 학습하면서도 Full Fine-tuning에 근접한 성능을 내는 기법입니다.

```
Full Fine-tuning:
  [베이스 모델 700억 파라미터]  ← 전부 업데이트
  ↑ GPU VRAM 수백 GB 필요

PEFT (LoRA):
  [베이스 모델 700억 파라미터]  ← 동결 (업데이트 안 함)
         +
  [LoRA 어댑터 7천만 파라미터]  ← 이것만 학습 (전체의 0.1%)
  ↑ GPU VRAM 수십 GB로 충분
```


### 2-2. 설치

```bash
pip install transformers datasets trl peft bitsandbytes accelerate
pip install unsloth  # 2026년 필수 속도 최적화 도구
```


***

## 3. LoRA — 저랭크 분해로 효율적 학습

### 3-1. LoRA 원리

**LoRA(Low-Rank Adaptation)** 는 거대한 가중치 행렬의 변화량을 두 개의 작은 행렬로 근사합니다.

```
기존 방식 (Full Fine-tuning):
  W_new = W_original + ΔW
  ΔW shape: (4096, 4096) = 1,677만 파라미터

LoRA 방식:
  ΔW ≈ A × B
  A shape: (4096, 16)  = 65,536 파라미터
  B shape: (16, 4096)  = 65,536 파라미터
  합계: 131,072 파라미터 (원본의 0.78%)

→ 동결된 W_original에 A×B를 더해서 사용:
  h = W_original · x + (A × B) · x × (α/r)
          ↑ 고정              ↑ 학습
```

```
α(lora_alpha)와 r(rank)의 관계:
  스케일링 계수 = α / r

예: r=16, α=32 → 스케일 = 2.0
    r=16, α=16 → 스케일 = 1.0

일반적 규칙:
  α = 2 × r  (가장 많이 사용되는 설정)
```


### 3-2. target_modules — 어느 레이어에 적용하나?

```python
# Transformer 모델의 Attention 레이어 구조
# Q, K, V, O 프로젝션 레이어가 주요 대상

# 최소 설정 (VRAM 최소):
target_modules = ["q_proj", "v_proj"]

# 권장 설정 (성능 극대화):
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
# FFN 레이어까지 포함

# 확인 방법
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
for name, module in model.named_modules():
    if "proj" in name or "linear" in name.lower():
        print(name)
```


### 3-3. LoRA 기본 구현

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 2026년 권장 오픈소스 모델

# 모델 불러오기 (float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype    = torch.float16,
    device_map     = "auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# LoRA 설정
lora_config = LoraConfig(
    r              = 16,             # rank — 클수록 표현력↑, VRAM↑
    lora_alpha     = 32,             # α = 2×r 권장
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,           # 과적합 방지
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM
)

# LoRA 어댑터 부착
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 출력 예시:
# trainable params: 83,886,080 || all params: 7,615,647,744 || trainable%: 1.10
```


***

## 4. QLoRA — 4비트 양자화 + LoRA

### 4-1. 양자화(Quantization)란?

양자화는 모델 가중치를 **낮은 비트로 표현해 메모리를 절감** 하는 기법입니다.

```
비트 수별 메모리 비교 (7B 모델 기준):

float32 (32비트): 28 GB  ← 일반 학습
float16 (16비트): 14 GB  ← 표준 추론
  int8  ( 8비트):  7 GB  ← 양자화 추론
   nf4  ( 4비트):  3.5 GB  ← QLoRA 핵심

nf4 (NormalFloat4):
- 정규분포를 가정해 4비트로 최적 표현
- 일반 int4보다 정밀도 손실 적음
- double quantization 적용 시 추가 0.3 GB 절감
```


### 4-2. QLoRA 동작 원리

```
QLoRA =  4비트로 동결된 베이스 모델
        + float16 LoRA 어댑터 학습

베이스 모델 (nf4, 동결)  ←── 순전파 시에만 사용
     +
LoRA 어댑터 (float16, 학습) ←── 역전파는 여기서만

→ VRAM 사용량: Full Fine-tuning 대비 ~75% 절감
→ 성능 손실: 거의 없음 (논문 기준 Full의 97% 수준)
```


### 4-3. GPU VRAM별 실행 가능 모델

| VRAM | 가능한 모델 | 방식 |
| :-- | :-- | :-- |
| 8 GB (RTX 3070) | 3B 이하 | LoRA |
| 12 GB (RTX 3060 Ti) | 7B | QLoRA |
| 16 GB (RTX 4060 Ti) | 7B | QLoRA |
| 24 GB (RTX 4090) | **14B QLoRA / 7B LoRA** | QLoRA |
| 48 GB (A40) | 34B QLoRA / 14B LoRA | QLoRA |
| 80 GB (A100) | 70B QLoRA / 34B LoRA | QLoRA |


***

## 5. DoRA — 2026년 새 표준

### 5-1. DoRA란?

**DoRA(Weight-Decomposed LoRA)** 는 2024년 말 등장해 2026년 사실상 표준이 된 LoRA 개선 기법입니다.

```
LoRA:
  ΔW = A × B  (방향과 크기를 함께 학습)
  → Full Fine-tuning과 학습 패턴이 다름

DoRA:
  W = 크기(Magnitude) × 방향(Direction)
       ↑ 스칼라 값         ↑ 정규화된 벡터
       학습 가능           LoRA로 학습

  → Full Fine-tuning과 거의 동일한 학습 패턴
  → 같은 rank에서 LoRA보다 일관되게 높은 성능
```


### 5-2. LoRA vs DoRA 성능 비교

| 벤치마크 | LoRA (r=16) | DoRA (r=16) | Full FT |
| :-- | :-- | :-- | :-- |
| Commonsense Reasoning | 78.3% | **81.7%** | 82.1% |
| Math (GSM8K) | 65.2% | **68.9%** | 69.4% |
| 코드 생성 | 51.3% | **56.1%** | 57.0% |

DoRA는 Full Fine-tuning 성능의 **99% 이상**을 달성합니다.

### 5-3. rsLoRA — 학습 안정화

```python
# 고랭크(r=64 이상) 사용 시 학습이 불안정해지는 문제 해결
# 스케일링 계수를 α/r 대신 α/√r 로 변경

lora_config = LoraConfig(
    r            = 64,
    lora_alpha   = 64,
    use_rslora   = True,    # rsLoRA 활성화
    use_dora     = True,    # DoRA 활성화
    ...
)
```


### 5-4. 2026년 황금 설정

```python
# 2026년 현재 커뮤니티 검증 표준 설정
lora_config = LoraConfig(
    r              = 16,       # rank
    lora_alpha     = 32,       # α = 2×r
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    use_dora       = True,     # DoRA 활성화 ← 2026 표준
    use_rslora     = False,    # r<=16 에서는 불필요
    task_type      = TaskType.CAUSAL_LM
)
```


***

## 6. Unsloth — 2026년 파인튜닝 필수 도구

### 6-1. Unsloth란?

**Unsloth** 는 LoRA/QLoRA Fine-tuning을 **표준 HuggingFace PEFT 대비 속도 2배, VRAM 70% 절감** 으로 수행하는 최적화 라이브러리입니다.

```
표준 HuggingFace PEFT:
  - 7B QLoRA 학습 속도: ~1.2 it/s
  - VRAM 사용: 18 GB (RTX 4090 기준)

Unsloth:
  - 7B QLoRA 학습 속도: ~2.4 it/s  (+100%)
  - VRAM 사용: 6~8 GB  (-70%)
  - CUDA 커널 직접 최적화로 달성
  - HuggingFace와 완전 호환 (코드 거의 동일)
```


### 6-2. Unsloth 설치

> 💡 **2025년 2월 말부터 Windows 네이티브 공식 지원**
> — WSL 없이도 Windows에서 직접 설치 가능합니다.

#### 공통 설치
```bash
pip install unsloth
```

#### Windows 네이티브 (PyTorch CUDA 먼저)
```bash
# 1. PyTorch CUDA 버전 설치 (CUDA 12.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. 의존성 설치
pip install peft accelerate trl bitsandbytes

# 3. Unsloth 설치 (triton-windows 자동 포함)
pip install unsloth
```

> ⚠️ Windows에서 Unsloth 설치 시 PyTorch가 CPU 버전으로 다운그레이드될 수 있음
> → 설치 후 `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`로 재설치

#### Linux/WSL (선택)
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```


### 6-3. Unsloth로 모델 불러오기

```python
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 2048  # 최대 입력 길이 (모델에 따라 8192까지 가능)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "Qwen/Qwen2.5-7B-Instruct",  # 2026 권장
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,    # None = 자동 감지 (bfloat16 권장)
    load_in_4bit   = True,    # QLoRA 4비트 양자화
)

print(f"모델 로드 완료: {model.__class__.__name__}")
```


### 6-4. Unsloth + QLoRA + DoRA 어댑터 설정

```python
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # VRAM 추가 절감
    use_dora       = True,                   # DoRA 활성화
    random_state   = 42,
    use_rslora     = False,
)

model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,615,647,744 || trainable%: 1.10
```


***

## 7. 데이터 준비 — 파인튜닝 데이터 만들기

### 7-1. Instruction 데이터셋 구조

LLM 파인튜닝에서 가장 많이 쓰이는 데이터 포맷입니다.

```json
{
    "instruction": "다음 파이썬 코드의 버그를 찾아 수정해줘",
    "input": "def add(a, b):\n    return a - b",
    "output": "버그: '-' 연산자 대신 '+' 를 사용해야 합니다.\n\n수정 코드:\ndef add(a, b):\n    return a + b"
}
```

데이터 품질에 대해 명심할 것이 있습니다.

```
[파인튜닝 황금 법칙]
데이터 품질 > 데이터 양 > 모델 크기

- 10,000개 저품질 데이터 < 1,000개 고품질 데이터
- GPT-4 생성 데이터로 소형 모델 파인튜닝
  → 훨씬 큰 모델을 능가하는 사례 다수 보고
```


### 7-2. Chat Template — 2026년 표준 포맷

2026년 현재 가장 많이 쓰이는 **ChatML 포맷** 입니다.

```
ChatML 포맷:
<|im_start|>system
당신은 친절한 한국어 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
파이썬에서 리스트 정렬하는 방법 알려줘<|im_end|>
<|im_start|>assistant
파이썬에서 리스트를 정렬하는 방법은 두 가지가 있습니다...
<|im_end|>

Alpaca 포맷 (간단한 Instruction 태스크):
### Instruction:
아래 문장을 영어로 번역하세요.

### Input:
오늘 날씨가 매우 좋습니다.

### Response:
The weather is very nice today.
```


### 7-3. HuggingFace datasets로 데이터 불러오기

```python
from datasets import load_dataset

# 공개 Instruction 데이터셋 예시
dataset = load_dataset("iamseungjun/korean-alpaca-gpt4", split="train")
# 또는 영어 데이터셋
dataset = load_dataset("tatsu-lab/alpaca", split="train")

print(dataset)
print(dataset[0])
# {
#   'instruction': 'Give three tips for staying healthy.',
#   'input': '',
#   'output': '1. Eat a balanced diet...'
# }

# 데이터 분포 확인
import matplotlib.pyplot as plt

lengths = [len(d["instruction"]) + len(d["output"])
           for d in dataset]
plt.figure(figsize=(10, 4))
plt.hist(lengths, bins=50, color="steelblue", edgecolor="black")
plt.title("데이터 텍스트 길이 분포")
plt.xlabel("문자 수")
plt.ylabel("샘플 수")
plt.axvline(x=2000, color="red", linestyle="--", label="MAX_SEQ 기준")
plt.legend()
plt.grid(True)
plt.show()
```


### 7-4. 커스텀 데이터셋 만들기

```python
from datasets import Dataset

# 직접 만든 데이터 (예: 법률 QA, 쇼핑몰 CS, 의료 상담 등)
raw_data = [
    {
        "instruction": "배송 조회 방법을 알려주세요.",
        "input": "",
        "output": "마이페이지 > 주문내역에서 운송장 번호를 확인하신 후 택배사 홈페이지에서 조회하실 수 있습니다."
    },
    {
        "instruction": "반품 신청은 어떻게 하나요?",
        "input": "구매한 지 5일이 지났습니다.",
        "output": "구매 후 7일 이내에 반품 신청이 가능합니다. 마이페이지 > 주문내역 > 반품/교환 신청을 이용해주세요."
    },
    # ... 최소 500개 이상 권장
]

# HuggingFace Dataset으로 변환
custom_dataset = Dataset.from_list(raw_data)
print(custom_dataset)
```


### 7-5. Chat Template 적용 함수

```python
# ChatML 포맷으로 데이터 변환
def format_instruction_chatML(example, tokenizer):
    """
    Instruction 데이터를 ChatML 포맷 문자열로 변환
    """
    messages = [
        {"role": "system",
         "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다."},
        {"role": "user",
         "content": example["instruction"] +
                    (f"\n\n{example['input']}" if example["input"] else "")},
        {"role": "assistant",
         "content": example["output"]}
    ]
    # tokenizer.apply_chat_template이 자동으로 ChatML 포맷으로 변환
    text = tokenizer.apply_chat_template(
        messages,
        tokenize          = False,
        add_generation_prompt = False
    )
    return {"text": text}


# 데이터셋 전체에 적용
formatted_dataset = dataset.map(
    lambda x: format_instruction_chatML(x, tokenizer),
    remove_columns = dataset.column_names
)

print("변환 후 샘플:")
print(formatted_dataset[0]["text"])
```


***

## 8. 실습 — QLoRA + DoRA + SFTTrainer 파인튜닝

### 8-1. 전체 설정 한눈에

> 💡 **GPU VRAM별 권장 설정**
> - 24GB+ (RTX 4090): `Qwen2.5-7B-Instruct`, `MAX_SEQ_LENGTH=2048`, `BATCH_SIZE=2`
> - 12~16GB (RTX 4060 Ti / 3080): `Qwen2.5-7B-Instruct`, `MAX_SEQ_LENGTH=1024`, `BATCH_SIZE=1`
> - **6~8GB (RTX 3060 Mobile / 3070): `Qwen2.5-1.5B-Instruct`, `MAX_SEQ_LENGTH=1024`, `BATCH_SIZE=2`**
> - 4GB 이하: Colab 또는 Cloud GPU 권장

```python
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from datasets import load_dataset

# ───────────────────────────────────────
# 설정 값 (여기만 수정하면 됨)
# ───────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"  # RTX 3060 6GB 환경 (24GB+: 7B 권장)
MAX_SEQ_LENGTH = 1024                            # 6GB VRAM에 맞춰 축소 (24GB+: 2048)
LORA_R         = 16                              # LoRA rank
LORA_ALPHA     = 32                              # LoRA alpha
BATCH_SIZE     = 2                               # 배치 크기 (OOM 시 1로 줄일 것)
GRAD_ACCUM     = 8                               # 그라디언트 누적 (유효 배치=16)
EPOCHS         = 3                               # 에포크
LEARNING_RATE  = 2e-4                            # 학습률
OUTPUT_DIR     = "./qwen_finetuned"              # 저장 경로
```


### 8-2. 모델 및 토크나이저 불러오기

```python
# Unsloth로 4비트 양자화 모델 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,   # bfloat16 자동 감지
    load_in_4bit   = True,   # 4비트 QLoRA
)

# pad_token 설정 (없는 경우 eos_token으로 대체)
if tokenizer.pad_token is None:
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.pad_token_id  = tokenizer.eos_token_id

print(f"모델 로드 완료")
print(f"어휘 크기: {tokenizer.vocab_size:,}")
print(f"최대 위치 임베딩: {model.config.max_position_embeddings:,}")
```


### 8-3. QLoRA + DoRA 어댑터 설정

```python
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # VRAM 절감
    use_dora       = True,                   # DoRA 활성화
    random_state   = 42,
)

model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,615,647,744 || trainable%: 1.10%
```


### 8-4. 데이터 준비

```python
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 샘플 수 제한 (테스트용: 1,000개 / 실제: 전체 사용 권장)
dataset = dataset.select(range(1000))

# ChatML 포맷 변환
formatted = dataset.map(
    lambda x: format_instruction_chatML(x, tokenizer),
    remove_columns = dataset.column_names
)

# 훈련 / 검증 분리
split = formatted.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"훈련 샘플: {len(train_dataset)}")
print(f"검증 샘플: {len(eval_dataset)}")
print(f"\n샘플 텍스트:\n{train_dataset[0]['text'][:200]}...")
```


### 8-5. SFTTrainer — 학습 설정 및 실행

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,  # 유효 배치 = 2×8 = 16
    learning_rate           = LEARNING_RATE,
    lr_scheduler_type       = "cosine",        # 코사인 감쇠
    warmup_ratio            = 0.03,            # 전체 3% warm-up
    max_seq_length          = MAX_SEQ_LENGTH,
    bf16                    = True,            # bfloat16 사용
    gradient_checkpointing  = True,            # VRAM 절감
    optim                   = "paged_adamw_8bit",  # QLoRA용 옵티마이저
    logging_steps           = 10,
    eval_strategy           = "steps",
    eval_steps              = 50,
    save_strategy           = "epoch",
    save_total_limit        = 2,
    dataset_text_field      = "text",          # 학습에 사용할 컬럼
    packing                 = True,            # 짧은 문장 묶어 효율 극대화
    report_to               = "none",          # wandb 연동 시 "wandb"
)

trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    args          = training_args,
)

# VRAM 사용량 확인 후 학습 시작
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"GPU: {gpu_stats.name}")
print(f"최대 VRAM: {round(gpu_stats.total_memory / 1024**3, 3)} GB")
print(f"학습 전 VRAM 사용: {start_gpu_memory} GB")
print("\n학습 시작...")

trainer_stats = trainer.train()

# 학습 완료 후 VRAM 사용량
used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n학습 완료!")
print(f"학습 소요 시간: {trainer_stats.metrics['train_runtime']:.0f}초")
print(f"학습 중 최대 VRAM: {used_memory} GB")
```

> 💡 **paged_adamw_8bit** — QLoRA 학습에 특화된 옵티마이저입니다. 옵티마이저 상태를 GPU/CPU 간에 페이징하여 VRAM을 추가로 절감합니다.
>
> 💡 **packing=True** — 짧은 학습 샘플들을 `max_seq_length`에 맞게 이어 붙여 패딩 낭비 없이 GPU를 풀로 활용합니다. 보통 10~30% 속도 향상.

### 8-6. 손실 시각화

```python
import matplotlib.pyplot as plt

log_history = trainer.state.log_history

train_losses = [(x["step"], x["loss"])
                for x in log_history if "loss" in x]
eval_losses  = [(x["step"], x["eval_loss"])
                for x in log_history if "eval_loss" in x]

fig, ax = plt.subplots(figsize=(12, 5))

if train_losses:
    steps, losses = zip(*train_losses)
    ax.plot(steps, losses, label="훈련 손실", color="steelblue")

if eval_losses:
    steps, losses = zip(*eval_losses)
    ax.plot(steps, losses, label="검증 손실",
            color="coral", linestyle="--", marker="o")

ax.set_title("QLoRA + DoRA Fine-tuning 손실 변화")
ax.set_xlabel("학습 스텝")
ax.set_ylabel("Cross Entropy Loss")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.show()
```


### 8-7. LoRA 어댑터 저장 및 병합

```python
# ── 방법 1: 어댑터만 저장 (작은 용량) ─────────────────
# 베이스 모델 + 어댑터 각각 보관, 추론 시 병합
model.save_pretrained("./qwen_lora_adapter")
tokenizer.save_pretrained("./qwen_lora_adapter")
print("LoRA 어댑터 저장 완료 (~80MB)")

# ── 방법 2: 베이스 모델과 병합 후 저장 (독립 사용 가능) ───────
# 병합 = LoRA 가중치를 베이스 모델에 흡수
# 추론 속도 빨라짐 (어댑터 계산 제거)
model_merged = model.merge_and_unload()  # Unsloth 제공 함수
model_merged.save_pretrained("./qwen_finetuned_merged",
                              safe_serialization=True)
tokenizer.save_pretrained("./qwen_finetuned_merged")
print("병합 모델 저장 완료 (~14GB)")

# ── 어댑터 불러오기 ──────────────────────────
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
model_loaded = PeftModel.from_pretrained(base_model, "./qwen_lora_adapter")
print("LoRA 어댑터 불러오기 완료")
```


***

## 9. 파인튜닝 모델 평가하기

### 9-1. 파인튜닝 전 vs 후 응답 비교

```python
def generate_response(model, tokenizer, instruction,
                      input_text="", max_new_tokens=256):
    """파인튜닝된 모델로 응답 생성"""
    # 추론 모드 전환 (Unsloth 사용 시)
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system",
         "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다."},
        {"role": "user",
         "content": instruction + (f"\n\n{input_text}" if input_text else "")}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens      = max_new_tokens,
            do_sample           = True,
            top_p               = 0.9,
            temperature         = 0.7,
            repetition_penalty  = 1.1,     # 반복 억제
            pad_token_id        = tokenizer.eos_token_id
        )

    # 입력 부분 제거 → 생성된 부분만 추출
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# 비교 테스트
test_questions = [
    "한국의 대표적인 전통 음식 3가지를 설명해줘",
    "파이썬에서 딕셔너리와 리스트의 차이점은?",
    "머신러닝과 딥러닝의 차이를 초보자도 이해하도록 설명해줘"
]

print("=" * 60)
for question in test_questions:
    print(f"\n📌 질문: {question}")
    print("-" * 40)
    response = generate_response(model, tokenizer, question)
    print(f"🤖 응답:\n{response}")
    print("=" * 60)
```


### 9-2. 자동 평가 지표 — ROUGE

```python
from rouge_score import rouge_scorer

# pip install rouge-score
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=False
)

# 예측값 vs 정답 비교
predictions = [
    generate_response(model, tokenizer, d["instruction"], d["input"])
    for d in eval_dataset.select(range(50))  # 50개 샘플 평가
]
references = [d["output"] for d in eval_dataset.select(range(50))]

scores = {"rouge1": [], "rouge2": [], "rougeL": []}
for pred, ref in zip(predictions, references):
    s = scorer.score(ref, pred)
    for k in scores:
        scores[k].append(s[k].fmeasure)

print("\n평가 결과 (ROUGE F1):")
for k, v in scores.items():
    print(f"  {k.upper():8s}: {sum(v)/len(v):.4f}")
```


### 9-3. LLM-as-Judge — GPT로 응답 품질 자동 평가

2026년 현재 가장 신뢰받는 평가 방식입니다. GPT-4에게 두 응답 중 어느 것이 더 좋은지 평가하도록 시킵니다.

```python
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 환경변수 필요

def llm_judge(question, response_a, response_b):
    """
    GPT에게 두 응답 중 어느 것이 더 좋은지 평가 요청
    파인튜닝 전(A) vs 파인튜닝 후(B) 비교
    """
    prompt = f"""다음 두 AI 응답 중 어느 것이 더 좋은지 평가해주세요.

질문: {question}

[응답 A]:
{response_a}

[응답 B]:
{response_b}

평가 기준:
1. 정확성 (사실에 맞는가?)
2. 완성도 (질문에 충분히 답했는가?)
3. 자연스러움 (한국어가 자연스러운가?)

반드시 "A가 낫다", "B가 낫다", "동등하다" 중 하나로만 답하고
이유를 한 문장으로 설명하세요."""

    response = client.chat.completions.create(
        model    = "gpt-4o",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 100
    )
    return response.choices[0].message.content


# 비교 평가 실행
test_q = "기계학습에서 과적합이란 무엇이고 어떻게 방지하나요?"

# 베이스 모델 응답 (파인튜닝 전 — 별도 로드 필요)
resp_base = "(파인튜닝 전 베이스 모델 응답)"
resp_finetuned = generate_response(model, tokenizer, test_q)

verdict = llm_judge(test_q, resp_base, resp_finetuned)
print(f"질문: {test_q}")
print(f"GPT 평가: {verdict}")
```


***

## 10. ORPO — 2026년 선호도 학습 최신 기법

### 10-1. RLHF / DPO / ORPO 발전 흐름

```
[SFT만 사용 시의 문제]
  모델이 "올바른 답"은 학습하지만
  "나쁜 답을 피하는 것"은 학습 못 함
  → 사람이 원하지 않는 응답 생성 가능

[해결 방법의 발전]

RLHF (2022):
  SFT → 보상 모델 학습 → PPO 강화학습
  → 효과 좋지만 파이프라인 복잡, 학습 불안정

DPO (2023):
  RLHF를 closed-form 손실로 단순화
  → 별도 보상 모델 불필요, 그러나 SFT와 별개 학습

ORPO (2024~2026 표준):
  SFT 손실 + 선호도 손실을 단일 손실로 통합
  → 1단계 학습으로 SFT + DPO 효과 동시에 달성
  → 더 빠르고, 더 안정적, 성능도 동등 이상
```


### 10-2. ORPO 원리

**ORPO(Odds Ratio Preference Optimization)** 는 선호 응답과 비선호 응답의 **오즈비(Odds Ratio)** 를 이용해 선호도를 학습합니다.

```python
# ORPO 손실 = SFT 손실 + λ × Odds Ratio 손실
#
# SFT 손실:
#   L_SFT = -log P(y_chosen | x)
#   → 선호 응답의 가능도를 높임
#
# Odds Ratio 손실:
#   odds(y|x) = P(y|x) / (1 - P(y|x))
#   L_OR = -log σ(log (odds(y_chosen|x) / odds(y_rejected|x)))
#   → 선호/비선호 응답의 오즈비를 직접 최대화
#
# λ는 두 손실의 균형 파라미터 (기본값 0.1)
```


### 10-3. ORPO 데이터셋 구조

```python
# ORPO는 선호/비선호 응답 쌍이 필요
# {"prompt": ..., "chosen": ..., "rejected": ...}

orpo_data = [
    {
        "prompt": "파이썬으로 피보나치 수열을 구현해줘",
        "chosen": (
            "피보나치 수열을 구현하는 효율적인 방법입니다:\n\n"
            "```python\n"
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
            "```\n"
            "이 방식은 O(n) 시간복잡도로 효율적입니다."
        ),
        "rejected": (
            "피보나치는 어렵습니다. "
            "그냥 인터넷에서 찾아보세요."
        )
    },
    {
        "prompt": "머신러닝이 뭔지 쉽게 설명해줘",
        "chosen": (
            "머신러닝은 컴퓨터가 데이터로부터 스스로 패턴을 학습하는 기술입니다. "
            "예를 들어, 수만 장의 고양이 사진을 보여주면 컴퓨터가 '고양이'의 특징을 "
            "스스로 파악해서 새로운 사진에서도 고양이를 인식할 수 있게 됩니다."
        ),
        "rejected": (
            "머신러닝은 Machine Learning의 약자이며 AI의 하위 분야입니다."
        )
    }
]

from datasets import Dataset
orpo_dataset = Dataset.from_list(orpo_data)
```


### 10-4. ChatML 포맷으로 변환

```python
def format_orpo_chatML(example, tokenizer):
    """ORPO 데이터를 ChatML 포맷으로 변환"""
    prompt_messages = [
        {"role": "system",
         "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다."},
        {"role": "user",
         "content": example["prompt"]}
    ]

    # 프롬프트 (system + user)
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # 선호 응답
    chosen_messages  = prompt_messages + [
        {"role": "assistant", "content": example["chosen"]}
    ]
    chosen = tokenizer.apply_chat_template(
        chosen_messages, tokenize=False, add_generation_prompt=False
    )

    # 비선호 응답
    rejected_messages = prompt_messages + [
        {"role": "assistant", "content": example["rejected"]}
    ]
    rejected = tokenizer.apply_chat_template(
        rejected_messages, tokenize=False, add_generation_prompt=False
    )

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


orpo_formatted = orpo_dataset.map(
    lambda x: format_orpo_chatML(x, tokenizer),
    remove_columns = orpo_dataset.column_names
)

print("ORPO 데이터 샘플:")
print(f"prompt:   {orpo_formatted[0]['prompt'][:100]}...")
print(f"chosen:   {orpo_formatted[0]['chosen'][:100]}...")
print(f"rejected: {orpo_formatted[0]['rejected'][:100]}...")
```


### 10-5. ORPOTrainer로 학습

```python
from trl import ORPOConfig, ORPOTrainer

orpo_args = ORPOConfig(
    output_dir                  = "./orpo_output",
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate               = 8e-6,          # ORPO는 더 낮은 학습률 사용
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.1,
    beta                        = 0.1,            # Odds Ratio 손실 가중치 λ
    max_length                  = MAX_SEQ_LENGTH,
    max_prompt_length           = 512,
    bf16                        = True,
    gradient_checkpointing      = True,
    optim                       = "paged_adamw_8bit",
    logging_steps               = 5,
    save_strategy               = "epoch",
    remove_unused_columns       = False,
)

orpo_trainer = ORPOTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset = orpo_formatted,
    args          = orpo_args,
)

print("ORPO 학습 시작...")
orpo_stats = orpo_trainer.train()
print(f"학습 완료! 총 시간: {orpo_stats.metrics['train_runtime']:.0f}초")
```


### 10-6. ORPO 학습 곡선 시각화

```python
orpo_log = orpo_trainer.state.log_history

steps         = [x["step"] for x in orpo_log if "loss" in x]
total_losses  = [x["loss"]     for x in orpo_log if "loss" in x]
sft_losses    = [x.get("sft_loss",    0) for x in orpo_log if "loss" in x]
odds_losses   = [x.get("odds_ratio_loss", 0) for x in orpo_log if "loss" in x]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(steps, total_losses, color="steelblue")
axes[0].set_title("전체 손실 (SFT + OR)")
axes[0].set_xlabel("스텝"); axes[0].grid(True)

axes[1].plot(steps, sft_losses, color="green")
axes[1].set_title("SFT 손실")
axes[1].set_xlabel("스텝"); axes[1].grid(True)

axes[2].plot(steps, odds_losses, color="coral")
axes[2].set_title("Odds Ratio 손실")
axes[2].set_xlabel("스텝"); axes[2].grid(True)

plt.suptitle("ORPO 학습 손실 변화", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```


***

## 11. 마무리

### 11-1. 오늘 배운 것 한눈에 정리

| 개념 | 핵심 내용 |
| :-- | :-- |
| PEFT | 전체 파라미터 1~3%만 학습, 비용·시간 90% 절감 |
| LoRA | 가중치 변화량 ΔW를 저랭크 행렬 A×B로 근사 |
| QLoRA | 베이스 모델 4비트 동결 + LoRA float16 학습 |
| DoRA | 가중치를 방향·크기로 분해해 Full FT에 근접한 품질 |
| rsLoRA | 고랭크 학습 안정화 (α/√r 스케일링) |
| Unsloth | 속도 2배, VRAM 70% 절감, 2026년 파인튜닝 필수 |
| SFTTrainer | packing + Flash Attention2 + bfloat16 표준 설정 |
| ORPO | SFT + 선호도 학습을 단일 손실로 통합, 2026 표준 |
| LLM-as-Judge | GPT-4로 응답 품질 자동 평가 |
| LoRA 병합 | `merge_and_unload()`로 어댑터를 베이스에 흡수 |

***