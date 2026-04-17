"""
03. QLoRA + DoRA + SFTTrainer 파인튜닝
- Unsloth 기반 4비트 양자화 모델 로드
- QLoRA + DoRA 어댑터 설정
- SFTTrainer로 학습 및 손실 시각화
- LoRA 어댑터 저장 및 병합
"""

import os

# ── 작업 디렉토리/캐시 위치를 LLM_Fine-tuning/data 로 고정 ──
# 반드시 unsloth import보다 먼저 실행해야 함
# (Unsloth는 import 시점의 CWD 또는 UNSLOTH_COMPILE_LOCATION을 캡처)
_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_data_dir, exist_ok=True)
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.join(_data_dir, "unsloth_compiled_cache")
os.chdir(_data_dir)

import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM

# ── 한글 폰트 설정 (Windows: Malgun Gothic) ─────────
plt.rcParams["font.family"]      = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ───────────────────────────────────────
# 설정 값 (여기만 수정하면 됨)
# ───────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"  # RTX 3060 6GB 환경
MAX_SEQ_LENGTH = 1024                          # 6GB VRAM에 맞춰 절반으로 축소
LORA_R         = 16                            # LoRA rank
LORA_ALPHA     = 32                            # LoRA alpha
BATCH_SIZE     = 2                             # 배치 크기 (OOM 발생 시 1로 줄일 것)
GRAD_ACCUM     = 8                             # 그라디언트 누적 (유효 배치=16)
EPOCHS         = 3                             # 에포크
LEARNING_RATE  = 2e-4                          # 학습률
OUTPUT_DIR     = "./qwen_finetuned"            # 저장 경로


# ── ChatML 포맷 변환 함수 ──────────────────────────
def format_instruction_chatML(example, tokenizer):
    """Instruction 데이터를 ChatML 포맷 문자열로 변환"""
    messages = [
        {"role": "system",
         "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다."},
        {"role": "user",
         "content": example["instruction"] +
                    (f"\n\n{example['input']}" if example["input"] else "")},
        {"role": "assistant",
         "content": example["output"]}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


# ═══════════════════════════════════════
# 1. 모델 및 토크나이저 불러오기
# ═══════════════════════════════════════
print("=" * 60)
print("1. Unsloth로 4비트 양자화 모델 로드")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,   # bfloat16 자동 감지
    load_in_4bit   = True,   # 4비트 QLoRA
)

# pad_token 설정 (없는 경우 eos_token으로 대체)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"모델 로드 완료")
print(f"어휘 크기: {tokenizer.vocab_size:,}")
print(f"최대 위치 임베딩: {model.config.max_position_embeddings:,}")


# ═══════════════════════════════════════
# 2. QLoRA + DoRA 어댑터 설정
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("2. QLoRA + DoRA 어댑터 설정")
print("=" * 60)

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


# ═══════════════════════════════════════
# 3. 데이터 준비
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("3. 데이터 준비")
print("=" * 60)

dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 샘플 수 제한 (테스트용: 1,000개 / 실제: 전체 사용 권장)
dataset = dataset.select(range(1000))

# ChatML 포맷 변환
formatted = dataset.map(
    lambda x: format_instruction_chatML(x, tokenizer),
    remove_columns=dataset.column_names
)

# 훈련 / 검증 분리
split = formatted.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"훈련 샘플: {len(train_dataset)}")
print(f"검증 샘플: {len(eval_dataset)}")
print(f"\n샘플 텍스트:\n{train_dataset[0]['text'][:200]}...")


# ═══════════════════════════════════════
# 4. SFTTrainer — 학습 설정 및 실행
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("4. SFTTrainer 학습 시작")
print("=" * 60)

training_args = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,  # 유효 배치 = 2×8 = 16
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",        # 코사인 감쇠
    warmup_ratio                = 0.03,            # 전체 3% warm-up
    max_seq_length              = MAX_SEQ_LENGTH,
    bf16                        = True,            # bfloat16 사용
    gradient_checkpointing      = True,            # VRAM 절감
    optim                       = "paged_adamw_8bit",  # QLoRA용 옵티마이저
    logging_steps               = 1,             # 작은 step 수에 맞춰 매 step 기록
    eval_strategy               = "steps",
    eval_steps                  = 5,             # 그래프에 충분한 포인트가 찍히도록
    save_strategy               = "epoch",
    save_total_limit            = 2,
    dataset_text_field          = "text",          # 학습에 사용할 컬럼
    packing                     = True,            # 짧은 문장 묶어 효율 극대화
    report_to                   = "none",          # wandb 연동 시 "wandb"
)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
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


# ═══════════════════════════════════════
# 5. 손실 시각화
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("5. 손실 시각화")
print("=" * 60)

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
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════
# 6. LoRA 어댑터 저장 및 병합
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("6. LoRA 어댑터 저장 및 병합")
print("=" * 60)

# ── 방법 1: 어댑터만 저장 (작은 용량) ─────────────────
model.save_pretrained("./qwen_lora_adapter")
tokenizer.save_pretrained("./qwen_lora_adapter")
print("LoRA 어댑터 저장 완료 (~80MB)")

# ── 방법 2: 베이스 모델과 병합 후 저장 (독립 사용 가능) ───────
# ⚠️ 주의: 4비트 양자화 모델 + LoRA 병합은 transformers 5.5.0의
#   새로운 weight conversion 시스템과 호환되지 않아 NotImplementedError 발생.
#   또한 bnb 경고대로 4-bit linear 병합은 rounding error로 추론 결과가 달라질 수 있음.
#   QLoRA 사용 시에는 어댑터만 저장(방법 1) + 추론 시 베이스 모델 + 어댑터 로드가 권장 방식.
try:
    model_merged = model.merge_and_unload()
    model_merged.save_pretrained("./qwen_finetuned_merged",
                                  safe_serialization=True)
    tokenizer.save_pretrained("./qwen_finetuned_merged")
    print("병합 모델 저장 완료 (~14GB)")
except NotImplementedError as e:
    print(f"⚠️ 4-bit + LoRA 병합 저장 미지원 (transformers 5.5.0): {e}")
    print("   → 어댑터만 저장된 상태(./qwen_lora_adapter)를 사용하세요.")
    print("   → 추론 시: PeftModel.from_pretrained(base, './qwen_lora_adapter')")

# ── 어댑터 불러오기 ──────────────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
model_loaded = PeftModel.from_pretrained(base_model, "./qwen_lora_adapter")
print("LoRA 어댑터 불러오기 완료")
