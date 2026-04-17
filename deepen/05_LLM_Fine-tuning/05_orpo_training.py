"""
05. ORPO — SFT + 선호도 학습을 단일 손실로 통합
- ORPO 데이터셋 구성 (선호/비선호 응답 쌍)
- ChatML 포맷 변환
- ORPOTrainer로 학습 및 시각화
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
from trl import ORPOConfig, ORPOTrainer
from datasets import Dataset, load_dataset

# ── 한글 폰트 설정 (Windows: Malgun Gothic) ─────────
plt.rcParams["font.family"]      = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ── 설정 ─────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"  # RTX 3060 6GB 환경
MAX_SEQ_LENGTH = 1024                            # 6GB VRAM에 맞춰 축소
LORA_R         = 16
LORA_ALPHA     = 32
ORPO_SAMPLES   = 500                             # ORPO 학습 샘플 수
# 영어 원본 사용: 라벨 품질이 높음 (한국어 번역본은 라벨이 부정확)
# 다국어 모델이므로 영어 선호도 학습이 한국어 응답 품질에도 전이됨
DATASET_NAME   = "HuggingFaceH4/ultrafeedback_binarized"
DATASET_SPLIT  = "train_prefs"
DATASET_FALLBACK = ("maywell/ko_Ultrafeedback_binarized", "train")


# ═══════════════════════════════════════
# 1. 모델 로드 + QLoRA + DoRA 어댑터
# ═══════════════════════════════════════
print("=" * 60)
print("1. 모델 로드 및 어댑터 설정")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    use_dora       = True,
    random_state   = 42,
)

model.print_trainable_parameters()


# ═══════════════════════════════════════
# 2. ORPO 데이터셋 로드 (선호 / 비선호 응답 쌍)
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print(f"2. ORPO 데이터셋 로드 ({ORPO_SAMPLES}개 샘플)")
print("=" * 60)

# 데이터셋의 chosen/rejected 컬럼이 문자열일 수도, 메시지 리스트일 수도 있음
# (예: HuggingFaceH4/ultrafeedback_binarized 는 [{"role": ..., "content": ...}, ...])
def _extract_assistant_text(value):
    """chosen/rejected에서 assistant 응답 텍스트만 추출"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for msg in reversed(value):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # assistant 못 찾으면 전부 합치기
        return "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in value
        )
    return str(value)


def _extract_user_prompt(value):
    """prompt가 메시지 리스트일 경우 user 발화만 추출"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for msg in reversed(value):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return str(value)


def normalize_pair(example):
    """다양한 컬럼 구조를 (prompt, chosen, rejected) 문자열로 표준화"""
    # prompt 컬럼이 없으면 chosen 메시지 리스트의 user 발화에서 추출
    if "prompt" in example and example["prompt"] is not None:
        prompt = _extract_user_prompt(example["prompt"])
    else:
        prompt = _extract_user_prompt(example["chosen"])
    return {
        "prompt"  : prompt,
        "chosen"  : _extract_assistant_text(example["chosen"]),
        "rejected": _extract_assistant_text(example["rejected"]),
    }


# 영어 원본 우선, 실패 시 한국어 번역본으로 폴백
try:
    raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"✅ 데이터셋 로드: {DATASET_NAME} ({DATASET_SPLIT})")
except Exception as e:
    fb_name, fb_split = DATASET_FALLBACK
    print(f"⚠️  {DATASET_NAME} 로드 실패: {e}")
    print(f"   → 폴백: {fb_name} ({fb_split})")
    raw = load_dataset(fb_name, split=fb_split)

raw = raw.shuffle(seed=42).select(range(min(ORPO_SAMPLES, len(raw))))
orpo_dataset = raw.map(normalize_pair, remove_columns=raw.column_names)
print(orpo_dataset)
print(f"샘플 prompt:   {orpo_dataset[0]['prompt'][:80]}...")
print(f"샘플 chosen:   {orpo_dataset[0]['chosen'][:80]}...")
print(f"샘플 rejected: {orpo_dataset[0]['rejected'][:80]}...")


# ═══════════════════════════════════════
# 3. ChatML 포맷 변환
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("3. ChatML 포맷 변환")
print("=" * 60)


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
    chosen_messages = prompt_messages + [
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
    remove_columns=orpo_dataset.column_names
)

print("ORPO 데이터 샘플:")
print(f"prompt:   {orpo_formatted[0]['prompt'][:100]}...")
print(f"chosen:   {orpo_formatted[0]['chosen'][:100]}...")
print(f"rejected: {orpo_formatted[0]['rejected'][:100]}...")


# ═══════════════════════════════════════
# 4. ORPOTrainer로 학습
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("4. ORPO 학습 시작")
print("=" * 60)

orpo_args = ORPOConfig(
    output_dir                  = "./orpo_output",
    num_train_epochs            = 1,              # 500개 × 1 epoch = ~62 steps
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate               = 5e-6,           # ORPO 권장값
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.1,
    beta                        = 0.1,            # Odds Ratio 손실 가중치 λ
    max_length                  = MAX_SEQ_LENGTH,
    max_prompt_length           = 512,
    bf16                        = True,
    gradient_checkpointing      = True,
    optim                       = "paged_adamw_8bit",
    logging_steps               = 10,
    save_strategy               = "no",           # 학습 후 어댑터만 따로 저장
    remove_unused_columns       = False,
)

orpo_trainer = ORPOTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = orpo_formatted,
    args          = orpo_args,
)

print("ORPO 학습 시작...")
orpo_stats = orpo_trainer.train()
print(f"학습 완료! 총 시간: {orpo_stats.metrics['train_runtime']:.0f}초")

# ── 어댑터 저장 ─────────────────────────────
ADAPTER_DIR = "./qwen_orpo_adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"ORPO 어댑터 저장 완료: {ADAPTER_DIR}")


# ═══════════════════════════════════════
# 5. ORPO 학습 곡선 시각화
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("5. ORPO 학습 곡선 시각화")
print("=" * 60)

orpo_log = orpo_trainer.state.log_history

steps        = [x["step"] for x in orpo_log if "loss" in x]
total_losses = [x["loss"]     for x in orpo_log if "loss" in x]
sft_losses   = [x.get("sft_loss",        0) for x in orpo_log if "loss" in x]
odds_losses  = [x.get("odds_ratio_loss", 0) for x in orpo_log if "loss" in x]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(steps, total_losses, color="steelblue")
axes[0].set_title("전체 손실 (SFT + OR)")
axes[0].set_xlabel("스텝")
axes[0].grid(True)

axes[1].plot(steps, sft_losses, color="green")
axes[1].set_title("SFT 손실")
axes[1].set_xlabel("스텝")
axes[1].grid(True)

axes[2].plot(steps, odds_losses, color="coral")
axes[2].set_title("Odds Ratio 손실")
axes[2].set_xlabel("스텝")
axes[2].grid(True)

plt.suptitle("ORPO 학습 손실 변화", fontsize=14, fontweight="bold")
plt.tight_layout()

# ── 그래프 저장 (data/figures/) ─────────────
os.makedirs("./figures", exist_ok=True)
fig_path = "./figures/05_orpo_loss.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
print(f"학습 곡선 저장: {fig_path}")
plt.show()
