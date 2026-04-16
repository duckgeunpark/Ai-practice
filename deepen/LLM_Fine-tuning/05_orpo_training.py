"""
05. ORPO — SFT + 선호도 학습을 단일 손실로 통합
- ORPO 데이터셋 구성 (선호/비선호 응답 쌍)
- ChatML 포맷 변환
- ORPOTrainer로 학습 및 시각화
"""

import os
import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from trl import ORPOConfig, ORPOTrainer
from datasets import Dataset

# ── 작업 디렉토리를 스크립트 위치(LLM_Fine-tuning)로 고정 ──
# orpo_output, unsloth_compiled_cache 등 모든 출력 폴더가
# 어디서 실행하든 LLM_Fine-tuning 폴더 안에 생성되도록 함
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 한글 폰트 설정 (Windows: Malgun Gothic) ─────────
plt.rcParams["font.family"]      = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ── 설정 ─────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"  # RTX 3060 6GB 환경
MAX_SEQ_LENGTH = 1024                            # 6GB VRAM에 맞춰 축소
LORA_R         = 16
LORA_ALPHA     = 32


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
# 2. ORPO 데이터셋 구성
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("2. ORPO 데이터셋 구성")
print("=" * 60)

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

orpo_dataset = Dataset.from_list(orpo_data)
print(orpo_dataset)


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
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = orpo_formatted,
    args          = orpo_args,
)

print("ORPO 학습 시작...")
orpo_stats = orpo_trainer.train()
print(f"학습 완료! 총 시간: {orpo_stats.metrics['train_runtime']:.0f}초")


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
plt.show()
