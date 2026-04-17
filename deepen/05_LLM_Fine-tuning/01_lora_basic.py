"""
01. LoRA 기본 구현 — 저랭크 분해로 효율적 학습
- LoRA 어댑터를 모델에 부착하고 학습 가능한 파라미터 수 확인
- target_modules 탐색
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # RTX 3060 6GB 환경 — 1.5B로 다운사이즈

# ── target_modules 확인 ──────────────────────────
print("=" * 60)
print("1. 모델의 프로젝션 레이어 확인")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for name, module in model.named_modules():
    if "proj" in name or "linear" in name.lower():
        print(name)

# ── LoRA 설정 ─────────────────────────────────────
print("\n" + "=" * 60)
print("2. LoRA 어댑터 부착")
print("=" * 60)

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
