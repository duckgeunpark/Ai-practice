"""
02. 데이터 준비 — 파인튜닝 데이터 만들기
- Instruction 데이터셋 로드 및 분포 확인
- 커스텀 데이터셋 생성
- ChatML 포맷 변환
"""

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset

# ── 한글 폰트 설정 (Windows: Malgun Gothic) ─────────
plt.rcParams["font.family"]      = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ── 공개 Instruction 데이터셋 불러오기 ──────────────
print("=" * 60)
print("1. 공개 데이터셋 로드")
print("=" * 60)

# 영어 데이터셋
dataset = load_dataset("tatsu-lab/alpaca", split="train")
# 또는 한국어: load_dataset("iamseungjun/korean-alpaca-gpt4", split="train")

print(dataset)
print("\n첫 번째 샘플:")
print(dataset[0])

# ── 데이터 분포 확인 ──────────────────────────────
print("\n" + "=" * 60)
print("2. 데이터 텍스트 길이 분포")
print("=" * 60)

lengths = [len(d["instruction"]) + len(d["output"]) for d in dataset]
plt.figure(figsize=(10, 4))
plt.hist(lengths, bins=50, color="steelblue", edgecolor="black")
plt.title("데이터 텍스트 길이 분포")
plt.xlabel("문자 수")
plt.ylabel("샘플 수")
plt.axvline(x=2000, color="red", linestyle="--", label="MAX_SEQ 기준")
plt.legend()
plt.grid(True)
plt.show()

# ── 커스텀 데이터셋 만들기 ─────────────────────────
print("\n" + "=" * 60)
print("3. 커스텀 데이터셋 생성")
print("=" * 60)

raw_data = [
    {
        "instruction": "배송 조회 방법을 알려주세요.",
        "input": "",
        "output": "마이페이지 > 주문내역에서 운송장 번호를 확인하신 후 "
                  "택배사 홈페이지에서 조회하실 수 있습니다."
    },
    {
        "instruction": "반품 신청은 어떻게 하나요?",
        "input": "구매한 지 5일이 지났습니다.",
        "output": "구매 후 7일 이내에 반품 신청이 가능합니다. "
                  "마이페이지 > 주문내역 > 반품/교환 신청을 이용해주세요."
    },
    # ... 최소 500개 이상 권장
]

custom_dataset = Dataset.from_list(raw_data)
print(custom_dataset)

# ── ChatML 포맷 변환 함수 ──────────────────────────
print("\n" + "=" * 60)
print("4. ChatML 포맷 변환")
print("=" * 60)


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


# 토크나이저 로드 후 변환 테스트
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

formatted_dataset = dataset.map(
    lambda x: format_instruction_chatML(x, tokenizer),
    remove_columns=dataset.column_names
)

print("변환 후 샘플:")
print(formatted_dataset[0]["text"])
