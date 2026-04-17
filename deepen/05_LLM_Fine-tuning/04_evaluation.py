"""
04. 파인튜닝 모델 평가
- 파인튜닝 전 vs 후 응답 비교
- ROUGE 자동 평가 지표
- LLM-as-Judge (GPT로 응답 품질 자동 평가)
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
from tqdm import tqdm
from unsloth import FastLanguageModel
from rouge_score import rouge_scorer
# openai는 섹션 3(LLM-as-Judge)에서 조건부 import (API 키 있을 때만)

# ── 설정 ─────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"  # RTX 3060 6GB 환경
MAX_SEQ_LENGTH = 1024                            # 6GB VRAM에 맞춰 축소
ROUGE_SAMPLES  = 10                              # ROUGE 평가 샘플 수 (50→10으로 시간 단축)

# 파인튜닝된 모델 로드 (03_qlora_dora_sft_training.py 실행 후)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "./qwen_lora_adapter",  # 어댑터 경로
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = True,
)


# ═══════════════════════════════════════
# 1. 파인튜닝 전 vs 후 응답 비교
# ═══════════════════════════════════════
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
            max_new_tokens     = max_new_tokens,
            max_length         = None,    # max_new_tokens와 충돌 경고 방지
            do_sample          = True,
            top_p              = 0.9,
            temperature        = 0.7,
            repetition_penalty = 1.1,     # 반복 억제
            pad_token_id       = tokenizer.eos_token_id
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
print("1. 파인튜닝 모델 응답 테스트")
print("=" * 60)
for question in test_questions:
    print(f"\n질문: {question}")
    print("-" * 40)
    response = generate_response(model, tokenizer, question)
    print(f"응답:\n{response}")
    print("=" * 60)


# ═══════════════════════════════════════
# 2. 자동 평가 지표 — ROUGE
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print(f"2. ROUGE 자동 평가 (샘플 {ROUGE_SAMPLES}개)")
print("=" * 60)

# pip install rouge-score
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=False
)

# eval_dataset 로드 (03에서 저장한 데이터 또는 재생성)
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.select(range(1000))
split = dataset.train_test_split(test_size=0.05, seed=42)
eval_dataset = split["test"]

# 평가용 서브셋 (한 번만 select 후 재사용)
eval_subset = eval_dataset.select(range(ROUGE_SAMPLES))

# 예측값 vs 정답 비교 (진행률 표시)
print(f"\n{ROUGE_SAMPLES}개 샘플 생성 중...")
predictions = []
for d in tqdm(eval_subset, desc="ROUGE 생성", ncols=80):
    pred = generate_response(
        model, tokenizer, d["instruction"], d["input"],
        max_new_tokens=128   # ROUGE 평가용은 짧게 (속도 향상)
    )
    predictions.append(pred)
references = [d["output"] for d in eval_subset]

scores = {"rouge1": [], "rouge2": [], "rougeL": []}
for pred, ref in zip(predictions, references):
    s = scorer.score(ref, pred)
    for k in scores:
        scores[k].append(s[k].fmeasure)

print("\n평가 결과 (ROUGE F1):")
for k, v in scores.items():
    print(f"  {k.upper():8s}: {sum(v)/len(v):.4f}")


# ═══════════════════════════════════════
# 3. LLM-as-Judge — GPT로 응답 품질 자동 평가 (옵션)
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("3. LLM-as-Judge (옵션 — OPENAI_API_KEY 환경변수 필요)")
print("=" * 60)

if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY 환경변수가 없어 LLM-as-Judge 섹션을 스킵합니다.")
    print("   사용하려면 다음과 같이 설정 후 재실행하세요:")
    print('     PowerShell: $env:OPENAI_API_KEY="sk-..."')
    print('     cmd      : set OPENAI_API_KEY=sk-...')
else:
    try:
        from openai import OpenAI
        client = OpenAI()

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
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
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
    except ImportError:
        print("⚠️  openai 패키지가 설치되지 않아 스킵합니다.")
        print("   설치: pip install openai")
    except Exception as e:
        print(f"⚠️  LLM-as-Judge 실행 실패: {e}")
