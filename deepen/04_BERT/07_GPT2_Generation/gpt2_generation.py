"""
[BERT 실습 7] GPT-2 — 텍스트 생성 전략 비교
실행: python gpt2_generation.py

- 사전 학습된 GPT-2 (117M) 로 다양한 디코딩 전략 비교
  · Greedy / Beam / Top-k / Top-p (Nucleus) Sampling
- temperature 파라미터의 확률 분포 변화 시각화
- 여러 문장 동시 생성 데모
- 참고: BERT.md 섹션 7
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

MODEL_NAME = "gpt2"  # gpt2 / gpt2-medium / gpt2-large


def load_gpt2(device):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    # GPT-2는 pad_token이 기본적으로 없음 → 반드시 설정
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 파라미터 수: {n_params:,}")
    return tokenizer, model


def compare_strategies(tokenizer, model, device):
    prompt = "In the future, artificial intelligence will"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    model.eval()

    # 전략 1. Greedy Search — 매 시점 최대 확률 선택 (반복적/단조로움)
    greedy_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=False,
    )
    print("【Greedy Search】")
    print(tokenizer.decode(greedy_out[0], skip_special_tokens=True))
    print()

    # 전략 2. Beam Search — 상위 num_beams개 후보 동시 유지
    beam_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    print("【Beam Search (num_beams=5)】")
    print(tokenizer.decode(beam_out[0], skip_special_tokens=True))
    print()

    # 전략 3. Top-k Sampling — 상위 k개 중 랜덤 샘플링
    topk_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        temperature=0.8,
    )
    print("【Top-k Sampling (k=50, temp=0.8)】")
    print(tokenizer.decode(topk_out[0], skip_special_tokens=True))
    print()

    # 전략 4. Top-p (Nucleus) Sampling — 누적 확률 p 이하 집합에서 샘플링
    topp_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.92,
        temperature=0.8,
    )
    print("【Top-p Sampling (p=0.92, temp=0.8)】")
    print(tokenizer.decode(topp_out[0], skip_special_tokens=True))
    print()


def softmax_with_temp(logits, temperature):
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()


def plot_temperature_effect():
    logits = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
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

    plt.suptitle(
        "Temperature에 따른 확률 분포 변화\n"
        "(낮을수록 뾰족 → 보수적 / 높을수록 평평 → 창의적)"
    )
    plt.tight_layout()
    plt.show()


def generate_texts(
    tokenizer,
    model,
    device,
    prompt,
    n=3,
    max_new_tokens=80,
    top_p=0.92,
    temperature=0.9,
):
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=n,
        no_repeat_ngram_size=3,
    )

    print(f'프롬프트: "{prompt}"\n')
    for i, out in enumerate(outputs):
        text = tokenizer.decode(out, skip_special_tokens=True)
        print(f"  [{i+1}] {text}\n")


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    tokenizer, model = load_gpt2(device)

    print("\n" + "=" * 60)
    print("7-2. 생성 전략 비교")
    print("=" * 60)
    compare_strategies(tokenizer, model, device)

    print("=" * 60)
    print("7-3. Temperature 파라미터 이해 (그래프)")
    print("=" * 60)
    plot_temperature_effect()

    print("=" * 60)
    print("7-4. 여러 문장 동시 생성")
    print("=" * 60)
    generate_texts(
        tokenizer, model, device, "Once upon a time in a land far away,", n=3
    )
    generate_texts(
        tokenizer,
        model,
        device,
        "The most important thing about deep learning is",
        n=3,
    )


if __name__ == "__main__":
    main()
