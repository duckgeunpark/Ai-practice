"""
[BERT 실습 5] HuggingFace Transformers 라이브러리 기초
실행: python huggingface_basics.py

- 5-2 pipeline: 감성 분석 / 텍스트 생성 / 빈칸 채우기 / 질의응답 4가지 데모
- 5-3 AutoModel / AutoTokenizer: 모델 로드 + 파라미터/임베딩 차원 확인
- 5-4 Tokenizer 완벽 이해: encode/decode, 패딩, 문장쌍, WordPiece(BERT) vs BPE(GPT)
- 참고: BERT.md 섹션 5
# pip install transformers torch matplotlib
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

# 출력 경로: deepen/BERT/data/figures/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))
FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 디바이스 자동 감지 (pipeline에 전달할 때만 사용; 0=GPU, -1=CPU)
DEVICE_ID = 0 if torch.cuda.is_available() else -1
print(f"사용 장치: {'cuda:0' if DEVICE_ID == 0 else 'cpu'}")


# ═══════════════════════════════════════════════════════════════
# 5-2. pipeline — 코드 5줄로 바로 사용
# ═══════════════════════════════════════════════════════════════
def demo_pipeline():
    print("\n" + "=" * 64)
    print("5-2. pipeline 데모")
    print("=" * 64)

    # ── 감성 분석 ──────────────────────────────────────────────
    print("\n[1/4] sentiment-analysis")
    try:
        classifier = pipeline("sentiment-analysis", device=DEVICE_ID)
        results = classifier([
            "This movie was absolutely fantastic!",
            "I hated every minute of this film.",
        ])
        for r in results:
            print(f"  레이블: {r['label']:10s}  점수: {r['score']:.4f}")
    except Exception as e:
        print(f"  ⚠️ 실패: {e}")

    # ── 텍스트 생성 (GPT-2) ────────────────────────────────────
    print("\n[2/4] text-generation (gpt2)")
    try:
        generator = pipeline("text-generation", model="gpt2", device=DEVICE_ID)
        outputs = generator(
            "Artificial intelligence will",
            max_new_tokens=40,
            num_return_sequences=2,
            do_sample=True,
            temperature=0.8,
            pad_token_id=50256,  # gpt2 eos_token_id
        )
        for i, o in enumerate(outputs):
            print(f"  생성 {i+1}: {o['generated_text']}")
    except Exception as e:
        print(f"  ⚠️ 실패: {e}")

    # ── 빈칸 채우기 (BERT 스타일 MLM) ─────────────────────────
    print("\n[3/4] fill-mask (bert-base-uncased)")
    try:
        fill_mask = pipeline(
            "fill-mask", model="bert-base-uncased", device=DEVICE_ID
        )
        results = fill_mask("The capital of France is [MASK].")
        for r in results[:3]:
            print(f"  {r['token_str']:15s} 확률: {r['score']:.4f}")
    except Exception as e:
        print(f"  ⚠️ 실패: {e}")

    # ── 질의응답 (QA) ──────────────────────────────────────────
    # transformers 5.5.0의 pipeline("question-answering") 등록 누락 이슈로
    # AutoModelForQuestionAnswering을 직접 사용 (pipeline 내부 동작 노출)
    print("\n[4/4] question-answering (AutoModel 직접 사용)")
    try:
        qa_name  = "distilbert-base-cased-distilled-squad"
        qa_tok   = AutoTokenizer.from_pretrained(qa_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_name)
        if DEVICE_ID == 0:
            qa_model = qa_model.to("cuda")

        question = "When was Hugging Face founded?"
        context  = (
            "Hugging Face is a company that develops tools for building "
            "machine learning applications. It was founded in 2016 and "
            "is headquartered in New York City."
        )

        inputs = qa_tok(question, context, return_tensors="pt")
        if DEVICE_ID == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = qa_model(**inputs)

        # 답변 span 추출 (start/end logits 최대값 위치)
        start = out.start_logits.argmax().item()
        end   = out.end_logits.argmax().item() + 1
        answer_ids = inputs["input_ids"][0][start:end]
        answer = qa_tok.decode(answer_ids, skip_special_tokens=True)

        # 신뢰도 = start확률 × end확률
        s_prob = torch.softmax(out.start_logits, dim=-1)[0, start].item()
        e_prob = torch.softmax(out.end_logits,   dim=-1)[0, end - 1].item()
        score  = s_prob * e_prob

        print(f"  질문: {question}")
        print(f"  답변: {answer}")
        print(f"  점수: {score:.4f}")
    except Exception as e:
        print(f"  ⚠️ 실패: {e}")


# ═══════════════════════════════════════════════════════════════
# 5-3. AutoModel / AutoTokenizer
# ═══════════════════════════════════════════════════════════════
def demo_automodel():
    print("\n" + "=" * 64)
    print("5-3. AutoModel / AutoTokenizer")
    print("=" * 64)

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    vocab_size = tokenizer.vocab_size

    print(f"모델 이름:        {model_name}")
    print(f"총 파라미터 수:   {total_params:,}")
    print(f"은닉 차원:        {hidden_size}")
    print(f"레이어 수:        {n_layers}")
    print(f"어텐션 헤드 수:   {n_heads}")
    print(f"어휘 크기:        {vocab_size:,}")

    # 간단한 forward pass로 [CLS] 출력 벡터 확인
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
    cls_vector = last_hidden[:, 0, :]  # [CLS] 토큰
    print(f"\nlast_hidden_state shape: {tuple(last_hidden.shape)}")
    print(f"[CLS] 벡터 shape:        {tuple(cls_vector.shape)}")
    print(f"[CLS] 벡터 앞 5개 값:    {cls_vector[0, :5].tolist()}")


# ═══════════════════════════════════════════════════════════════
# 5-4. Tokenizer 완벽 이해
# ═══════════════════════════════════════════════════════════════
def demo_tokenizer():
    print("\n" + "=" * 64)
    print("5-4. Tokenizer 완벽 이해")
    print("=" * 64)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Hello, I am learning about BERT!"

    # ── 기본 토큰화 ────────────────────────────────────────────
    tokens = tokenizer.tokenize(text)
    print(f"\n원문:   {text}")
    print(f"토큰:   {tokens}")

    # ── 인코딩 (max_length 패딩) ───────────────────────────────
    encoding = tokenizer(
        text,
        max_length=20,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"][0].tolist()
    attn_mask = encoding["attention_mask"][0].tolist()
    type_ids = encoding["token_type_ids"][0].tolist()

    print(f"\ninput_ids:      {input_ids}")
    print(f"attention_mask: {attn_mask}")
    print(f"token_type_ids: {type_ids}")
    print(f"디코딩:         {tokenizer.decode(input_ids)}")

    # ── 두 문장 동시 인코딩 (문장 쌍 태스크) ──────────────────
    enc_pair = tokenizer(
        "I had lunch.", "I felt full afterwards.",
        max_length=20,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print("\n─ 문장 쌍 인코딩 ─")
    print(f"input_ids:      {enc_pair['input_ids'][0].tolist()}")
    print(f"token_type_ids: {enc_pair['token_type_ids'][0].tolist()}")
    print("                ↑ 0 = 문장A, 1 = 문장B, 0(뒤쪽) = 패딩")

    # ── 시각화: attention_mask & token_type_ids ───────────────
    pair_ids = enc_pair["input_ids"][0].tolist()
    pair_attn = enc_pair["attention_mask"][0].tolist()
    pair_type = enc_pair["token_type_ids"][0].tolist()
    pair_tokens = tokenizer.convert_ids_to_tokens(pair_ids)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5))
    x = np.arange(len(pair_tokens))

    ax1.bar(x, pair_attn, color="steelblue", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_tokens, rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.2)
    ax1.set_title("attention_mask  (1 = 실제 토큰 / 0 = 패딩)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    colors = ["#5B9BD5" if t == 0 else "#ED7D31" for t in pair_type]
    ax2.bar(x, [1] * len(x), color=colors, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_tokens, rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.2)
    ax2.set_title("token_type_ids  (파란색 = 문장A=0 / 주황색 = 문장B=1)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("BERT 문장 쌍 인코딩 — attention_mask & token_type_ids",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "05_tokenizer_pair_encoding.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\n시각화 저장: {fig_path}")
    plt.close(fig)

    # ── WordPiece (BERT) vs BPE (GPT-2) ───────────────────────
    print("\n─ 서브워드 토큰화 비교 ─")
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    samples = ["unbelievable", "playing", "tokenization", "lowest"]

    print(f"{'단어':15s} | {'BERT (WordPiece)':40s} | GPT-2 (BPE)")
    print("-" * 90)
    for w in samples:
        bert_pieces = bert_tok.tokenize(w)
        gpt2_pieces = gpt2_tok.tokenize(w)
        print(f"{w:15s} | {str(bert_pieces):40s} | {gpt2_pieces}")

    print("\n💡 ##는 BERT WordPiece에서 '앞 토큰과 이어지는 서브워드' 표시")
    print("   Ġ는 GPT-2 BPE에서 '단어 시작 = 앞에 공백' 표시")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo_pipeline()
    demo_automodel()
    demo_tokenizer()
    print("\n" + "=" * 64)
    print("모든 데모 완료!")
    print("=" * 64)
