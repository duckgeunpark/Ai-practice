"""
[BERT 실습 6] BERT Fine-tuning — IMDB 감성 분석
실행: python bert_imdb_sentiment.py

- 사전 학습된 bert-base-uncased 모델을 IMDB 영화 리뷰에 Fine-tuning
- [CLS] 토큰 출력 → Linear(768→2) 분류 헤드
- AdamW + Linear Warm-up 스케줄러 (BERT 표준 설정)
- 참고: BERT.md 섹션 6
# pip install transformers
  datasets scikit-learn
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
# 체크포인트 저장/로드 경로: deepen/BERT/data/best_bert_imdb.pth
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))
BEST_PATH = os.path.join(DATA_DIR, "best_bert_imdb.pth")


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

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
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def predict_sentiment(model, tokenizer, text, device):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
            token_type_ids=encoding["token_type_ids"].to(device),
        )
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    label = "긍정 😊" if probs.argmax().item() == 1 else "부정 😞"
    print(f"입력: {text[:60]}...")
    print(f"예측: {label}  (부정: {probs[0]:.4f} / 긍정: {probs[1]:.4f})")


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="훈련 손실", marker="o")
    ax1.plot(history["val_loss"], label="검증 손실", marker="o", linestyle="--")
    ax1.set_title("BERT Fine-tuning 손실")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="훈련 정확도", marker="o")
    ax2.plot(history["val_acc"], label="검증 정확도", marker="o", linestyle="--")
    ax2.set_title("BERT Fine-tuning 정확도")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    dataset = load_dataset("imdb")
    print(dataset)

    # IMDB는 label 순으로 정렬되어 있어 앞에서 잘라쓰면 전부 부정만 들어옴
    # → 반드시 shuffle 후 slicing
    train_shuffled = dataset["train"].shuffle(seed=SEED)
    test_shuffled = dataset["test"].shuffle(seed=SEED)

    train_texts = train_shuffled["text"][:2000]
    train_labels = train_shuffled["label"][:2000]
    test_texts = test_shuffled["text"][:500]
    test_labels = test_shuffled["label"][:500]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=SEED
    )
    print(
        f"훈련: {len(train_texts)}개 / 검증: {len(val_texts)}개 / "
        f"테스트: {len(test_texts)}개"
    )
    print(f"샘플: {train_texts[0][:80]}...")
    print(f"레이블: {'긍정' if train_labels[0] == 1 else '부정'}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = IMDBDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_ds = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    sample = train_ds[0]
    print(f"input_ids shape:      {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"label: {sample['label']}")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(
        device
    )
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {total_params:,}")
    print(f"학습 파라미터: {train_params:,}")

    # 기존 베스트 체크포인트가 있으면 이어서 시작
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(BEST_PATH):
        model.load_state_dict(torch.load(BEST_PATH, map_location=device))
        prev_val_loss, prev_val_acc = eval_epoch(model, val_loader, device)
        print(
            f"\n🔄 기존 체크포인트 로드: {BEST_PATH}"
            f"\n   (Val Acc: {prev_val_acc:.4f} / Val Loss: {prev_val_loss:.4f})"
        )
    else:
        prev_val_acc = 0.0

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"총 학습 스텝: {total_steps}")
    print(f"Warm-up 스텝: {warmup_steps}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = prev_val_acc  # 이전 베스트보다 좋아져야 갱신

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  ✅ 최고 모델 저장 (Val Acc: {val_acc:.4f})")

    model.load_state_dict(torch.load(BEST_PATH))
    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"\n최종 테스트 정확도: {test_acc:.4f}")

    plot_history(history)

    predict_sentiment(
        model,
        tokenizer,
        "This movie was absolutely brilliant! The acting was superb.",
        device,
    )
    predict_sentiment(
        model,
        tokenizer,
        "Worst movie I've ever seen. Complete waste of time.",
        device,
    )


if __name__ == "__main__":
    main()
