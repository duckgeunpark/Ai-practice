"""
[Step 5/5] 예측 & 평가
실행: python CNN_prediction_Keras.py
필요: data/data.npz, data/model_trained.keras (Step 1, 3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Step 1, 3 결과 확인
for f, step in [("data.npz", "CNN_dataset_Keras.py"),
                ("model_trained.keras", "CNN_compile_Keras.py")]:
    if not os.path.exists(os.path.join(DATA_DIR, f)):
        print(f"❌ data/{f} 가 없습니다. 먼저 {step} 를 실행하세요.")
        exit(1)

# 데이터 & 모델 로드
data = np.load(os.path.join(DATA_DIR, "data.npz"))
X_test, y_test = data["X_test"], data["y_test"]
model = load_model(os.path.join(DATA_DIR, "model_trained.keras"))
print("✔ 데이터 & 학습된 모델 로드 완료")

class_names = [
    "비행기", "자동차", "새", "고양이", "사슴",
    "개", "개구리", "말", "배", "트럭",
]

# 최종 정확도
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도: {test_acc:.4f}")

# 예측
y_pred = np.argmax(model.predict(X_test), axis=1)

# 분류 리포트
print(classification_report(y_test, y_pred, target_names=class_names))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("CIFAR-10 혼동 행렬")
plt.tight_layout()
plt.show()

# 예측 결과 시각화 (맞춘 것 / 틀린 것)
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i])
    pred = class_names[y_pred[i]]
    true = class_names[y_test[i]]
    color = "blue" if y_pred[i] == y_test[i] else "red"
    ax.set_title(f"예측: {pred}\n정답: {true}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("파란색: 정답  /  빨간색: 오답")
plt.tight_layout()
plt.show()
